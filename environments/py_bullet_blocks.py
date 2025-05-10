import pybullet as p
import pybullet_data
import time
import numpy as np
import gym
from gym import spaces
import os

FIXED_GOAL_POS = [0.5, 0.5, 0.5]

class RobotArmReachEnv(gym.Env):
    def __init__(self, gui=False):
        super(RobotArmReachEnv, self).__init__()
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set up robot arm
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Get the path to the Panda URDF
        panda_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        if not os.path.exists(panda_path):
            raise FileNotFoundError(f"Panda URDF not found at {panda_path}")
        
        # Load the robot with error checking
        self.robot_id = p.loadURDF(panda_path, robot_start_pos, robot_start_orientation, useFixedBase=True)
        if self.robot_id is None:
            raise RuntimeError("Failed to load Panda robot model")
        
        # Get all joints and identify actuated joints
        self.num_joints = p.getNumJoints(self.robot_id)
        self.actuated_joints = []
        self.joint_limits = []
        
        print(f"Robot ID: {self.robot_id}")
        print(f"Total number of joints: {self.num_joints}")
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Only include revolute joints (type 0) that are not part of the gripper
            if joint_type == p.JOINT_REVOLUTE and 'finger' not in joint_name and 'hand' not in joint_name:
                self.actuated_joints.append(i)
                self.joint_limits.append((joint_info[8], joint_info[9]))  # lower and upper limits
                print(f"Actuated joint {i}: {joint_name}")
        
        if len(self.actuated_joints) != 7:
            raise RuntimeError(f"Expected 7 actuated joints for Panda robot, got {len(self.actuated_joints)}")
        
        # Set up target visualization (small red sphere)
        self.target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7])
        self.target_pos = None
        self.target_id = None
        
        # Define action and observation spaces
        # Action space: 7 joint positions for Panda
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Observation space: current joint positions (7) + joint velocities (7) + target position (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        # Define max episode steps
        self.max_episode_steps = 50
        self.current_step = 0
        
        # Set up for real-time simulation
        p.setRealTimeSimulation(0)
    
    def reset(self):
        self.current_step = 0
        
        # Reset joint positions to default with error checking
        for joint_idx in self.actuated_joints:
            try:
                p.resetJointState(self.robot_id, joint_idx, 0)
            except p.error as e:
                print(f"Warning: Failed to reset joint {joint_idx}: {e}")
                # Try to reload the robot if joint reset fails
                self._reload_robot()
                break
        
        # Set fixed target position
        self.target_pos = FIXED_GOAL_POS
        
        # Update target visualization
        if self.target_id is not None:
            p.removeBody(self.target_id)
        self.target_id = p.createMultiBody(baseMass=0,
                                         baseVisualShapeIndex=self.target_visual_shape,
                                         basePosition=self.target_pos)
        
        return self._get_observation()
    
    def _reload_robot(self):
        """Reload the robot if it gets into a bad state"""
        # Remove old robot
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
        
        # Reload robot
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                 robot_start_pos, 
                                 robot_start_orientation, 
                                 useFixedBase=True)
        
        # Re-identify actuated joints
        self.actuated_joints = []
        self.joint_limits = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE and 'finger' not in joint_name and 'hand' not in joint_name:
                self.actuated_joints.append(i)
                self.joint_limits.append((joint_info[8], joint_info[9]))
    
    def step(self, action):
        self.current_step += 1
        
        # Scale actions from [-1, 1] to actual joint limits
        scaled_action = np.array(action) * 2.9671  # Panda joint limits
        
        # Apply action to each actuated joint with error checking
        for i, joint_idx in enumerate(self.actuated_joints):
            try:
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                      jointIndex=joint_idx,
                                      controlMode=p.POSITION_CONTROL,
                                      targetPosition=scaled_action[i],
                                      force=500)
            except p.error as e:
                print(f"Warning: Failed to set joint {joint_idx}: {e}")
                self._reload_robot()
                return self._get_observation(), -1.0, True, {"error": "joint_control_failed"}
        
        # Step simulation
        p.stepSimulation()
        
        # Get new observation
        try:
            obs = self._get_observation()
        except p.error as e:
            print(f"Warning: Failed to get observation: {e}")
            self._reload_robot()
            return np.zeros(self.observation_space.shape), -1.0, True, {"error": "observation_failed"}
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check if done
        done = self._is_done() or self.current_step >= self.max_episode_steps
        
        return obs, reward, done, {}
    
    def _get_observation(self):
        joint_states = []
        for joint_idx in self.actuated_joints:
            try:
                state = p.getJointState(self.robot_id, joint_idx)
                joint_states.extend([state[0], state[1]])  # position and velocity
            except p.error as e:
                print(f"Warning: Failed to get joint state for joint {joint_idx}: {e}")
                self._reload_robot()
                raise
        
        return np.array(joint_states + list(self.target_pos))
    
    def _compute_reward(self):
        try:
            # Get end effector position
            state = p.getLinkState(self.robot_id, 7)  # Panda's end effector is link 7
            end_effector_pos = state[0]
            
            # Calculate distance to target
            distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_pos))
            
            # Reward is negative distance (closer is better)
            reward = -distance
            
            return reward
        except p.error as e:
            print(f"Warning: Failed to compute reward: {e}")
            return -1.0
    
    def _is_done(self):
        try:
            # Get end effector position
            state = p.getLinkState(self.robot_id, 7)  # Panda's end effector is link 7
            end_effector_pos = state[0]
            
            # Check if we're close enough to target
            distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_pos))
            return distance < 0.05  # 5cm threshold
        except p.error as e:
            print(f"Warning: Failed to check done condition: {e}")
            return True
    
    def render(self, mode='human'):
        pass  # PyBullet already handles rendering
    
    def close(self):
        p.disconnect(self.physics_client)

def main():
    # Create and test the environment
    env = RobotArmReachEnv(gui=True)
    obs = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        time.sleep(1./240)  # Run at 240 Hz
        
        if done:
            print("Target reached!")
            obs = env.reset()
    
    env.close()

if __name__ == '__main__':
    main()
