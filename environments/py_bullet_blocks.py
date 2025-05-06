import pybullet as p
import pybullet_data
import time
import numpy as np
import gym
from gym import spaces

class RobotArmReachEnv(gym.Env):
    def __init__(self):
        super(RobotArmReachEnv, self).__init__()
        
        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set up robot arm
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", robot_start_pos, robot_start_orientation, useFixedBase=True)
        
        # Set up target visualization (small red sphere)
        self.target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7])
        self.target_pos = None
        self.target_id = None
        
        # Define action and observation spaces
        # Action space: 7 joint positions for KUKA IIWA
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Observation space: current joint positions (7) + joint velocities (7) + target position (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        
        #Define max episode steps
        self.max_episode_steps = 100
        self.current_step = 0
        
        # Set up for real-time simulation
        p.setRealTimeSimulation(0)
        
    def reset(self):
        self.current_step = 0
        # Reset joint positions to default
        for i in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, i, 0)
        
        # Generate new random target position
        self.target_pos = self._generate_random_target()
        
        # Update target visualization
        if self.target_id is not None:
            p.removeBody(self.target_id)
        self.target_id = p.createMultiBody(baseMass=0,
                                         baseVisualShapeIndex=self.target_visual_shape,
                                         basePosition=self.target_pos)
        
        return self._get_observation()
    
    def step(self, action):
        self.current_step += 1
        # Scale actions from [-1, 1] to actual joint limits
        scaled_action = np.array(action) * 2.9671  # KUKA joint limits
        
        # Apply action to each joint
        for i in range(7):
            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=scaled_action[i],
                                  force=500)
        
        # Step simulation
        p.stepSimulation()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check if done
        done = self._is_done() or self.current_step >= self.max_episode_steps
        
        return obs, reward, done, {}
    
    def _get_observation(self):
        joint_states = []
        for i in range(7):
            state = p.getJointState(self.robot_id, i)
            joint_states.extend([state[0], state[1]])  # position and velocity
        
        return np.array(joint_states + list(self.target_pos))
    
    def _compute_reward(self):
        # Get end effector position
        state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = state[0]
        
        # Calculate distance to target
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_pos))
        
        # Reward is negative distance (closer is better)
        reward = -distance
        
        return reward
    
    def _is_done(self):
        # Get end effector position
        state = p.getLinkState(self.robot_id, 6)
        end_effector_pos = state[0]
        
        # Check if we're close enough to target
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(self.target_pos))
        return distance < 0.05  # 5cm threshold
    
    def _generate_random_target(self):
        # Generate random target position within reasonable workspace
        x = np.random.uniform(0.2, 0.8)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0.2, 0.7)
        return [x, y, z]
    
    def render(self, mode='human'):
        pass  # PyBullet already handles rendering
    
    def close(self):
        p.disconnect(self.physics_client)

def main():
    # Create and test the environment
    env = RobotArmReachEnv()
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
