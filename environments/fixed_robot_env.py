import pybullet as p
import pybullet_data
import time
import numpy as np
import gym
from gym import spaces
import os

DEFAULT_GOAL_POS = [0.5, 0.0, 0.5]


class LockedRobotArmReachEnv(gym.Env):
    def __init__(self, gui=False, controlled_joints=1, controlled_joint_indices=[1], goal_position=DEFAULT_GOAL_POS):
        super(LockedRobotArmReachEnv, self).__init__()

        self.target_pos = goal_position if goal_position else DEFAULT_GOAL_POS

        # Connect PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load scene
        self.plane_id = p.loadURDF("plane.urdf")
        robot_start_pos = [0, 0, 0]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        panda_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        if not os.path.exists(panda_path):
            raise FileNotFoundError(f"Panda URDF not found at {panda_path}")

        self.robot_id = p.loadURDF(panda_path, robot_start_pos, robot_start_orientation, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)

        # Discover actuated joints
        actuated_candidates = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            if joint_type == p.JOINT_REVOLUTE and 'finger' not in joint_name and 'hand' not in joint_name:
                actuated_candidates.append(i)

        # Validate user input
        if controlled_joint_indices is not None:
            self.controlled_joints = controlled_joint_indices
        else:
            assert 1 <= controlled_joints <= len(actuated_candidates), "Invalid number of controlled joints"
            self.controlled_joints = actuated_candidates[:controlled_joints]

        self.joint_limits = []
        for joint_idx in self.controlled_joints:
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            self.joint_limits.append((joint_info[8], joint_info[9]))

        # Lock the rest
        self.locked_joints = [j for j in actuated_candidates if j not in self.controlled_joints]
        for i in self.locked_joints:
            state = p.getJointState(self.robot_id, i)[0]
            p.setJointMotorControl2(self.robot_id, i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=state,
                                    force=1e10)

        # Setup visuals
        self.target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.7])
        self.target_id = None

        # Define spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.controlled_joints),), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * len(self.controlled_joints) + 3 + 3,),  # joint pos/vel + ee pos + target
            dtype=np.float32
        )

        self.max_episode_steps = 50
        self.current_step = 0

        p.setRealTimeSimulation(0)

    def reset(self):
        self.current_step = 0
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0.0)

        # Re-lock joints
        for i in self.locked_joints:
            state = p.getJointState(self.robot_id, i)[0]
            p.setJointMotorControl2(self.robot_id, i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=state,
                                    force=1e10)

        # Create/replace target marker
        if self.target_id is not None:
            p.removeBody(self.target_id)
        self.target_id = p.createMultiBody(baseMass=0,
                                           baseVisualShapeIndex=self.target_visual_shape,
                                           basePosition=self.target_pos)

        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)

        locked_joint_angles_before = [p.getJointState(self.robot_id, j)[0] for j in self.locked_joints]

        for i, joint_idx in enumerate(self.controlled_joints):
            scaled_pos = action[i] * 2.9671
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=scaled_pos,
                force=500
            )

        p.stepSimulation()

        # Check locked joints are static
        locked_joint_angles_after = [p.getJointState(self.robot_id, j)[0] for j in self.locked_joints]
        for j, before, after in zip(self.locked_joints, locked_joint_angles_before, locked_joint_angles_after):
            delta = abs(after - before)
            if delta > 1e-3:
                print(f"Locked joint {j} moved! delta={delta:.4f} radians")

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done() or self.current_step >= self.max_episode_steps

        return obs, reward, done, {"distance": np.linalg.norm(self.ee_pos - self.target_pos)}

    def _get_observation(self):
        joint_obs = []
        for joint_idx in self.controlled_joints:
            state = p.getJointState(self.robot_id, joint_idx)
            joint_obs.extend([state[0], state[1]])

        ee_state = p.getLinkState(self.robot_id, 7)
        self.ee_pos = np.array(ee_state[0])

        return np.array(joint_obs + list(self.ee_pos) + list(self.target_pos), dtype=np.float32)

    def _compute_reward(self):
        return -np.linalg.norm(self.ee_pos - self.target_pos)

    def _is_done(self):
        return np.linalg.norm(self.ee_pos - self.target_pos) < 0.05

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            width, height, view_matrix, proj_matrix = 320, 240, None, None

            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1, 1, 1],
                cameraTargetPosition=[0, 0, 0.4],
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width)/height,
                nearVal=0.1, farVal=3.1
            )

            _, _, px, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            rgb_array = np.reshape(px, (height, width, 4))[:, :, :3]
            return rgb_array
        else:
            return super().render(mode=mode)

    def close(self):
        p.disconnect(self.physics_client)


if __name__ == '__main__':
    env = LockedRobotArmReachEnv(gui=True, controlled_joint_indices=[1], goal_position=DEFAULT_GOAL_POS)

    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(1. / 240.)
        if done:
            print("Target reached or max steps exceeded. Resetting.")
            obs = env.reset()

    env.close()
