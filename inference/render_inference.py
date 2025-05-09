import os
import torch
import numpy as np
import cv2
import pybullet as p

from agents.ppo import PPOAgent
from environments.py_bullet_blocks import RobotArmReachEnv

def load_agent(model_path, obs_dim, act_dim):
    agent = PPOAgent(obs_dim, act_dim)
    agent.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Checkpoint Loaded: {model_path}")
    agent.eval()
    return agent


def convert_to_h264(video_path):
    h264_path = video_path.replace(".mp4", "_h264.mp4")
    os.system(f"ffmpeg -y -i {video_path} -vcodec libx264 -crf 23 {h264_path}")
    os.remove(video_path)
    os.rename(h264_path, video_path)


def render_trajectory_video(trajectory, env, filename_prefix="clip"):
    width, height = 320, 240
    video_path = f"data/clips/{filename_prefix}.mp4"
    data_path = f"data/trajectories/{filename_prefix}.npz"
    os.makedirs("data/clips", exist_ok=True)
    os.makedirs("data/trajectories", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    env.reset()
    for action in trajectory["actions"]:
        p.stepSimulation()
        env.step(action)

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.5],
            distance=1.0,
            yaw=50,
            pitch=-35,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100.0
        )
        _, _, px, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix,
                                          renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                          flags=p.ER_NO_SEGMENTATION_MASK)
        rgba = np.array(px, dtype=np.uint8).reshape((height, width, 4))
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr)

    video_writer.release()
    convert_to_h264(video_path)
    np.savez(data_path, obs=trajectory['observations'], act=trajectory['actions'])
    print(f"🎥 Saved: {video_path}")


# def render_trajectory(env, agent, filename_prefix="inference", max_steps=100):
#     width, height = 320, 240
#     video_path = f"inference/inference_videos/{filename_prefix}.mp4"
#     os.makedirs("inference/inference_videos", exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

#     obs = env.reset()
#     for step in range(max_steps):
#         obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
#         action = agent.get_action(obs_tensor, deterministic=True).squeeze(0).numpy()

#         obs, reward, done, _ = env.step(action)

#         view_matrix = p.computeViewMatrixFromYawPitchRoll(
#             cameraTargetPosition=[0.5, 0, 0.5],
#             distance=1.0,
#             yaw=50,
#             pitch=-35,
#             roll=0,
#             upAxisIndex=2
#         )
#         proj_matrix = p.computeProjectionMatrixFOV(
#             fov=60,
#             aspect=width / height,
#             nearVal=0.1,
#             farVal=100.0
#         )
#         _, _, px, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix,
#                                           renderer=p.ER_BULLET_HARDWARE_OPENGL,
#                                           flags=p.ER_NO_SEGMENTATION_MASK)
#         rgba = np.array(px, dtype=np.uint8).reshape((height, width, 4))
#         rgb = np.ascontiguousarray(rgba[:, :, :3])
#         bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#         video_writer.write(bgr)

#         if done:
#             break

#     video_writer.release()
#     print(f"🎥 Saved video to: {video_path}")

def render_trajectory(env, agent, filename_prefix="inference", max_steps=100):
    obs = env.reset()
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": []
    }

    for step in range(max_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = agent.get_action(obs_tensor, deterministic=True).squeeze(0).numpy()

        next_obs, reward, done, _ = env.step(action)

        # Save step info
        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["dones"].append(done)

        obs = next_obs
        if done:
            break

    # Convert to numpy arrays
    for k in trajectory:
        trajectory[k] = np.array(trajectory[k])

    # Use existing video rendering logic
    render_trajectory_video(trajectory, env, filename_prefix)


if __name__ == "__main__":
    env = RobotArmReachEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = load_agent("saved_models/policy_model.pt", obs_dim, act_dim)
    render_trajectory(env, agent, filename_prefix="policy_run")
    env.close()
