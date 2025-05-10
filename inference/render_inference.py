import os
import torch
import numpy as np
import cv2
import pybullet as p
import subprocess
from pathlib import Path

from agents.ppo import PPOAgent
from environments.py_bullet_blocks import RobotArmReachEnv

def load_agent(model_path, obs_dim, act_dim):
    agent = PPOAgent(obs_dim, act_dim)
    agent.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Checkpoint Loaded: {model_path}")
    agent.eval()
    return agent


def convert_to_h264(video_path):
    try:
        h264_path = video_path.replace(".mp4", "_h264.mp4")
        # Use subprocess.run instead of os.system for better error handling
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vcodec", "libx264", "-crf", "23", h264_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error converting video: {result.stderr}")
            return False
            
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rename(h264_path, video_path)
        return True
    except Exception as e:
        print(f"Error in video conversion: {str(e)}")
        return False


def render_trajectory_video(trajectory, env, filename_prefix="clip"):
    try:
        width, height = 320, 240
        
        # Create directories with proper error handling
        clips_dir = Path("data/clips")
        trajectories_dir = Path("data/trajectories")
        
        clips_dir.mkdir(parents=True, exist_ok=True)
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = str(clips_dir / f"{filename_prefix}.mp4")
        data_path = str(trajectories_dir / f"{filename_prefix}.npz")

        # Try different codecs if mp4v fails
        codecs = ['mp4v', 'avc1', 'XVID']
        video_writer = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                if video_writer.isOpened():
                    break
            except Exception as e:
                print(f"Failed to initialize video writer with codec {codec}: {str(e)}")
                continue
        
        if video_writer is None or not video_writer.isOpened():
            raise Exception("Failed to initialize video writer with any codec")

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
        
        # Only attempt conversion if the video was written successfully
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            if not convert_to_h264(video_path):
                print(f"Warning: Could not convert video to H.264, keeping original format")
        else:
            raise Exception("Video file was not created or is empty")
            
        # Save trajectory data
        np.savez(data_path, obs=trajectory['observations'], act=trajectory['actions'])
        print(f"🎥 Saved: {video_path}")
        
    except Exception as e:
        print(f"Error in render_trajectory_video: {str(e)}")
        if video_writer is not None:
            video_writer.release()
        raise


def render_trajectory(env, agent, filename_prefix="inference", max_steps=None):
    if max_steps is None:
        max_steps = env.max_episode_steps
        
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
