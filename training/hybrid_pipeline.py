import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import subprocess
from training.trainer import RLHFTrainer
from training.human_feedback_collector import convert_preferences_to_dataset
from inference.render_inference import render_trajectory_video
from agents.ppo import PPOTrainer
from training.train_ppo import evaluate_agent


def plot_performance(performance_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(performance_history, label='Average Distance to Target')
    plt.xlabel('Iteration')
    plt.ylabel('Distance to Target')
    plt.title('Agent Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def calculate_baseline_reward(trajectory):
    """Calculate baseline reward based on distance to target"""
    target_pos = np.array([0.5, 0.5, 0.5])
    distances = []
    for obs in trajectory['observations']:
        end_effector_pos = obs[14:17]  # Ensure this matches your env's observation format
        distance = np.linalg.norm(end_effector_pos - target_pos)
        distances.append(distance)
    return -np.mean(distances)

def main():
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    
    # ✅ FIX: Re-initialize PPOTrainer exactly like in standalone PPO file
    from agents.ppo import PPOAgent
    from environments.py_bullet_blocks import RobotArmReachEnv
    from environments.fixed_robot_env import LockedRobotArmReachEnv
    
    env = RobotArmReachEnv()
    # env = LockedRobotArmReachEnv()
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim)
    ppo_trainer = PPOTrainer(agent, env)
    
    trainer = RLHFTrainer(env, ppo_trainer, save_dir="saved_models")

    performance_history = []
    total_iterations = 200
    ppo_only_iterations = 100
    ask_every = 2

    print("Starting hybrid training pipeline...")
    print(f"First {ppo_only_iterations} iterations will use normal PPO training")

    for iter in range(total_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iter + 1}/{total_iterations}")
        print(f"{'='*50}")
        
        if iter < ppo_only_iterations:
            print("\n[PPO Training Phase]")
            trajectories = []
            for _ in range(20):
                traj = ppo_trainer.collect_trajectory()
                advantages, returns = ppo_trainer.compute_advantages(
                    traj['rewards'], traj['values'], traj['dones']
                )
                traj['advantages'] = advantages
                traj['returns'] = returns
                trajectories.append(traj)

            policy_loss, value_loss, entropy = ppo_trainer.update(trajectories, epochs=5)
            
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        else:
            if iter % ask_every == 0:
                print("\n[Preference Collection Phase]")
                pref_trajectories = trainer.collect_trajectories(n_trajectories=2)
                for i, traj in enumerate(pref_trajectories):
                    prefix = f"{'left' if i == 0 else 'right'}_clip_{iter:03d}"
                    print(f"Rendering trajectory {i+1} for preference collection...")
                    render_trajectory_video(traj, trainer.env, filename_prefix=prefix)
                
                print("\nLaunching labeling UI...")
                subprocess.Popen(["streamlit", "run", "scripts/visualize_trajectories.py"])
                input("Please label the trajectories in the Streamlit UI. Press Enter when done...")

                print("\n[Reward Model Training Phase]")
                convert_preferences_to_dataset()
                trainer.load_preferences_into_buffer()
                trainer.train_reward_model()
            
            print("\n[PPO Training Phase with Reward Model]")
            policy_loss, value_loss, entropy = trainer.train_step(n_trajectories=20, n_ppo_epochs=3)
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")

        # eval_trajectory = trainer.collect_trajectories(n_trajectories=1)[0]
        # baseline_reward = calculate_baseline_reward(eval_trajectory)
        # performance_history.append(baseline_reward)
        # print(f"Current Performance (Distance to Target): {-baseline_reward:.4f}")
        
        avg_reward = evaluate_agent(trainer.env, agent)
        performance_history.append(avg_reward)

        if (iter + 1) % 1 == 0:
            trainer.save_models()
            print("\n[Model Saving] Models saved")
            plot_performance(
                performance_history, 
                os.path.join("training_plots", f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            )
            print("Performance plot updated")

if __name__ == "__main__":
    main()
