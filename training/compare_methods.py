import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, PPOTrainer
from environments.py_bullet_blocks import RobotArmReachEnv
from reward_models.reward_predictor import RewardPredictor
from training.auto_preference_pipeline import AutoPreferenceTrainer
from training.trainer import RLHFTrainer
from inference.render_inference import render_trajectory_video

def calculate_true_reward(trajectory):
    """Calculate true reward based on distance to target"""
    target_pos = np.array([0.5, 0.5, 0.5])  # Fixed target position
    total_reward = 0
    for obs in trajectory['observations']:
        end_effector_pos = obs[14:17]  # Assuming last 3 values before target are end effector position
        distance = np.linalg.norm(end_effector_pos - target_pos)
        total_reward -= distance  # Negative distance as reward
    return total_reward

def evaluate_agent(env, agent, n_episodes=5):
    """Evaluate agent's performance over multiple episodes"""
    total_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action = agent.get_action(obs_tensor, deterministic=True)
            obs, reward, done, _ = env.step(action.numpy())
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

def train_ppo_with_rewards(n_iterations=100, n_trajectories_per_iter=20):
    """Train PPO with direct reward function"""
    env = RobotArmReachEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, act_dim)
    trainer = PPOTrainer(agent, env)
    
    rewards_history = []
    
    print("Starting PPO with rewards training...")
    for iteration in range(n_iterations):
        # Collect trajectories
        trajectories = []
        for _ in range(n_trajectories_per_iter):
            traj = trainer.collect_trajectory()
            advantages, returns = trainer.compute_advantages(
                traj['rewards'], traj['values'], traj['dones']
            )
            traj['advantages'] = advantages
            traj['returns'] = returns
            trajectories.append(traj)
        
        # Update policy
        policy_loss, value_loss, entropy = trainer.update(trajectories)
        
        # Evaluate using true reward
        eval_reward = evaluate_agent(env, agent)
        rewards_history.append(eval_reward)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}, Reward: {eval_reward:.2f}")
    
    return rewards_history

def train_auto_preference(n_iterations=100, n_trajectories_per_iter=4):
    """Train using automated preference collection"""
    trainer = AutoPreferenceTrainer()
    rewards_history = []
    
    print("Starting automated preference training...")
    for iteration in range(n_iterations):
        # Collect preferences periodically
        if iteration % 2 == 0:
            trajectories = trainer.collect_trajectories(n_trajectories=n_trajectories_per_iter)
            trainer.add_preferences_to_buffer(trajectories)
            trainer.train_reward_model(n_epochs=15)
        
        # Train policy
        policy_loss, value_loss, entropy = trainer.train_step()
        
        # Evaluate using true reward
        eval_reward = evaluate_agent(trainer.env, trainer.agent)
        rewards_history.append(eval_reward)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}, Reward: {eval_reward:.2f}")
    
    return rewards_history

def train_human_preference(n_iterations=100, n_trajectories_per_iter=4):
    """Train using human preference collection"""
    trainer = RLHFTrainer()
    rewards_history = []
    
    print("Starting human preference training...")
    for iteration in range(n_iterations):
        # Collect preferences periodically
        if iteration % 2 == 0:
            # Only collect 2 trajectories for comparison
            trajectories = trainer.collect_trajectories(n_trajectories=2)
            
            # Render only the two trajectories for comparison
            for i, traj in enumerate(trajectories):
                prefix = f"{'left' if i == 0 else 'right'}_clip_{iteration:03d}"
                print(f"Rendering trajectory {i+1} for preference collection...")
                render_trajectory_video(traj, trainer.env, filename_prefix=prefix)
            
            print("\nLaunching labeling UI...")
            # Launch Streamlit in a separate process
            subprocess.Popen(["streamlit", "run", "scripts/visualize_trajectories.py"])
            
            # Wait for user input before proceeding
            input("Please label the trajectories in the Streamlit UI. Press Enter when done...")
            
            # Load preferences from the saved file
            trainer.load_preferences_into_buffer()
            trainer.train_reward_model()
        
        # Train policy
        policy_loss, value_loss, entropy = trainer.train_step()
        
        # Evaluate using true reward
        eval_reward = evaluate_agent(trainer.env, trainer.agent)
        rewards_history.append(eval_reward)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}, Reward: {eval_reward:.2f}")
    
    return rewards_history

def plot_comparison(ppo_rewards, auto_rewards, human_rewards, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_rewards, label='PPO with Rewards', color='blue')
    plt.plot(auto_rewards, label='Automated Preferences', color='green')
    plt.plot(human_rewards, label='Human Preferences', color='red')
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Reward')
    plt.title('Comparison of Training Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Create directories
    os.makedirs("training_plots", exist_ok=True)
    
    # Training parameters
    n_iterations = 100
    n_trajectories_per_iter = 20
    
    print("Starting comparison of training methods...")
    
    # First: Train using human preference collection
    print("\n=== Starting Human Preference Training ===")
    human_rewards = train_human_preference(n_iterations, n_trajectories_per_iter)
    
    # Second: Train using automated preference collection
    print("\n=== Starting Automated Preference Training ===")
    auto_rewards = train_auto_preference(n_iterations, n_trajectories_per_iter)
    
    # Third: Train using PPO with direct rewards
    print("\n=== Starting PPO with Rewards Training ===")
    ppo_rewards = train_ppo_with_rewards(n_iterations, n_trajectories_per_iter)
    
    # Plot comparison
    plot_comparison(
        ppo_rewards,
        auto_rewards,
        human_rewards,
        os.path.join("training_plots", f'method_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    )
    
    print("\nTraining comparison completed!")
    print(f"Final Human Preference reward: {human_rewards[-1]:.2f}")
    print(f"Final Auto Preference reward: {auto_rewards[-1]:.2f}")
    print(f"Final PPO reward: {ppo_rewards[-1]:.2f}")

if __name__ == "__main__":
    main() 