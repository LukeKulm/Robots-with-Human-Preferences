import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, PPOTrainer
from environments.py_bullet_blocks import RobotArmReachEnv

def plot_learning_curve(rewards_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label='Average Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('PPO Learning Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

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

def main():
    # Create directories
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Initialize environment and agent
    env = RobotArmReachEnv()  # Use DIRECT mode for faster training
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, act_dim)
    trainer = PPOTrainer(agent, env)
    
    # Training parameters
    n_iterations = 500
    n_trajectories_per_iter = 20
    n_ppo_epochs = 5
    eval_every = 20
    
    # Training loop
    rewards_history = []
    best_reward = float('-inf')
    
    print("Starting PPO training...")
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
        policy_loss, value_loss, entropy = trainer.update(trajectories, n_ppo_epochs)
        
        # Evaluate agent
        if iteration % eval_every == 0:
            avg_reward = evaluate_agent(env, agent)
            rewards_history.append(avg_reward)
            print(f"Iteration {iteration + 1}/{n_iterations}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.state_dict(), "saved_models/best_ppo_model.pt")
                print(f"New best model saved! Reward: {best_reward:.2f}")
            
            # Update plot
            plot_learning_curve(
                rewards_history,
                f"training_plots/ppo_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
    
    print("\nTraining completed!")
    print(f"Best average reward achieved: {best_reward:.2f}")
    
    # Save final model
    torch.save(agent.state_dict(), "saved_models/final_ppo_model.pt")
    
    # Final evaluation
    final_reward = evaluate_agent(env, agent, n_episodes=10)
    print(f"Final evaluation reward: {final_reward:.2f}")

if __name__ == "__main__":
    main() 