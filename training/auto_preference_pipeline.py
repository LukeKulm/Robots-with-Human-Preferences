import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, PPOTrainer
from environments.py_bullet_blocks import RobotArmReachEnv
from reward_models.reward_predictor import RewardPredictor

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

def calculate_true_reward(trajectory):
    """Calculate true reward based on distance to target"""
    target_pos = np.array([0.5, 0.5, 0.5])  # Fixed target position
    total_reward = 0
    for obs in trajectory['observations']:
        # Extract end effector position from observation
        # Last 3 values in observation are target position, so we need to get the position before that
        end_effector_pos = obs[0]  # Assuming last 3 values before target are end effector position
        distance = np.linalg.norm(end_effector_pos - target_pos)
        total_reward -= distance  # Negative distance as reward
    return total_reward

def get_automated_preference(traj1, traj2):
    """Get preference based on true reward function"""
    reward1 = calculate_true_reward(traj1)
    reward2 = calculate_true_reward(traj2)
    
    if abs(reward1 - reward2) < 0.1:  # If rewards are very close
        return 0  # Equal
    elif reward1 > reward2:
        return 1  # Prefer trajectory 1
    else:
        return 2  # Prefer trajectory 2

class AutoPreferenceTrainer:
    def __init__(self, save_dir="saved_models"):
        self.env = RobotArmReachEnv()  # Use DIRECT mode for faster training
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        # Initialize PPO agent
        self.agent = PPOAgent(self.obs_dim, self.act_dim)
        
        # Initialize reward model with improved architecture
        self.reward_model = RewardPredictor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_dim=256,
            dropout_rate=0.2,
            l2_reg=1e-4
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(self.agent, self.env, self.reward_model)
        
        # Setup save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training buffer for preferences
        self.preference_buffer = []
        self.max_buffer_size = 1000  # Limit buffer size to prevent memory issues
    
    def collect_trajectories(self, n_trajectories=2):
        """Collect trajectories for preference learning"""
        trajectories = []
        for _ in range(n_trajectories):
            traj = self.ppo_trainer.collect_trajectory(max_steps=self.env.max_episode_steps)
            trajectories.append(traj)
        return trajectories
    
    def train_reward_model(self, n_epochs=10):
        """Train reward model on collected preferences"""
        if len(self.preference_buffer) == 0:
            print("No preferences to train on!")
            return
        
        # Use Adam optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            
            # Shuffle preferences
            indices = np.random.permutation(len(self.preference_buffer))
            
            for idx in indices:
                traj1, traj2, pref = self.preference_buffer[idx]
                if pref == 0:  # Skip if equal
                    continue
                
                # Compute loss using the new compute_loss method
                loss = self.reward_model.compute_loss(traj1, traj2, pref)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def train_step(self, n_trajectories=10, n_ppo_epochs=10):
        """Complete training step including PPO update"""
        trajectories = []
        for _ in range(n_trajectories):
            traj = self.ppo_trainer.collect_trajectory()
            advantages, returns = self.ppo_trainer.compute_advantages(
                traj['rewards'], traj['values'], traj['dones']
            )
            traj['advantages'] = advantages
            traj['returns'] = returns
            trajectories.append(traj)
        
        policy_loss, value_loss, entropy = self.ppo_trainer.update(trajectories, n_ppo_epochs)
        return policy_loss, value_loss, entropy
    
    def evaluate_performance(self, n_episodes=5):
        """Evaluate agent's performance over multiple episodes"""
        env = self.env
        agent = self.agent
        
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
    
    def add_preferences_to_buffer(self, trajectories):
        """Add preferences to buffer with size limit"""
        for i in range(0, len(trajectories), 2):
            if i + 1 < len(trajectories):
                pref = get_automated_preference(trajectories[i], trajectories[i+1])
                if pref != 0:  # Skip if trajectories are equal
                    self.preference_buffer.append((trajectories[i], trajectories[i+1], pref))
        
        # Limit buffer size
        if len(self.preference_buffer) > self.max_buffer_size:
            # Remove oldest preferences
            self.preference_buffer = self.preference_buffer[-self.max_buffer_size:]

def main():
    # Create directories
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Initialize trainer
    trainer = AutoPreferenceTrainer()
    performance_history = []
    reward_model_loss_history = []
    
    # Training parameters
    n_iterations = 200
    collect_preferences_every = 2
    n_trajectories_per_iter = 4  # Collect more trajectories for better preference learning
    
    print("Starting automated preference-based training...")
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        
        # Collect preferences periodically
        if iteration % collect_preferences_every == 0:
            print("Collecting automated preferences...")
            trajectories = trainer.collect_trajectories(n_trajectories=n_trajectories_per_iter)
            
            # Add preferences to buffer
            trainer.add_preferences_to_buffer(trajectories)
            
            print(f"Training reward model with {len(trainer.preference_buffer)} preferences...")
            trainer.train_reward_model(n_epochs=15)  # Increased epochs for better convergence
        
        # Train policy
        policy_loss, value_loss, entropy = trainer.train_step()
        print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Evaluate performance
        avg_reward = trainer.evaluate_performance()
        performance_history.append(avg_reward)
        print(f"Average Reward: {avg_reward:.2f}")
        
        # Save models periodically
        if (iteration + 1) % 10 == 0:
            torch.save(trainer.agent.state_dict(), 
                      os.path.join(trainer.save_dir, 'policy_model.pt'))
            torch.save(trainer.reward_model.state_dict(), 
                      os.path.join(trainer.save_dir, 'reward_model.pt'))
            
            # Update performance plot
            plot_performance(
                performance_history,
                os.path.join("training_plots", 
                            f'auto_preference_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            )
    
    print("\nTraining completed!")
    print(f"Final average reward: {performance_history[-1]:.2f}")

if __name__ == "__main__":
    main() 