import torch
import torch.nn as nn
import numpy as np
import time
import os
import streamlit as st
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo import PPOAgent, PPOTrainer
from environments.py_bullet_blocks import RobotArmReachEnv
from reward_models.reward_predictor import RewardPredictor

class RLHFTrainer:
    def __init__(self, save_dir="saved_models"):
        self.env = RobotArmReachEnv()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        # Initialize PPO agent
        self.agent = PPOAgent(self.obs_dim, self.act_dim)
        
        # Initialize reward model
        self.reward_model = RewardPredictor(self.obs_dim, self.act_dim)
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(self.agent, self.env, self.reward_model)
        
        # Setup save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training buffer for preferences
        self.preference_buffer = []
    
    def collect_trajectories(self, n_trajectories=2):
        """Collect trajectories for preference learning"""
        trajectories = []
        for _ in range(n_trajectories):
            traj = self.ppo_trainer.collect_trajectory()
            trajectories.append(traj)
        return trajectories
    
    def get_human_preference(self, traj1, traj2):
        """Get human preference between two trajectories using Streamlit"""
        st.title("Human Preference Interface")
        
        # Display trajectory information
        st.write("Trajectory 1:")
        st.write(f"Total reward: {sum(traj1['rewards']):.2f}")
        st.write(f"Length: {len(traj1['rewards'])}")
        
        st.write("\nTrajectory 2:")
        st.write(f"Total reward: {sum(traj2['rewards']):.2f}")
        st.write(f"Length: {len(traj2['rewards'])}")
        
        # Get preference
        preference = st.radio(
            "Which trajectory was better?",
            ["Trajectory 1", "Trajectory 2", "Equal", "Both Bad"]
        )
        
        if st.button("Submit Preference"):
            if preference == "Trajectory 1":
                return 1
            elif preference == "Trajectory 2":
                return 2
            elif preference == "Equal":
                return 0
            else:  # Both Bad
                return -1
        
        return None
    
    def train_reward_model(self, n_epochs=10):
        """Train reward model on collected preferences"""
        if len(self.preference_buffer) == 0:
            print("No preferences to train on!")
            return
        
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        
        for epoch in range(n_epochs):
            total_loss = 0
            for traj1, traj2, pref in self.preference_buffer:
                if pref in [-1, 0]:  # Skip if both bad or equal
                    continue
                    
                # Convert trajectories to tensors
                obs1 = torch.FloatTensor(traj1['observations'])
                act1 = torch.FloatTensor(traj1['actions'])
                obs2 = torch.FloatTensor(traj2['observations'])
                act2 = torch.FloatTensor(traj2['actions'])
                
                # Compute rewards
                r1 = self.reward_model(obs1, act1).sum()
                r2 = self.reward_model(obs2, act2).sum()
                
                # Compute loss (higher reward for preferred trajectory)
                if pref == 1:
                    loss = torch.nn.functional.softplus(r2 - r1)
                else:  # pref == 2
                    loss = torch.nn.functional.softplus(r1 - r2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(self.preference_buffer):.4f}")
    
    def train_step(self, n_trajectories=10, n_ppo_epochs=10):
        """Complete training step including PPO update"""
        # Collect trajectories
        trajectories = []
        for _ in range(n_trajectories):
            traj = self.ppo_trainer.collect_trajectory()
            
            # Compute advantages and returns
            advantages, returns = self.ppo_trainer.compute_advantages(
                traj['rewards'], traj['values'], traj['dones']
            )
            traj['advantages'] = advantages
            traj['returns'] = returns
            
            trajectories.append(traj)
        
        # Update PPO
        policy_loss, value_loss, entropy = self.ppo_trainer.update(trajectories, n_ppo_epochs)
        return policy_loss, value_loss, entropy
    
    def save_models(self):
        """Save both policy and reward models"""
        torch.save(self.agent.state_dict(), 
                  os.path.join(self.save_dir, 'policy_model.pt'))
        torch.save(self.reward_model.state_dict(), 
                  os.path.join(self.save_dir, 'reward_model.pt'))
    
    def load_models(self):
        """Load both policy and reward models"""
        self.agent.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'policy_model.pt')))
        self.reward_model.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'reward_model.pt')))

def main():
    # Initialize trainer
    trainer = RLHFTrainer()
    
    # Training loop
    n_iterations = 10 # 100
    collect_preferences_every = 5
    
    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}")
        
        # Collect human preferences periodically
        if iteration % collect_preferences_every == 0:
            print("Collecting human preferences...")
            trajectories = trainer.collect_trajectories(n_trajectories=2)
            pref = trainer.get_human_preference(trajectories[0], trajectories[1])
            
            if pref is not None:
                trainer.preference_buffer.append((trajectories[0], trajectories[1], pref))
                trainer.train_reward_model()
        
        # Train policy
        policy_loss, value_loss, entropy = trainer.train_step()
        print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
        
        # Save models periodically
        if (iteration + 1) % 10 == 0:
            trainer.save_models()

if __name__ == "__main__":
    main()
