import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(PPOAgent, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            mean = self.actor(obs)
            mean = self.mean_head(mean)
            
            if deterministic:
                return mean
            
            std = self.log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample()
            return action
    
    def evaluate_actions(self, obs, actions):
        mean = self.actor(obs)
        mean = self.mean_head(mean)
        std = self.log_std.exp()
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().mean()
        
        values = self.critic(obs).squeeze(-1)
        return log_probs, values, entropy

class PPOTrainer:
    def __init__(self, agent, env, reward_model=None, learning_rate=3e-4, clip_param=0.1,
                 value_loss_coef=0.5, entropy_coef=0.001, max_grad_norm=0.5):
        self.agent = agent
        self.env = env
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
        
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def collect_trajectory(self, max_steps=None):
        obs = self.env.reset()
        done = False
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Use environment's max_episode_steps if not specified
        if max_steps is None:
            max_steps = self.env.max_episode_steps
        
        for _ in range(max_steps):
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action = self.agent.get_action(obs_tensor)
                log_prob, value, _ = self.agent.evaluate_actions(obs_tensor.unsqueeze(0), 
                                                               action.unsqueeze(0))
            
            next_obs, reward, done, _ = self.env.step(action.numpy())
            
            # If using reward model, override environment reward
            if self.reward_model is not None:
                with torch.no_grad():
                    learned_reward = self.reward_model(
                        obs_tensor.unsqueeze(0).unsqueeze(1),  # → (B=1, T=1, obs_dim)
                        action.unsqueeze(0).unsqueeze(1)       # → (B=1, T=1, act_dim)
                    )
                reward = learned_reward.item()
            
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action.numpy())
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value.item())
            trajectory['log_probs'].append(log_prob.item())
            trajectory['dones'].append(done)
            
            obs = next_obs
            if done:
                break
        
        # Convert to numpy arrays
        for k, v in trajectory.items():
            trajectory[k] = np.array(v)
        
        return trajectory
    
    def compute_advantages(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advs = 0
        
        for t in reversed(range(len(rewards))):
            running_returns = rewards[t] + gamma * running_returns * (1 - dones[t])
            running_tderror = rewards[t] + gamma * previous_value * (1 - dones[t]) - values[t]
            running_advs = running_tderror + gamma * lambda_ * running_advs * (1 - dones[t])
            
            returns[t] = running_returns
            advantages[t] = running_advs
            previous_value = values[t]
        
        return advantages, returns
    
    def update(self, trajectories, epochs=10):
        observations = np.concatenate([t['observations'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        returns = np.concatenate([t['returns'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        old_log_probs = np.concatenate([t['log_probs'] for t in trajectories])
        
        observations = torch.FloatTensor(observations)
        actions = torch.FloatTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            log_probs, values, entropy = self.agent.evaluate_actions(observations, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        return policy_loss.item(), value_loss.item(), entropy.item()