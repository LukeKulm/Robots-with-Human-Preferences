import torch
import torch.nn as nn
import torch.nn.functional as F


def preference_loss(reward_pred, traj1, traj2, label):
    """
    reward_pred: RewardPredictor
    traj1, traj2: dicts with 'obs' and 'act' keys, shape (B, T, dim)
    label: tensor of 0 (prefer traj1), 1 (prefer traj2), or 0.5 (tie)
    """
    r1 = reward_pred(traj1['obs'], traj1['act']).sum(dim=1)  # (B, 1)
    r2 = reward_pred(traj2['obs'], traj2['act']).sum(dim=1)  # (B, 1)

    logits = torch.cat([r1, r2], dim=1)  # (B, 2)
    probs = F.softmax(logits, dim=1)

    labels = torch.zeros_like(probs)
    labels[range(len(label)), label.long()] = 1.0

    return F.binary_cross_entropy(probs, labels)


class RewardPredictor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
        super().__init__()
        input_dim = obs_dim + act_dim
        layers = []
        last_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, 1))  # scalar reward output
        self.model = nn.Sequential(*layers)

    def forward(self, observations, actions):
        """
        Inputs:
          - observations: (B, T, obs_dim)
          - actions: (B, T, act_dim)
        Returns:
          - rewards: (B, T, 1)
        """
        x = torch.cat([observations, actions], dim=-1)  # (B, T, obs+act)
        B, T, D = x.shape
        x = x.view(B*T, D)
        rewards = self.model(x)  # (B*T, 1)
        return rewards.view(B, T, 1)

