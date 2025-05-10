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
    def __init__(self, obs_dim, act_dim, hidden_dim=64, dropout_rate=0.2, l2_reg=1e-4):
        super(RewardPredictor, self).__init__()
        
        # Input dimensions
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        # Simple MLP architecture using LayerNorm instead of BatchNorm
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, obs, act):
        # Reshape inputs if they're batched
        if len(obs.shape) == 3:  # (B, T, D)
            batch_size, seq_len, _ = obs.shape
            obs = obs.reshape(-1, self.obs_dim)
            act = act.reshape(-1, self.act_dim)
        
        # Concatenate observations and actions
        x = torch.cat([obs, act], dim=-1)
        
        # Predict reward
        reward = self.model(x)
        
        # Reshape back if input was batched
        if len(obs.shape) == 3:
            reward = reward.reshape(batch_size, seq_len, -1)
        
        return reward
    
    def compute_loss(self, traj1, traj2, preference):
        """
        Compute loss with L2 regularization
        traj1, traj2: dictionaries containing 'observations' and 'actions'
        preference: 1 for traj1 preferred, 2 for traj2 preferred
        """
        # Convert to tensors
        obs1 = torch.FloatTensor(traj1['obs']).unsqueeze(0)
        act1 = torch.FloatTensor(traj1['act']).unsqueeze(0)
        obs2 = torch.FloatTensor(traj2['obs']).unsqueeze(0)
        act2 = torch.FloatTensor(traj2['act']).unsqueeze(0)
        
        # Compute rewards
        r1 = self(obs1, act1).sum()
        r2 = self(obs2, act2).sum()
        
        # Compute preference loss
        if preference == 1:
            loss = F.softplus(r2 - r1)
        else:  # preference == 2
            loss = F.softplus(r1 - r2)
            
        l2_reg = sum(torch.norm(p, 2) for p in self.parameters() if p.requires_grad)
        loss += self.l2_reg * l2_reg
        
        return loss

