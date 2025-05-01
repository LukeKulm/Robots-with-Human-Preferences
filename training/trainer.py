import torch
from reward_models.reward_predictor import RewardPredictor, preference_loss

DATA_PATH = "data/reward_training_data.pt"
OBS_DIM = 10  # placeholder: adjust to match your data
ACT_DIM = 4   # placeholder: adjust to match your data
EPOCHS = 10
LR = 1e-3

def train():
    dataset = torch.load(DATA_PATH)
    model = RewardPredictor(obs_dim=OBS_DIM, act_dim=ACT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for traj1, traj2, label in dataset:
            traj1 = {k: v.unsqueeze(0) for k, v in traj1.items()}
            traj2 = {k: v.unsqueeze(0) for k, v in traj2.items()}
            label = torch.tensor([label])
            
            loss = preference_loss(model, traj1, traj2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "trained_reward_model.pt")

if __name__ == "__main__":
    train()
