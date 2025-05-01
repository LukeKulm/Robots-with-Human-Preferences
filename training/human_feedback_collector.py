import os
import json
import torch
from glob import glob
import numpy as np

CLIP_DATA_DIR = "data/trajectories"
PREFS_FILE = "data/preferences.json"
OUTPUT_FILE = "data/reward_training_data.pt"

def load_preferences():
    with open(PREFS_FILE, "r") as f:
        return json.load(f)

def load_trajectory(path):
    # You can customize this based on how you store observations/actions per trajectory
    # For now assume each path like `left_clip_001.mp4` maps to `left_clip_001.npz`
    traj_path = path.replace(".mp4", ".npz").replace("clips", "trajectories")
    data = np.load(traj_path)
    return {
        "obs": torch.tensor(data["obs"], dtype=torch.float32),
        "act": torch.tensor(data["act"], dtype=torch.float32)
    }

def main():
    prefs = load_preferences()
    dataset = []

    for entry in prefs:
        traj1 = load_trajectory(entry["left"])
        traj2 = load_trajectory(entry["right"])
        label = {"Left": 0, "Right": 1, "Tie": 0.5, "Can't Tell": 0.5}[entry["preference"]]
        dataset.append((traj1, traj2, label))

    torch.save(dataset, OUTPUT_FILE)
    print(f"Saved {len(dataset)} trajectory preferences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
