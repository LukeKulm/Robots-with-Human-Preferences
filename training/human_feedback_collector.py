import json
import torch
import numpy as np
import os


CLIP_DATA_DIR = "data/trajectories"
PREFS_FILE = "data/preferences.json"
OUTPUT_FILE = "data/reward_training_data.pt"


def load_preferences():
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(PREFS_FILE), exist_ok=True)
    
    # Create file with empty list if it doesn't exist
    if not os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "w") as f:
            json.dump([], f)
        return []
    
    # Read existing preferences
    with open(PREFS_FILE, "r") as f:
        try:
            prefs = json.load(f)
            if not isinstance(prefs, list):
                print("Warning: preferences file is not a list, initializing empty list")
                prefs = []
            return prefs
        except json.JSONDecodeError:
            print("Warning: preferences file is invalid JSON, initializing empty list")
            return []


def load_trajectory(path):
    # You can customize this based on how you store observations/actions per trajectory
    # For now assume each path like `left_clip_001.mp4` maps to `left_clip_001.npz`
    traj_path = path.replace(".mp4", ".npz").replace("clips", "trajectories")
    data = np.load(traj_path)
    return {
        "obs": torch.tensor(data["obs"], dtype=torch.float32),
        "act": torch.tensor(data["act"], dtype=torch.float32)
    }
    

def convert_preferences_to_dataset():
    prefs = load_preferences()
    dataset = []

    for entry in prefs:
        traj1 = load_trajectory(entry["left"])
        traj2 = load_trajectory(entry["right"])
        label = {"Left": 0, "Right": 1, "Tie": 0.5, "Can't Tell": 0.5}[entry["preference"]]
        dataset.append((traj1, traj2, label))

    torch.save(dataset, OUTPUT_FILE)
    print(f"Saved {len(dataset)} pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    convert_preferences_to_dataset()
