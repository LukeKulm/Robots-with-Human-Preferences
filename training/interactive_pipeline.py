import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import subprocess
from training.trainer import RLHFTrainer
from training.human_feedback_collector import convert_preferences_to_dataset
from inference.render_inference import render_trajectory_video

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

def calculate_baseline_reward(trajectory):
    """Calculate baseline reward based on distance to target"""
    target_pos = np.array([0.5, 0.5, 0.5])  # Example target position
    distances = []
    for obs in trajectory['observations']:
        end_effector_pos = obs[:3]  # Assuming first 3 values are end effector position
        distance = np.linalg.norm(end_effector_pos - target_pos)
        distances.append(distance)
    return -np.mean(distances)  # Negative because lower distance is better

trainer = RLHFTrainer(save_dir="saved_models")
performance_history = []

total_iterations = 200
ask_every = 2

# Create directory for plots
plots_dir = "training_plots"
os.makedirs(plots_dir, exist_ok=True)

for iter in range(total_iterations):
    print(f"\n{'='*50}")
    print(f"Iteration {iter + 1}/{total_iterations}")
    print(f"{'='*50}")

    # Only collect and render trajectories for preference collection when needed
    if iter % ask_every == 0:
        print("\n[Preference Collection Phase]")
        # Collect and save 2 new trajectories for preference collection
        pref_trajectories = trainer.collect_trajectories(n_trajectories=2)
        
        # Only render the trajectories that will be shown to the user
        for i, traj in enumerate(pref_trajectories):
            prefix = f"{'left' if i == 0 else 'right'}_clip_{iter:03d}"
            print(f"Rendering trajectory {i+1} for preference collection...")
            render_trajectory_video(traj, trainer.env, filename_prefix=prefix)

        print("\nLaunching labeling UI...")
        # Launch Streamlit in a separate process
        subprocess.Popen(["streamlit", "run", "scripts/visualize_trajectories.py"])
        
        # Wait for user input before proceeding
        input("Please label the trajectories in the Streamlit UI. Press Enter when done...")

        print("\n[Reward Model Training Phase]")
        convert_preferences_to_dataset()
        trainer.train_reward_model()

    print("\n[PPO Training Phase]")
    # Collect separate trajectories for PPO training
    policy_loss, value_loss, entropy = trainer.train_step(n_trajectories=20, n_ppo_epochs=3)
    print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")

    # Calculate and store performance metrics
    eval_trajectory = trainer.collect_trajectories(n_trajectories=1)[0]
    baseline_reward = calculate_baseline_reward(eval_trajectory)
    performance_history.append(baseline_reward)
    print(f"Current Performance (Distance to Target): {-baseline_reward:.4f}")

    if (iter + 1) % 5 == 0:
        trainer.save_models()
        print("\n[Model Saving] Models saved")
        
        # Update performance plot
        plot_performance(performance_history, 
                        os.path.join(plots_dir, f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        print("Performance plot updated")

