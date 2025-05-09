import os
import subprocess
from training.trainer import RLHFTrainer
from training.human_feedback_collector import convert_preferences_to_dataset
from inference.render_inference import render_trajectory_video

trainer = RLHFTrainer(save_dir="saved_models")

total_iterations = 20
ask_every = 2

for iter in range(total_iterations):
    print(f"\nIteration {iter + 1}/{total_iterations}")

    # Collect and save 2 new trajectories
    trajectories = trainer.collect_trajectories(n_trajectories=2)
    for i, traj in enumerate(trajectories):
        prefix = f"{'left' if i == 0 else 'right'}_clip_{iter:03d}"
        render_trajectory_video(traj, trainer.env, filename_prefix=prefix)

    if iter % ask_every == 0:
        print("Launching labeling UI...")
        # subprocess.run(["streamlit", "run", "scripts/visualize_trajectories.py"])

        convert_preferences_to_dataset()
        trainer.train_reward_model()

    print("PPO Update")
    trainer.train_step(n_trajectories=4, n_ppo_epochs=3)

    if (iter + 1) % 5 == 0:
        trainer.save_models()
