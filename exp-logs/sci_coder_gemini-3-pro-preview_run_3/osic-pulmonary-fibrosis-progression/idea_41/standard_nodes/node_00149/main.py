import os
import shutil
import torch
import pandas as pd
import numpy as np
import sys

# Ensure library imports work by adding current directory to path
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.train import Runner


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Train
    # Using 25 epochs to ensure full convergence of the scheduler.
    # Cite solution_lesson_node_00100: Dynamically link scheduler horizon to training duration.
    print("Initializing Runner...")
    runner = Runner(epochs=25)

    print("Starting Training...")
    runner.run()

    # 3. Validation Metric
    # Print exact format required by the task
    print(f"Final Validation Metric: {runner.best_metric}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load best model weights to ensure analysis reflects the optimal state
    if os.path.exists(runner.best_model_path):
        runner.model.load_state_dict(
            torch.load(runner.best_model_path, map_location=runner.device)
        )
    else:
        print("Warning: Best model checkpoint not found. Using current weights.")

    runner.model.eval()

    # Retrieve stats for denormalization
    fvc_mean = runner.stats["fvc_mean"]
    fvc_std = runner.stats["fvc_std"]

    all_errors = []
    all_clinical = []

    # Collect predictions and features from validation set
    with torch.no_grad():
        for img, clinical, target in runner.val_loader:
            img = img.to(runner.device)
            clinical = clinical.to(runner.device)
            target = target.to(runner.device)

            mu_norm, sigma_norm = runner.model(img, clinical)

            # Denormalize to get error in ml (interpretable units)
            mu_abs = mu_norm * fvc_std + fvc_mean
            target_abs = target * fvc_std + fvc_mean

            error = torch.abs(target_abs - mu_abs)

            all_errors.append(error.cpu().numpy())
            all_clinical.append(clinical.cpu().numpy())

    if len(all_errors) > 0:
        all_errors = np.concatenate(all_errors)
        all_clinical = np.concatenate(all_clinical)  # Shape (N, 5)

        # Clinical columns mapping from LungDataset: [BaseFVC, Time, Age, Sex, Smoking]
        df_analysis = pd.DataFrame(
            all_clinical, columns=["BaseFVC", "Time", "Age", "Sex", "Smoking"]
        )
        df_analysis["Error"] = all_errors

        # Calculate correlation
        correlations = df_analysis.corr()["Error"].drop("Error")
        print("Correlation between Absolute Error and Input Features:")
        print(correlations)
    else:
        print("No validation data found for failure analysis.")

    # 5. Submission
    # Threshold defined in the task
    THRESHOLD = -6.573619738753321

    if runner.best_metric > THRESHOLD:
        print(f"\nMetric {runner.best_metric} > {THRESHOLD}. Generating submission...")
        runner.generate_submission()

        # The runner saves to Config.SUBMISSION_DIR (./working/idea_41/submission)
        # We need to ensure the file is at ./submission/submission.csv as requested
        src_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        dest_dir = "./submission"
        dest_path = os.path.join(dest_dir, "submission.csv")

        os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Submission file saved to {dest_path}")
        else:
            print(f"Error: Submission file not found at {src_path}")
    else:
        print(
            f"\nMetric {runner.best_metric} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
