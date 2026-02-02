import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import components from the provided library files
from library.config import Config
from library.dataset import SETIDataset
from library.model import BaselineCNN, set_seeds
from library.engine import run_training, generate_submission


def main():
    # Suppress unnecessary warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Ensure reproducibility
    set_seeds()

    print("=== Starting Fast Baseline Pipeline ===")

    # 1. Train the model
    print("Initiating training...")
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=False,  # Use full dataset
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 2. Validation & Failure Analysis
    print("Training complete. Starting validation and failure analysis...")

    # Load the best model saved during training
    device = torch.device(Config.DEVICE)
    model = BaselineCNN().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No model file found. Using random weights.")

    model.eval()

    # Prepare Validation Loader
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = SETIDataset(df_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Storage for metrics and analysis
    all_targets = []
    all_preds = []
    meta_features = []
    errors = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            # Move data to CPU for analysis
            batch_targets = targets.cpu().numpy().flatten()
            batch_preds = probs.cpu().numpy().flatten()

            all_targets.extend(batch_targets)
            all_preds.extend(batch_preds)

            # Calculate absolute error for failure analysis
            batch_errors = np.abs(batch_targets - batch_preds)
            errors.extend(batch_errors)

            # Extract Meta-Features from images for correlation analysis
            # Image shape: (Batch, 6, 273, 256)
            # On-Target panels: 0, 2, 4
            # Off-Target panels: 1, 3, 5

            # Isolate panels
            on_target_imgs = inputs[:, [0, 2, 4], :, :]
            off_target_imgs = inputs[:, [1, 3, 5], :, :]

            # Calculate stats (mean and max) across spatial and frequency dimensions
            # dim=(1, 2, 3) reduces (Channels, Freq, Time) to a single scalar per sample
            mean_on = torch.mean(on_target_imgs, dim=(1, 2, 3)).cpu().numpy()
            mean_off = torch.mean(off_target_imgs, dim=(1, 2, 3)).cpu().numpy()
            max_on = torch.amax(on_target_imgs, dim=(1, 2, 3)).cpu().numpy()

            # Calculate contrast
            mean_diff = mean_on - mean_off

            # Store features
            for i in range(len(batch_targets)):
                meta_features.append(
                    {
                        "mean_on_target": mean_on[i],
                        "mean_off_target": mean_off[i],
                        "max_on_target": max_on[i],
                        "mean_diff": mean_diff[i],
                    }
                )

    # 3. Report Metrics
    final_auc = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 4. Report Failure Analysis
    df_analysis = pd.DataFrame(meta_features)
    df_analysis["error"] = errors

    # Compute correlation between error magnitude and signal features
    correlations = (
        df_analysis.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("Failure Analysis (Correlation with Error Magnitude):")
    print(correlations)

    # 5. Generate Submission
    if final_auc > 0.5170457784564271:
        print("Generating submission file...")
        generate_submission(
            batch_size=Config.BATCH_SIZE,
            debug=False,
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_SAVE_PATH,
        )
    else:
        print(
            f"Validation AUC ({final_auc}) did not improve upon baseline (0.5170). Skipping submission."
        )

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
