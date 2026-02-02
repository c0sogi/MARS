import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import necessary components from the provided library files
from library.config import SEED, DEVICE, NUM_FOLDS, get_model_path
from library.utils import seed_everything, load_checkpoint
from library.model import GA_WBN, generate_submission
from library.train import run_fold_training
from library.data_loader import get_dataloaders, process_and_cache_data


def main():
    # 1. Initialization and Reproducibility
    seed_everything(SEED)
    print(f"Starting execution on device: {DEVICE}")

    # 2. Data Processing
    # Ensure data is processed and cached before starting the loop.
    # This creates the .npz cache file used by subsequent steps.
    print("Initializing data pipeline...")
    process_and_cache_data(load_cached_data=True)

    # 3. Cross-Validation Training & OOF Inference
    oof_preds = []
    oof_targets = []

    # Lists to store features for failure analysis
    fa_errors = []
    fa_angles = []
    fa_means = []
    fa_stds = []

    print(f"Starting {NUM_FOLDS}-Fold Cross-Validation...")

    for fold_idx in range(NUM_FOLDS):
        # --- Train ---
        # run_fold_training handles training, validation monitoring, and saving the best model.
        best_fold_score = run_fold_training(fold_idx)
        print(f"Fold {fold_idx} completed. Best tracked loss: {best_fold_score}")

        # --- Inference on Validation Set ---
        # We need to generate predictions on the validation set using the best model
        # to calculate the global metric and perform failure analysis.

        # Load the best model for this fold
        model_path = get_model_path(fold_idx)
        model = GA_WBN().to(DEVICE)

        if not os.path.exists(model_path):
            print(
                f"Warning: Model checkpoint for fold {fold_idx} not found. Skipping inference."
            )
            continue

        load_checkpoint(model, model_path, device=DEVICE)
        model.eval()

        # Get the validation data loader for this fold
        _, val_loader = get_dataloaders(fold_idx, load_cached_data=True)

        fold_preds = []
        fold_targets = []
        fold_angles = []
        fold_img_means = []
        fold_img_stds = []

        with torch.no_grad():
            for imgs, angs, lbls in val_loader:
                imgs = imgs.to(DEVICE)
                angs = angs.to(DEVICE)

                # Forward pass
                outputs = model(imgs, angs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                # Store predictions and targets
                fold_preds.extend(probs)
                fold_targets.extend(lbls.numpy().flatten())

                # Store metadata for analysis
                fold_angles.extend(angs.cpu().numpy().flatten())

                # Calculate image statistics (Mean/Std of Band 1 & 2) for analysis
                # imgs shape: (B, 3, 75, 75). Channels: 0=Band1, 1=Band2, 2=Mean
                imgs_cpu = imgs.cpu().numpy()
                # Compute stats over spatial dims (2, 3) and channels 0, 1
                b_means = np.mean(imgs_cpu[:, 0:2, :, :], axis=(1, 2, 3))
                b_stds = np.std(imgs_cpu[:, 0:2, :, :], axis=(1, 2, 3))

                fold_img_means.extend(b_means)
                fold_img_stds.extend(b_stds)

        # Append to global OOF lists
        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

        # Calculate errors for this fold for failure analysis
        fold_errors = np.abs(np.array(fold_targets) - np.array(fold_preds))

        fa_errors.extend(fold_errors)
        fa_angles.extend(fold_angles)
        fa_means.extend(fold_img_means)
        fa_stds.extend(fold_img_stds)

    # 4. Global Validation Metric Calculation
    y_true = np.array(oof_targets)
    y_pred = np.array(oof_preds)

    # Clip predictions to prevent log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

    final_metric = log_loss(y_true, y_pred_clipped)
    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    df_analysis = pd.DataFrame(
        {
            "error": fa_errors,
            "inc_angle": fa_angles,
            "img_mean": fa_means,
            "img_std": fa_stds,
        }
    )

    # Calculate correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(f"\nSuccess: Metric ({final_metric}) is below threshold ({THRESHOLD}).")
        print("Generating submission file...")
        generate_submission()
    else:
        print(f"\nFailure: Metric ({final_metric}) is above threshold ({THRESHOLD}).")
        print("Submission file will not be generated.")


if __name__ == "__main__":
    main()
