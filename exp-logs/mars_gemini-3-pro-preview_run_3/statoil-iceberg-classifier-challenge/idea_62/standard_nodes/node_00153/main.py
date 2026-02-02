import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import get_fold_loaders
from library.model import LSEIsomorphicCNN
from library.train import run_cross_validation
from library.predict import generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config to use the root working directory and ensure correct paths
    Config.WORKING_DIR = "./working"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set Fast Baseline Hyperparameters
    # Reducing epochs to 25 to ensure execution completes well within the time limit
    # while allowing sufficient convergence for the small dataset.
    Config.NUM_EPOCHS = 25

    # Clear stale cache to prevent IndexError due to mismatched array sizes (Cite debug_lesson_4)
    if os.path.exists(Config.CACHE_DIR):
        print(f"Clearing stale cache at {Config.CACHE_DIR} to force regeneration...")
        shutil.rmtree(Config.CACHE_DIR)

    # Initialize environment (create dirs, set seeds)
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    # Run the 5-fold cross-validation defined in library.train
    run_cross_validation()

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    # We need to reconstruct the Out-Of-Fold (OOF) predictions to calculate
    # the global validation metric and perform failure analysis.

    oof_preds = []
    oof_targets = []

    # Lists to store features for failure analysis
    # We will correlate these with the error magnitude
    fa_angles = []
    fa_means = []
    fa_stds = []

    print("Starting OOF Validation and Failure Analysis...")

    for fold_idx in range(Config.N_FOLDS):
        # Retrieve validation loader for the current fold
        # load_cached_data=True uses the .npy files generated during training
        _, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        # Initialize model and load the best checkpoint for this fold
        model = LSEIsomorphicCNN().to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for fold {fold_idx} not found. Skipping this fold."
            )
            continue

        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)
        model.eval()

        # Inference on validation set
        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward pass
                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                targets_np = targets.numpy().flatten()

                # Store predictions and targets
                oof_preds.extend(probs)
                oof_targets.extend(targets_np)

                # Extract features for failure analysis
                # 1. Incidence Angle
                fa_angles.extend(angles.cpu().numpy().flatten())

                # 2. Image Statistics (Mean and Std)
                # Flatten images to (B, Pixels) to compute stats
                # images shape: (B, 3, 75, 75)
                imgs_flat = images.view(images.size(0), -1).cpu().numpy()
                fa_means.extend(np.mean(imgs_flat, axis=1))
                fa_stds.extend(np.std(imgs_flat, axis=1))

    # Convert to numpy arrays
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Calculate Log Loss
    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    oof_preds_clipped = np.clip(oof_preds, epsilon, 1 - epsilon)
    final_metric = log_loss(oof_targets, oof_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    # Calculate Error Magnitude
    errors = np.abs(oof_targets - oof_preds)

    # Calculate Correlations
    # We use Pearson correlation to check linear relationship between error and features
    corr_angle, _ = pearsonr(errors, fa_angles)
    corr_mean, _ = pearsonr(errors, fa_means)
    corr_std, _ = pearsonr(errors, fa_stds)

    print("-" * 40)
    print("Failure Analysis (Correlation with Error Magnitude)")
    print("-" * 40)
    print(f"Incidence Angle Correlation:  {corr_angle:.4f}")
    print(f"Image Mean Intensity Corr:    {corr_mean:.4f}")
    print(f"Image Contrast (Std) Corr:    {corr_std:.4f}")
    print("-" * 40)

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        print(
            f"Metric {final_metric:.6f} is better than threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Metric {final_metric:.6f} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
