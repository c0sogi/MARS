import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.data_loader import get_dataloaders, get_test_dataloader
from library.trainer import IceCubeTrainer
from library.model import DualStreamNetwork, predict_submission
from library.utils import spherical_to_cartesian, cartesian_to_spherical

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def compute_angular_error(az_true, zen_true, az_pred, zen_pred):
    """
    Computes element-wise angular error between two sets of spherical coordinates.
    """
    # Convert to cartesian unit vectors
    tx, ty, tz = spherical_to_cartesian(az_true, zen_true)
    px, py, pz = spherical_to_cartesian(az_pred, zen_pred)

    # Dot product
    dot = np.clip(tx * px + ty * py + tz * pz, -1.0, 1.0)

    # Angle
    return np.arccos(dot)


def main():
    # 1. Setup and Configuration
    print("Initializing configuration...")
    seed_everything(Config.SEED)
    Config.setup()

    # Modify Config for Fast Baseline Execution
    # We limit the data size and epochs to ensure completion within the time limit
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 200000  # Train on a subset of 200k events
    Config.NUM_EPOCHS = 2  # Train for 2 epochs
    Config.BATCH_SIZE = 512  # Increase batch size for A100 efficiency
    Config.NUM_WORKERS = 12  # Use all vCPUs

    print(
        f"Configured for fast baseline: {Config.DEBUG_SUBSET_SIZE} samples, {Config.NUM_EPOCHS} epochs."
    )

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(Config)

    # 3. Training
    print("Starting training pipeline...")
    trainer = IceCubeTrainer(Config, train_loader, val_loader)
    best_model_path = trainer.fit()

    # 4. Validation & Failure Analysis
    print("\n" + "=" * 40)
    print("Performing Final Validation & Failure Analysis")
    print("=" * 40)

    device = torch.device(Config.DEVICE)
    model = DualStreamNetwork().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_errors = []
    val_geom_features = []

    # Lists to store full predictions for metric calculation
    all_true_az = []
    all_true_zen = []
    all_pred_az = []
    all_pred_zen = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for seq_x, geom_x, target_angles, _ in val_loader:
            seq_x = seq_x.to(device)
            geom_x = geom_x.to(device)

            # Inference
            preds_vec = model(seq_x, geom_x)

            # Convert predictions to spherical
            p_x = preds_vec[:, 0]
            p_y = preds_vec[:, 1]
            p_z = preds_vec[:, 2]
            pred_az, pred_zen = cartesian_to_spherical(p_x, p_y, p_z)

            # Move to CPU
            pred_az = pred_az.cpu().numpy()
            pred_zen = pred_zen.cpu().numpy()
            true_az = target_angles[:, 0].numpy()
            true_zen = target_angles[:, 1].numpy()
            geom_feats = geom_x.cpu().numpy()

            # Compute errors for this batch
            batch_errors = compute_angular_error(true_az, true_zen, pred_az, pred_zen)

            # Store
            val_errors.append(batch_errors)
            val_geom_features.append(geom_feats)

            all_true_az.append(true_az)
            all_true_zen.append(true_zen)
            all_pred_az.append(pred_az)
            all_pred_zen.append(pred_zen)

    # Concatenate results
    val_errors = np.concatenate(val_errors)
    val_geom_features = np.concatenate(val_geom_features)
    all_true_az = np.concatenate(all_true_az)
    all_true_zen = np.concatenate(all_true_zen)
    all_pred_az = np.concatenate(all_pred_az)
    all_pred_zen = np.concatenate(all_pred_zen)

    # Calculate Final Metric
    final_metric = np.mean(val_errors)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Geometric Features
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    feature_names = [
        "cog_x",
        "cog_y",
        "cog_z",
        "cov_xx",
        "cov_yy",
        "cov_zz",
        "cov_xy",
        "cov_xz",
        "cov_yz",
    ]

    for i, name in enumerate(feature_names):
        feat_values = val_geom_features[:, i]
        # Calculate Pearson correlation
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(feat_values, val_errors)[0, 1]
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (Constant feature)")

    # 5. Submission
    # Threshold defined in task description logic (must be better than ~1.51)
    # We use the provided threshold from the prompt
    SUBMISSION_THRESHOLD = 1.510849213393748

    if final_metric < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission for test set...")

        test_loader = get_test_dataloader(Config)
        predict_submission(Config, test_loader, best_model_path)
    else:
        print(
            f"\nValidation metric ({final_metric}) does NOT meet threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
