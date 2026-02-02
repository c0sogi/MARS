import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.model import BHAResNet
from library.data_loader import get_dataloaders
from library.train import train_model


def main():
    # 1. Setup and Config Override
    # Set fixed seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # We use 35 epochs to ensure convergence within the time limit while maintaining performance.
    Config.NUM_EPOCHS = 35

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)

    # 2. Train Model
    # We train Fold 0. Given the fixed metadata split, this represents our training set.
    # The provided library handles the training loop and saving checkpoints.
    print("Starting training for Fold 0...")
    train_model(fold=0, epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE)

    # 3. Validation & Metrics
    print("Starting validation...")

    # Get dataloaders (this ensures data is processed and cached)
    # We use load_cached_data=True to speed up if data exists
    _, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Load the best model from the training run
    model = BHAResNet().to(device)
    best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")

    # Fallback to last checkpoint if best doesn't exist (unlikely with patience)
    if not os.path.exists(best_ckpt_path):
        best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_fold_0.pth")

    load_checkpoint(best_ckpt_path, model)
    model.eval()

    # Inference on Validation Set
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for images, angles, targets in val_loader:
            images = images.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            val_probs.extend(probs)
            val_targets.extend(targets.numpy())

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)

    # Calculate Log Loss
    # Clip probabilities to avoid log(0) errors (standard practice, though sigmoid outputs (0,1))
    metric = log_loss(val_targets, val_probs, eps=1e-15)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("Performing failure analysis...")

    # Load cached validation data for feature extraction
    # Path logic matches library.data_loader.process_data
    val_npy_path = os.path.join(Config.CACHE_DIR, "X_val.npy")
    val_angle_path = os.path.join(Config.CACHE_DIR, "angle_val.npy")

    if os.path.exists(val_npy_path) and os.path.exists(val_angle_path):
        X_val = np.load(val_npy_path)
        angles_val = np.load(val_angle_path)

        # Calculate error magnitude
        errors = np.abs(val_targets - val_probs)

        # Extract features
        # X_val shape: (N, 3, 75, 75). Channels: 0=HH, 1=HV, 2=Avg
        b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
        b1_std = np.std(X_val[:, 0, :, :], axis=(1, 2))
        b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))
        b2_std = np.std(X_val[:, 1, :, :], axis=(1, 2))

        df_analysis = pd.DataFrame(
            {
                "error": errors,
                "inc_angle": angles_val,
                "b1_mean": b1_mean,
                "b1_std": b1_std,
                "b2_mean": b2_mean,
                "b2_std": b2_std,
            }
        )

        # Correlation
        corr = (
            df_analysis.corrwith(df_analysis["error"])
            .drop("error")
            .sort_values(ascending=False)
        )
        print("Correlation of Error with Features:")
        print(corr)
    else:
        print("Cached validation data not found. Skipping detailed failure analysis.")

    # 5. Submission
    threshold = 0.1806015565870406
    if metric < threshold:
        print(f"Metric {metric} meets threshold {threshold}. Generating submission...")

        test_ids = []
        test_probs = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                test_ids.extend(ids)
                test_probs.extend(probs)

        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Metric {metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
