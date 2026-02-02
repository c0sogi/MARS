import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib

# ---------------------------------------------------------
# Fast Baseline Configuration Override
# ---------------------------------------------------------
# Reduce epochs to ensure the script completes quickly (Fast Baseline requirement)
# We modify the variable in the train module directly because it was imported via 'from ... import ...'
train_lib.NUM_EPOCHS = 30
print(
    f"Configuration: Reduced NUM_EPOCHS to {train_lib.NUM_EPOCHS} for fast baseline execution."
)


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Data
    # We load the processed data arrays directly to handle OOF indexing and analysis
    data = data_loader.process_data(load_cached_data=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    angles_train = data["angles_train"]

    # Prepare OOF prediction array
    oof_probs = np.zeros_like(y_train, dtype=np.float32)

    # Prepare Test Storage
    test_loader = data_loader.get_test_loader(load_cached_data=True)
    test_ids = data["ids_test"]
    # Accumulator for test predictions (will be averaged later)
    test_probs_sum = np.zeros(len(test_ids), dtype=np.float32)

    # 3. Cross-Validation Loop
    # We replicate the split logic to ensure we know which indices belong to validation
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )
    splits = list(skf.split(X_train, y_train))

    for fold in range(config.NUM_FOLDS):
        print(f"\n=== Processing Fold {fold} ===")

        # Get Loaders for this fold
        train_loader, val_loader = data_loader.get_loaders(fold, load_cached_data=True)

        # Train the model
        # This function saves the best model to checkpoints/model_fold_{fold}.pth
        train_lib.train_fold(fold, train_loader, val_loader)

        # Load Best Model for Inference
        model = model_lib.MAPCNN().to(device)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # --- OOF Inference ---
        # Identify validation indices for this fold
        _, val_idx = splits[fold]

        fold_val_probs = []
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_val_probs.append(probs)

        fold_val_probs = np.concatenate(fold_val_probs)
        # Assign predictions to the global OOF array
        # Note: val_loader yields samples in the order of X_train[val_idx]
        oof_probs[val_idx] = fold_val_probs

        # --- Test Inference ---
        fold_test_probs = []
        with torch.no_grad():
            for images, angles, img_ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)
                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_test_probs.append(probs)

        fold_test_probs = np.concatenate(fold_test_probs)
        test_probs_sum += fold_test_probs

    # 4. Validation Metric
    # Clip probabilities to avoid log(0)
    oof_probs_clipped = np.clip(oof_probs, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_train, oof_probs_clipped)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_train - oof_probs)

    # Compute simple image statistics for correlation analysis
    # X_train shape: (N, 3, 75, 75). Channel 0=Band1, Channel 1=Band2
    b1_mean = np.mean(X_train[:, 0, :, :], axis=(1, 2))
    b1_max = np.max(X_train[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_train[:, 0, :, :], axis=(1, 2))

    b2_mean = np.mean(X_train[:, 1, :, :], axis=(1, 2))
    b2_max = np.max(X_train[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_train[:, 1, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_train,
            "b1_mean": b1_mean,
            "b1_max": b1_max,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_max": b2_max,
            "b2_std": b2_std,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = 0.18120490171618245

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric {final_metric} < {THRESHOLD}. Generating submission..."
        )

        # Average predictions across folds
        avg_test_probs = test_probs_sum / config.NUM_FOLDS

        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_probs})

        submission_path = config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
    else:
        print(
            f"\nValidation metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
