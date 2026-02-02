import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.utils import set_seed, get_device
from library.dataset import get_dataset, IcebergDataset, get_transforms
from library.model import predict_with_tta
from library.trainer import run_fold_training, generate_submission


def main():
    # 1. Setup Environment
    set_seed(42)
    device = get_device()
    working_dir = "./working/idea_simple_cnn"
    os.makedirs(working_dir, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Load Data
    # Load cached data to optimize runtime
    print("Loading datasets...")
    ds_train_part = get_dataset("train", load_cached_data=True)
    ds_val_part = get_dataset("val", load_cached_data=True)

    # Merge datasets for Stratified K-Fold CV
    # This allows us to train on all available labeled data and produce OOF predictions
    X_all = np.concatenate([ds_train_part.X, ds_val_part.X], axis=0)
    angles_all = np.concatenate([ds_train_part.angles, ds_val_part.angles], axis=0)
    y_all = np.concatenate([ds_train_part.labels, ds_val_part.labels], axis=0)

    print(f"Total labeled samples: {len(y_all)}")

    # 3. Cross Validation Loop
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Array to store Out-Of-Fold predictions
    oof_preds = np.zeros(len(y_all))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Create datasets for the current fold
        train_ds = IcebergDataset(
            X_all[train_idx],
            angles_all[train_idx],
            labels=y_all[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_all[val_idx],
            angles_all[val_idx],
            labels=y_all[val_idx],
            transform=get_transforms("val"),
        )

        # Train the model for this fold
        # Limiting epochs to 35 ensures a fast baseline execution while allowing convergence
        model, best_loss = run_fold_training(
            fold_idx=fold_idx,
            train_dataset=train_ds,
            val_dataset=val_ds,
            batch_size=32,
            lr=1e-3,
            epochs=35,
            patience=8,
            save_dir=working_dir,
        )

        # Generate OOF predictions using the best model
        # Reload best weights
        model_path = os.path.join(working_dir, f"fold_{fold_idx}", "model_best.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Create validation loader (no shuffle)
        val_loader = DataLoader(
            val_ds,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        # Predict using Test-Time Augmentation (TTA)
        # Note: predict_with_tta returns (ids, preds). For validation data, 'ids' corresponds to labels.
        # We ignore the first return value.
        _, preds = predict_with_tta(model, val_loader, device=device)

        # Store predictions
        oof_preds[val_idx] = preds

    # 4. Final Validation Metric
    # Compute Log Loss on the full set of OOF predictions
    final_metric = log_loss(y_all, oof_preds)
    print(f"\nFinal Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_all - oof_preds)

    # Extract features for correlation analysis
    # Flatten spatial dimensions: (N, 3, 75, 75) -> (N, 3, 5625)
    # Channel 0 is Band 1 (HH), Channel 1 is Band 2 (HV)
    b1_flat = X_all[:, 0, :, :].reshape(len(X_all), -1)
    b2_flat = X_all[:, 1, :, :].reshape(len(X_all), -1)

    features = {
        "inc_angle": angles_all,
        "b1_mean": np.mean(b1_flat, axis=1),
        "b1_std": np.std(b1_flat, axis=1),
        "b2_mean": np.mean(b2_flat, axis=1),
        "b2_std": np.std(b2_flat, axis=1),
    }

    print("Correlation between Error and Features:")
    for name, vals in features.items():
        # Calculate Pearson correlation
        corr = np.corrcoef(errors, vals)[0, 1]
        print(f"{name}: {corr:.10f}")

    # 6. Submission Generation
    threshold = 0.18145903282502943

    if final_metric < threshold:
        print(
            f"\nMetric condition met ({final_metric} < {threshold}). Generating submission..."
        )
        generate_submission(
            folds=n_folds,
            batch_size=32,
            model_dir=working_dir,
            output_path="./submission/submission.csv",
        )
    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
