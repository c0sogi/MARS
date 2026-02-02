import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.utils import set_seed, get_device
from library.data_loader import get_loaders
from library.train import run_fold
from library.predict import generate_submission
from library.model import IcebergCNN


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to leverage preprocessed numpy arrays in ./working
    batch_size = 32
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=batch_size, num_workers=2, load_cached_data=True
    )

    # 3. Training
    # The metadata defines a specific Train/Val split (train.csv, val.csv).
    # We treat this as "Fold 0".
    # Limiting epochs to 30 for a fast baseline execution.
    print("\n--- Starting Training ---")
    model = run_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
        epochs=30,
        patience=5,
        lr=1e-3,
        save_dir="./checkpoints",
    )

    # 4. Validation & Metric Calculation
    print("\n--- Starting Validation ---")
    model.eval()

    all_preds = []
    all_targets = []
    all_angles = []

    # Inference loop
    with torch.no_grad():
        for (imgs, angles), labels in val_loader:
            imgs = imgs.to(device)
            angles = angles.to(device)

            # Forward pass
            outputs = model(imgs, angles)
            probs = torch.sigmoid(outputs)

            # Store results
            all_preds.extend(probs.cpu().numpy().flatten().tolist())
            all_targets.extend(labels.numpy().flatten().tolist())
            all_angles.extend(angles.cpu().numpy().flatten().tolist())

    # Convert to numpy arrays
    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    angles_arr = np.array(all_angles)

    # Calculate Log Loss (Metric)
    # Epsilon clipping is handled internally by sklearn usually, but good to be safe
    final_metric = log_loss(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error per sample
    errors = np.abs(y_true - y_pred)

    # Correlation with Incidence Angle
    # Check for NaN in angles (though loader should have handled imputation)
    valid_mask = ~np.isnan(angles_arr)
    if np.sum(valid_mask) > 1:
        corr, _ = pearsonr(angles_arr[valid_mask], errors[valid_mask])
        print(f"Correlation between Error and Incidence Angle: {corr:.6f}")
    else:
        print("Insufficient valid incidence angle data for correlation analysis.")

    # 6. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )
        # We trained 1 fold (Fold 0), so we tell the generator to use 1 fold.
        generate_submission(
            test_loader=test_loader,
            checkpoint_dir="./checkpoints",
            output_path="./submission/submission.csv",
            num_folds=1,
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
