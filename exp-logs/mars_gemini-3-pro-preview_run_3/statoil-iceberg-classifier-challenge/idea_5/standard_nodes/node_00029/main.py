import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.trainer import run_cross_validation, predict_ensemble


def main():
    # 1. Setup Environment
    seed_everything(42)
    device = get_device()
    model_dir = "./working/models"

    # 2. Training (5-Fold CV on Train Set)
    print("Starting 5-Fold Cross-Validation Training...")
    # This trains models and saves them to model_dir
    run_cross_validation(
        n_splits=5,
        epochs=35,
        batch_size=32,
        lr=1e-3,
        patience=10,
        num_workers=4,
        load_cached_data=True,
        seed=42,
        model_dir=model_dir,
    )

    # 3. Data Loading for Evaluation
    print("Loading hold-out validation data...")
    # We use get_dataloaders to easily get the val and test loaders
    # Note: This will re-load metadata but leverage the cache we just updated/used
    _, val_loader, test_loader = get_dataloaders(
        batch_size=32, num_workers=4, load_cached_data=True
    )

    # 4. Evaluation on Hold-out Validation Set
    print("Evaluating Ensemble on hold-out validation set...")
    val_preds = predict_ensemble(val_loader, 5, model_dir, device)

    # Extract targets and auxiliary data for analysis
    val_targets = val_loader.dataset.y.numpy()
    val_angles = val_loader.dataset.angles.numpy()
    val_images = val_loader.dataset.X.numpy()  # Shape (N, 3, 75, 75)

    # Calculate Log Loss
    final_metric = log_loss(val_targets, val_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(val_targets - val_preds)

    feat_b1_mean = np.mean(val_images[:, 0, :, :], axis=(1, 2))
    feat_b2_mean = np.mean(val_images[:, 1, :, :], axis=(1, 2))

    valid_mask = ~np.isnan(val_angles)
    if np.sum(valid_mask) > 1:
        corr_angle = pearsonr(errors[valid_mask], val_angles[valid_mask])[0]
    else:
        corr_angle = 0.0

    corr_b1 = pearsonr(errors, feat_b1_mean)[0]
    corr_b2 = pearsonr(errors, feat_b2_mean)[0]

    print("Correlation between Error Magnitude and Features:")
    print(f"  Incidence Angle: {corr_angle:.4f}")
    print(f"  Band 1 Mean: {corr_b1:.4f}")
    print(f"  Band 2 Mean: {corr_b2:.4f}")

    # 6. Submission Generation
    threshold = 0.18145903282502943

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} meets threshold {threshold}. Generating submission..."
        )

        # Predict on Test Set using Ensemble
        test_preds = predict_ensemble(test_loader, 5, model_dir, device)
        test_ids = test_loader.dataset.ids

        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
