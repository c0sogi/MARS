import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import MicroResNet, train_model
from library.inference import predict_with_tta


from library.trainer import (
    load_subset_data,
    run_cross_validation,
    predict_ensemble,
    load_test_data,
)


def main():
    # 1. Setup Environment
    seed_everything(42)

    # 2. Data Loading
    print("Loading data...")
    # Load Train set for CV training
    X_train, ang_train, y_train, ids_train = load_subset_data(
        "train", load_cached_data=True
    )
    # Load Val set for Hold-out evaluation
    X_val, ang_val, y_val, ids_val = load_subset_data("val", load_cached_data=True)

    # 3. Training (Ensemble CV)
    print("Starting 5-Fold Cross-Validation on Train set...")
    run_cross_validation(
        X_train,
        ang_train,
        y_train,
        ids_train,
        n_splits=5,
        epochs=35,
        batch_size=32,
        lr=1e-3,
        patience=10,
    )

    # 4. Validation Assessment (Ensemble Prediction)
    print("Evaluating Ensemble on hold-out validation set...")
    val_preds = predict_ensemble(X_val, ang_val, ids_val, n_splits=5)

    # Calculate Log Loss
    final_metric = log_loss(y_val, val_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val - val_preds)

    # Features for correlation
    feat_b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    feat_b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))

    valid_mask = ~np.isnan(ang_val)
    if np.sum(valid_mask) > 1:
        corr_angle = pearsonr(errors[valid_mask], ang_val[valid_mask])[0]
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

        # Load Test Data
        X_test, ang_test, ids_test = load_test_data(load_cached_data=True)

        # Predict
        test_preds = predict_ensemble(X_test, ang_test, ids_test, n_splits=5)

        # Create Submission
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds})
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
