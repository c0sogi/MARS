import sys
import os
import numpy as np
import pandas as pd

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import TaxiDataset
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("Initializing Trainer and Model...")
    trainer = Trainer()

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading Datasets...")

    # Load Full Training Data
    train_dataset = TaxiDataset(split="train", load_cached_data=True)
    X_train, y_train = train_dataset.get_data()

    # Cite solution_lesson_node_00017: Sanitize target variable to prevent instability.
    print("Sanitizing target variable...")
    mask = (y_train >= Config.TARGET_MIN) & (y_train <= Config.TARGET_MAX)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Subsample Training Data for Speed (Cite solution_lesson_node_00017)
    if len(y_train) > Config.TRAIN_SUBSET_SIZE:
        print(f"Subsampling training set to {Config.TRAIN_SUBSET_SIZE} samples.")
        indices = np.random.choice(
            len(y_train), Config.TRAIN_SUBSET_SIZE, replace=False
        )
        X_train = X_train[indices]
        y_train = y_train[indices]
    else:
        print(f"Using full filtered training set ({len(y_train)} samples).")

    # Load Full Validation Data
    val_dataset = TaxiDataset(split="val", load_cached_data=True)
    X_val, y_val = val_dataset.get_data()

    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    print("Starting Training...")
    trainer.fit(X_train, y_train, X_val, y_val)

    # ---------------------------------------------------------
    # 4. Final Evaluation
    # ---------------------------------------------------------
    # Compute metric on the entire hold-out validation set
    final_rmse = trainer.validate(X_val, y_val)
    print(f"Final Validation Metric: {final_rmse}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Analyze a subset of validation data
    ANALYSIS_SIZE = 100_000
    if len(X_val) > ANALYSIS_SIZE:
        indices = np.random.choice(len(X_val), ANALYSIS_SIZE, replace=False)
        X_val_sample = X_val[indices]
        y_val_sample = y_val[indices]
    else:
        X_val_sample = X_val
        y_val_sample = y_val

    preds = trainer.model.predict(X_val_sample)
    preds = np.maximum(preds, Config.MIN_FARE_PREDICTION)

    errors = np.abs(preds - y_val_sample)

    print("Correlation between Error Magnitude and Input Features:")
    feature_names = train_dataset.feature_cols

    for idx, col_name in enumerate(feature_names):
        if idx < X_val_sample.shape[1]:
            feature_vals = X_val_sample[:, idx]
            if np.std(feature_vals) > 0 and np.std(errors) > 0:
                corr = np.corrcoef(errors, feature_vals)[0, 1]
                print(f"  {col_name}: {corr:.4f}")
            else:
                print(f"  {col_name}: NaN (Constant values)")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 4.278504866347902

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({final_rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation RMSE ({final_rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
