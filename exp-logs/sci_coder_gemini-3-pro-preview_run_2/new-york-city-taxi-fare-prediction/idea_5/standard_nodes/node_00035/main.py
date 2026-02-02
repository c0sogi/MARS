import sys
import os
import numpy as np
import pandas as pd

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_processor import DataProcessor
from library.trainer import Trainer


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading and Processing Data...")
    processor = DataProcessor()
    # Load cached data if available, otherwise process
    train_df, val_df, test_df = processor.process_data(load_cached_data=True)

    # Subsample Training Data for Speed
    TRAIN_SUBSET_SIZE = 5_000_000
    if len(train_df) > TRAIN_SUBSET_SIZE:
        train_df = train_df.sample(n=TRAIN_SUBSET_SIZE, random_state=Config.SEED)
        print(f"Subsampled training set to {TRAIN_SUBSET_SIZE} samples.")

    # Cite solution_lesson_node_00017: Sanitize target variable to remove extreme outliers
    # This prevents L2 loss from exploding during training.
    # We do NOT sanitize validation data, as we must evaluate on the full distribution.
    print("Sanitizing training target variable...")
    train_df = train_df[
        (train_df[Config.TARGET_COL] >= Config.MIN_FARE_TRAIN)
        & (train_df[Config.TARGET_COL] <= Config.MAX_FARE_TRAIN)
    ]

    # Prepare features and targets
    feature_cols = processor.continuous_cols + processor.categorical_cols
    target_col = Config.TARGET_COL

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # ---------------------------------------------------------
    # 3. Training
    # ---------------------------------------------------------
    print("Initializing Trainer (XGBoost)...")
    trainer = Trainer()

    trainer.fit(X_train, y_train, X_val, y_val)

    # ---------------------------------------------------------
    # 4. Final Evaluation
    # ---------------------------------------------------------
    final_rmse = trainer.validate(X_val, y_val)
    print(f"Final Validation Metric: {final_rmse}")

    # ---------------------------------------------------------
    # 5. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")

    # Predict on validation set
    preds = trainer.model.predict(X_val)
    preds = np.maximum(preds, Config.MIN_FARE_PREDICTION)

    # Calculate Absolute Error
    errors = np.abs(preds - y_val.values)

    print("Correlation between Error Magnitude and Input Features:")

    # Compute correlation for each feature
    # Using pandas corrwith would be cleaner but let's stick to numpy for speed/consistency
    for col in feature_cols:
        feat_vals = X_val[col].values
        if np.std(feat_vals) > 0 and np.std(errors) > 0:
            corr = np.corrcoef(errors, feat_vals)[0, 1]
            print(f"  {col}: {corr:.4f}")
        else:
            print(f"  {col}: NaN (Constant values)")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    THRESHOLD = 4.278504866347902

    if final_rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({final_rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        X_test = test_df[feature_cols]
        keys = test_df["key"]
        trainer.generate_submission(X_test, keys)
    else:
        print(
            f"\nValidation RMSE ({final_rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
