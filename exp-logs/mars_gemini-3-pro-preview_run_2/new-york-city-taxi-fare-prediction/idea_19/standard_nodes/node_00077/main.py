import os
import sys
import gc
import numpy as np
import pandas as pd
import xgboost as xgb

# Import provided library modules
from library.config import ProjectConfig
from library.data_pipeline import DataPipeline
from library.model_trainer import ModelTrainer


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(ProjectConfig.SEED)

    # 2. Train Model
    # The ModelTrainer handles the heavy lifting: data loading, training, and test prediction.
    print("Starting Model Training Pipeline...")
    trainer = ModelTrainer()
    trainer.train_model(load_cached_data=True)

    # 3. Validation Assessment
    print("\n=== Validation Assessment ===")

    # Reload validation data to perform comprehensive evaluation
    # We unpack the tuple (train, val, test) and discard train/test to save memory
    print("Loading validation data for analysis...")
    pipeline = DataPipeline()
    _, val_df, _ = pipeline.get_data(load_cached=True)

    # Clean up memory
    gc.collect()

    # Prepare Validation Features
    # Ensure we use the exact same features as training
    if trainer.features is None:
        # Fallback if features weren't stored (though train_model stores them)
        exclude_cols = {"key", "fare_amount", "pickup_datetime", "fold"}
        features = [c for c in val_df.columns if c not in exclude_cols]
    else:
        features = trainer.features

    target_col = "fare_amount"

    # Create DMatrix for validation inference
    dval = xgb.DMatrix(val_df[features])

    # Generate Predictions
    # The model is stored in trainer.model after training
    print("Generating validation predictions...")
    val_preds = trainer.model.predict(dval)

    # Apply post-processing (Minimum Fare Floor)
    val_preds = np.maximum(val_preds, ProjectConfig.PRED_MIN_FARE)

    # Calculate RMSE
    y_true = val_df[target_col].values
    mse = np.mean((y_true - val_preds) ** 2)
    rmse = np.sqrt(mse)

    # Print Metric in strict format
    print(f"Final Validation Metric: {rmse}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude
    val_df["error"] = val_preds - y_true
    val_df["abs_error"] = np.abs(val_df["error"])

    # Calculate correlation between features and error magnitude
    print("Correlation between Input Features and Error Magnitude:")
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    # Exclude the error columns themselves from correlation
    cols_to_corr = [c for c in numeric_cols if c not in ["error", "abs_error"]]

    correlations = (
        val_df[cols_to_corr]
        .corrwith(val_df["abs_error"])
        .abs()
        .sort_values(ascending=False)
    )
    print(correlations.head(10))

    # 5. Submission Logic
    # Threshold defined in task
    THRESHOLD = 3.438959912830025
    submission_path = os.path.join(ProjectConfig.SUBMISSION_DIR, "submission.csv")

    if rmse > THRESHOLD:
        print(f"\nMetric ({rmse}) did not meet threshold ({THRESHOLD}).")
        if os.path.exists(submission_path):
            print("Removing submission file...")
            os.remove(submission_path)
        else:
            print("No submission file found to remove.")
    else:
        print(f"\nMetric ({rmse}) meets threshold ({THRESHOLD}).")
        if os.path.exists(submission_path):
            print(f"Submission file confirmed at: {submission_path}")
        else:
            # If for some reason trainer.train_model didn't save it (unlikely), generate it now
            print("Submission file missing. Regenerating...")
            _, _, test_df = pipeline.get_data(load_cached=True)
            trainer.predict(test_df)


if __name__ == "__main__":
    main()
