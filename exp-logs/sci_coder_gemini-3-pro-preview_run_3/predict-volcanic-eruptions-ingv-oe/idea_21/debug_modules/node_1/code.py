import os
import sys
import random
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
import library.config as config
import library.data_processor as dp
import library.model_trainer as mt


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    print("Initializing demonstration...")
    set_seed(config.SEED)

    # 2. Configuration Overrides for Speed and Demo Purposes
    print("Configuring parameters for rapid execution...")

    # Enable debug mode to process only a small subset of data
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 100  # Process 100 rows for train/val/test

    # Reduce CV folds to 2 for speed
    config.N_FOLDS = 2

    # Adjust LightGBM parameters for the small dataset
    # Reduce estimators to minimize training time
    config.LGBM_PARAMS["n_estimators"] = 10
    # Reduce min_child_samples to allow splits on small data (100 samples / 2 folds = 50 train)
    config.LGBM_PARAMS["min_child_samples"] = 5
    # Ensure silent execution
    config.LGBM_PARAMS["verbosity"] = -1

    # Ensure working directory is clean for this run (optional but good for demo)
    # We use a specific demo directory to avoid messing with existing caches if any
    config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 3. Data Processing & Loading
    print("Step 1: Loading and Processing Data...")
    # load_cached_data=False forces the feature engineering pipeline to run
    train_df, val_df, test_df = dp.load_data(load_cached_data=False)

    # 4. Validation of Data Processing
    print("Validating processed data...")

    # Check dimensions
    # Note: In DEBUG mode, we expect roughly DEBUG_SAMPLE_SIZE rows if available
    assert len(train_df) > 0, "Training DataFrame is empty."
    assert len(test_df) > 0, "Test DataFrame is empty."

    # Check for feature columns (should be > 100 given the feature engineering logic)
    feature_cols = mt.get_feature_columns(train_df)
    print(f"Generated {len(feature_cols)} features.")
    assert len(feature_cols) > 0, "No features were extracted."

    # Check for NaNs (Pipeline should handle imputation)
    assert (
        not train_df[feature_cols].isnull().values.any()
    ), "NaNs found in training features."

    # Check target column existence
    assert "time_to_eruption" in train_df.columns, "Target column missing in train_df."

    # 5. Model Training
    print("\nStep 2: Training Model (CV)...")
    models, oof_preds, scores = mt.train_model_cv(train_df)

    # Validate Training Output
    assert (
        len(models) == config.N_FOLDS
    ), f"Expected {config.N_FOLDS} models, got {len(models)}."
    assert len(oof_preds) == len(train_df), "OOF predictions shape mismatch."
    assert len(scores) == config.N_FOLDS, "Scores list length mismatch."

    print(f"Training completed. Average MAE: {np.mean(scores):.4f}")

    # 6. Submission Generation
    print("\nStep 3: Generating Submission...")
    submission_df = mt.generate_submission(models, test_df)

    # Validate Submission Output
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Validate Submission Content
    assert (
        "segment_id" in submission_df.columns
    ), "segment_id column missing in submission."
    assert (
        "time_to_eruption" in submission_df.columns
    ), "time_to_eruption column missing in submission."
    assert len(submission_df) == len(test_df), "Submission row count mismatch."
    assert (
        submission_df["time_to_eruption"] >= 0
    ).all(), "Negative predictions found in submission."

    print("\nDemonstration completed successfully!")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
