import os
import sys
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
import library.utils as utils
import library.data_loader as data_loader
import library.model_trainer as model_trainer
import library.config as config


def main():
    print("Initializing demonstration...")

    # 1. Setup and Reproducibility
    # Set random seeds for consistency
    utils.seed_everything(seed=42)

    # Define demo parameters
    DEBUG_SIZE = 20  # Small subset for speed
    N_SPLITS = 2  # Minimal folds for CV demo

    # Patch hyperparameters in the model_trainer module to ensure quick execution.
    # The library imports these values from config, so we must update them
    # in the model_trainer namespace where they are used.
    print(f"Patching hyperparameters for speed (N_ESTIMATORS=10, EARLY_STOPPING=5)...")
    model_trainer.N_ESTIMATORS = 10
    model_trainer.EARLY_STOPPING_ROUNDS = 5
    model_trainer.VERBOSITY = -1

    # Ensure working directory for cache exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # 2. Data Loading & Feature Engineering
    # ----------------------------------------------------------------
    print("\n--- Step 1: Data Loading & Feature Extraction ---")

    # Load Training Data (Subset)
    # This triggers the parallel feature extraction pipeline in library/feature_engineering.py
    print(f"Generating training dataset (debug_size={DEBUG_SIZE})...")
    train_df = data_loader.create_dataset(
        split="train",
        load_cached_data=False,  # Force re-computation to demonstrate the pipeline
        debug_size=DEBUG_SIZE,
        n_jobs=2,  # Use 2 cores for demo to avoid overhead
    )

    # Validation of Train Data
    print("Validating training data structure...")
    assert isinstance(train_df, pd.DataFrame), "train_df should be a DataFrame"
    assert (
        len(train_df) == DEBUG_SIZE
    ), f"Expected {DEBUG_SIZE} rows, got {len(train_df)}"
    assert "segment_id" in train_df.columns, "Missing 'segment_id' column"
    assert "time_to_eruption" in train_df.columns, "Missing target 'time_to_eruption'"
    # Check for some expected feature columns from feature_engineering.py
    expected_feats = ["spatial_mean_std", "spatial_rms_std"]
    for feat in expected_feats:
        assert feat in train_df.columns, f"Missing expected feature: {feat}"

    print(f"Train DataFrame shape: {train_df.shape}")

    # Load Test Data (Subset)
    print(f"Generating test dataset (debug_size={DEBUG_SIZE})...")
    test_df = data_loader.create_dataset(
        split="test", load_cached_data=False, debug_size=DEBUG_SIZE, n_jobs=2
    )

    # Validation of Test Data
    print("Validating test data structure...")
    assert len(test_df) == DEBUG_SIZE, f"Expected {DEBUG_SIZE} rows, got {len(test_df)}"
    assert (
        "time_to_eruption" not in test_df.columns
    ), "Test set should not have target column"

    print(f"Test DataFrame shape: {test_df.shape}")

    # ----------------------------------------------------------------
    # 3. Model Training (Cross-Validation)
    # ----------------------------------------------------------------
    print("\n--- Step 2: Model Training (Cross-Validation) ---")

    # Run Stratified K-Fold CV
    # This uses the patched hyperparameters for speed
    models, oof_df, metrics = model_trainer.run_cross_validation(
        train_df, n_splits=N_SPLITS
    )

    # Validation of Training Results
    print("Validating training results...")
    assert len(models) == N_SPLITS, f"Expected {N_SPLITS} models, got {len(models)}"
    assert len(oof_df) == len(train_df), "OOF predictions length mismatch"
    assert "overall_mae" in metrics, "Metrics dictionary missing 'overall_mae'"
    assert metrics["overall_mae"] > 0, "MAE should be positive"

    print(f"Cross-Validation completed. Overall MAE: {metrics['overall_mae']:.4f}")

    # ----------------------------------------------------------------
    # 4. Prediction & Submission
    # ----------------------------------------------------------------
    print("\n--- Step 3: Generating Predictions ---")

    # Generate average predictions using all trained models
    submission_df = model_trainer.generate_predictions(models, test_df)

    # Validation of Submission
    print("Validating submission structure...")
    assert len(submission_df) == len(test_df), "Submission length mismatch"
    assert "segment_id" in submission_df.columns, "Submission missing 'segment_id'"
    assert (
        "time_to_eruption" in submission_df.columns
    ), "Submission missing 'time_to_eruption'"
    assert not submission_df.isnull().values.any(), "Submission contains null values"

    # Save submission (simulating the final step)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
