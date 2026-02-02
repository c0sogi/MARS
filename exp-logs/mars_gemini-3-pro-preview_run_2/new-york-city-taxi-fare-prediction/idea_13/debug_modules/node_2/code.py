import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings

# Import library modules
# We import them to patch their configurations before usage
import library.config
import library.data_manager
import library.model_trainer
import library.spatial_engine

from library.data_manager import DataManager
from library.model_trainer import ModelTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_small_datasets_and_patch_config():
    """
    Creates small subsets of the original data to ensure the demo runs quickly.
    Patches the library modules to use these small datasets and faster training parameters.
    """
    print("Optimization: Creating small dataset subsets for demonstration...")

    # Define paths for small datasets
    small_train_path = "./working/demo_train_small.parquet"
    small_val_path = "./working/demo_val_small.parquet"

    # Load a small sample of the original data
    # We use the metadata files which are already parquet
    original_train_path = library.config.TRAIN_DATA_PATH
    original_val_path = library.config.VAL_DATA_PATH

    # Read top 20,000 rows for speed
    # PyArrow engine is generally faster
    df_train = pd.read_parquet(original_train_path).head(20000)
    df_val = pd.read_parquet(original_val_path).head(5000)

    # Save small datasets
    df_train.to_parquet(small_train_path, index=False)
    df_val.to_parquet(small_val_path, index=False)

    print(f"Created small train set: {df_train.shape}")
    print(f"Created small val set: {df_val.shape}")

    # --- PATCHING CONFIGURATIONS ---
    # We must update the variables in the modules where they are imported

    # 1. Patch Data Paths in library.data_manager
    library.data_manager.TRAIN_DATA_PATH = small_train_path
    library.data_manager.VAL_DATA_PATH = small_val_path

    # 2. Patch Subsample Size in library.data_manager
    # Set to full size of our small dataset so we don't downsample further
    library.data_manager.SUBSAMPLE_SIZE = 20000

    # 3. Patch XGBoost Parameters in library.model_trainer
    # Reduce estimators and early stopping for speed
    new_xgb_params = library.config.XGB_PARAMS.copy()
    new_xgb_params["n_estimators"] = 100
    new_xgb_params["learning_rate"] = 0.1
    # Ensure we use the GPU if available, otherwise CPU is fine for this small size
    # The prompt says we have an A100, so 'cuda' is correct.

    library.model_trainer.XGB_PARAMS = new_xgb_params
    library.model_trainer.EARLY_STOPPING_ROUNDS = 10
    library.model_trainer.VERBOSE_EVAL = 50

    print("Configuration patched successfully for speed.")


def main():
    set_seed(42)

    # 1. Setup Data and Config
    create_small_datasets_and_patch_config()

    # 2. Initialize Data Manager
    print("\n=== Initializing Data Manager ===")
    dm = DataManager()

    # 3. Prepare Training Data
    # load_cached_data=False ensures we run the logic (SpatialEngine, FeatureEngineer)
    print("\n=== Preparing Training Data ===")
    train_df = dm.prepare_training_data(load_cached_data=False)

    # Validation: Check Training Data
    print("Verifying Training Data...")
    assert not train_df.empty, "Training DataFrame is empty."
    assert (
        "fare_amount" in train_df.columns
    ), "Target 'fare_amount' missing from training data."
    # Check for engineered features
    expected_features = ["dist_haversine", "pickup_rot_lat", "pickup_mean_3"]
    for feat in expected_features:
        assert feat in train_df.columns, f"Expected feature {feat} missing."
    print(f"Training Data Shape: {train_df.shape}")
    print("Training Data Verification Passed.")

    # 4. Prepare Validation Data
    print("\n=== Preparing Validation Data ===")
    # Pass train_df to use its global stats for spatial priors
    val_df = dm.prepare_validation_data(full_train_df=train_df, load_cached_data=False)

    # Validation: Check Validation Data
    print("Verifying Validation Data...")
    assert not val_df.empty, "Validation DataFrame is empty."
    assert (
        "fare_amount" in val_df.columns
    ), "Target 'fare_amount' missing from validation data."
    # Allow for rows dropped during cleaning (non-positive fares)
    assert (
        val_df.shape[0] <= 5000
    ), f"Expected <= 5000 validation rows, got {val_df.shape[0]}"
    print(f"Validation Data Shape: {val_df.shape}")
    print("Validation Data Verification Passed.")

    # 5. Prepare Test Data
    print("\n=== Preparing Test Data ===")
    test_df = dm.prepare_test_data(full_train_df=train_df, load_cached_data=False)

    # Validation: Check Test Data
    print("Verifying Test Data...")
    assert not test_df.empty, "Test DataFrame is empty."
    assert (
        "fare_amount" not in test_df.columns
    ), "Target 'fare_amount' should not be in test data."
    assert "key" in test_df.columns, "ID column 'key' missing from test data."
    # Check if spatial priors were applied (should not have NaNs if logic is correct,
    # though global mean fill handles it)
    assert (
        not test_df["pickup_mean_3"].isnull().any()
    ), "NaNs found in spatial features."
    print(f"Test Data Shape: {test_df.shape}")
    print("Test Data Verification Passed.")

    # 6. Initialize Model Trainer
    print("\n=== Initializing Model Trainer ===")
    trainer = ModelTrainer()

    # 7. Train Model
    print("\n=== Training XGBoost Model ===")
    model = trainer.train_xgboost(train_df, val_df)

    # Validation: Check Model
    assert model is not None, "Model training failed to return a model object."
    assert os.path.exists(
        trainer.model_path
    ), f"Model file not found at {trainer.model_path}"
    print("Model trained and saved successfully.")

    # 8. Generate Predictions
    print("\n=== Generating Predictions ===")
    predictions = trainer.predict_and_postprocess(model, test_df)

    # Validation: Check Predictions
    print("Verifying Predictions...")
    assert len(predictions) == len(test_df), "Prediction count mismatch."
    assert np.all(
        predictions >= 2.50
    ), "Post-processing failed: Fares below $2.50 found."
    print(f"Predictions generated: {len(predictions)}")
    print(f"Mean Predicted Fare: ${predictions.mean():.2f}")

    # 9. Create Submission
    print("\n=== Creating Submission File ===")
    trainer.generate_submission(test_df, predictions)

    # Validation: Check Submission File
    submission_path = library.config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Read back submission to verify format
    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == [
        "key",
        "fare_amount",
    ], "Submission columns mismatch."
    assert len(sub_df) == len(test_df), "Submission row count mismatch."
    print(f"Submission file verified at {submission_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
