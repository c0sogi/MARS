import os
import sys
import numpy as np
import pandas as pd
import shutil
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
import library.config as config
import library.data_utils as data_utils
import library.feature_engineering as fe
import library.model_trainer as trainer


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 0. Setup & Configuration Overrides
    # ==========================================
    print("\n[Step 0] Configuring environment for rapid demonstration...")
    set_seed(config.SEED)

    # Override config for speed
    config.N_FOLDS = 2
    config.DEBUG_SAMPLE_SIZE = 10  # Only process 10 files per split
    config.LGBM_PARAMS["n_estimators"] = 20
    config.LGBM_PARAMS["early_stopping_rounds"] = 5
    config.LGBM_PARAMS["verbose"] = -1

    # Ensure working directory is clean for the demo to prove file generation
    if os.path.exists(config.WORKING_DIR):
        # We don't delete the whole dir to avoid permission issues, just specific cache files if they exist
        for f in [
            config.TRAIN_FEATURES_PATH,
            config.VAL_FEATURES_PATH,
            config.TEST_FEATURES_PATH,
        ]:
            if os.path.exists(f):
                os.remove(f)
    else:
        os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(
        f"Configuration updated: N_FOLDS={config.N_FOLDS}, DEBUG_SAMPLE_SIZE={config.DEBUG_SAMPLE_SIZE}"
    )

    # ==========================================
    # 1. Data Utilities Verification
    # ==========================================
    print("\n[Step 1] Verifying Data Utilities (library.data_utils)...")

    # Test load_metadata
    train_meta = data_utils.load_metadata("train")
    print(f"Loaded train metadata. Shape: {train_meta.shape}")

    assert not train_meta.empty, "Train metadata should not be empty."
    assert "segment_id" in train_meta.columns, "Metadata missing 'segment_id'."
    assert "file_path" in train_meta.columns, "Metadata missing 'file_path'."

    # Test load_sensor_segment
    # Pick the first file from metadata
    sample_file_path = train_meta.iloc[0]["file_path"]
    segment_id = train_meta.iloc[0]["segment_id"]
    print(f"Loading sample sensor segment: {segment_id} ({sample_file_path})")

    sensor_df = data_utils.load_sensor_segment(sample_file_path)
    print(f"Sensor data loaded. Shape: {sensor_df.shape}")

    # Assertions
    assert sensor_df.shape[0] == 60001, f"Expected 60001 rows, got {sensor_df.shape[0]}"
    assert (
        sensor_df.shape[1] == 10
    ), f"Expected 10 sensor columns, got {sensor_df.shape[1]}"
    assert sensor_df.dtypes.iloc[0] == "float32", "Data should be loaded as float32."

    # Test save/load features (Parquet utils)
    dummy_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    dummy_path = os.path.join(config.WORKING_DIR, "dummy_test.parquet")

    data_utils.save_features(dummy_df, dummy_path)
    assert os.path.exists(dummy_path), "save_features failed to create file."

    loaded_dummy = data_utils.load_features(dummy_path)
    pd.testing.assert_frame_equal(dummy_df, loaded_dummy)
    print("Data utilities verified successfully.")

    # ==========================================
    # 2. Feature Engineering Verification
    # ==========================================
    print("\n[Step 2] Verifying Feature Engineering (library.feature_engineering)...")

    # Test single row processing logic first (unit test style)
    # We use the sensor_df loaded in Step 1
    # We create a mock row series to simulate what process_row expects
    mock_row = train_meta.iloc[0]

    print("Testing single segment processing...")
    # We call process_row directly. Note: process_row loads the file internally based on the row info.
    # Since we are calling the library function, it will do the I/O.
    features_dict = fe.process_row(mock_row)

    assert features_dict is not None, "process_row returned None."
    assert "segment_id" in features_dict, "Features missing segment_id."
    assert "time_to_eruption" in features_dict, "Features missing target."
    # Check for some expected feature keys (e.g., from sensor_1)
    assert any(
        k.startswith("sensor_1_trend") for k in features_dict.keys()
    ), "Missing trend features."
    assert any(
        k.startswith("sensor_1_psd") for k in features_dict.keys()
    ), "Missing spectral features."

    print(f"Single segment features extracted. Count: {len(features_dict)}")

    # Test Dataset Processing (Batch)
    # We use debug=True to only process DEBUG_SAMPLE_SIZE (10) rows
    print("Testing batch processing (process_dataset) in DEBUG mode...")

    # Force load_cached_data=False to ensure calculation happens
    train_features_df = fe.process_dataset("train", load_cached_data=False, debug=True)

    print(f"Processed train features shape: {train_features_df.shape}")
    assert (
        len(train_features_df) <= config.DEBUG_SAMPLE_SIZE
    ), "Debug mode did not limit sample size."
    assert (
        "time_to_eruption" in train_features_df.columns
    ), "Target column missing in train features."

    # Check for NaNs (imputation should have handled raw data, but let's check output)
    if train_features_df.isnull().sum().sum() > 0:
        print("Warning: NaNs found in feature matrix. Checking columns...")
        # In a real scenario, we might raise an error, but for demo we just note it.
        # LightGBM handles NaNs, but feature engineering should ideally be clean.

    print("Feature engineering verified successfully.")

    # ==========================================
    # 3. Model Training Verification
    # ==========================================
    print("\n[Step 3] Verifying Model Training (library.model_trainer)...")

    # run_cross_validation will call process_dataset internally.
    # Since we cleared cache at start and ran process_dataset with debug=True above (but didn't save to main cache path because of debug flag),
    # we run CV with debug=True so it generates its own small dataset on the fly.

    print("Running Cross-Validation...")
    models, avg_mae = trainer.run_cross_validation(load_cached_data=False, debug=True)

    print(f"CV Completed. Average MAE: {avg_mae}")

    assert (
        len(models) == config.N_FOLDS
    ), f"Expected {config.N_FOLDS} models, got {len(models)}."
    assert isinstance(avg_mae, float), "MAE should be a float."
    assert avg_mae > 0, "MAE should be positive."

    # Check if model was saved
    assert os.path.exists(
        config.MODEL_PATH
    ), f"Model file not found at {config.MODEL_PATH}"
    print("Model training verified successfully.")

    # ==========================================
    # 4. Prediction & Submission Verification
    # ==========================================
    print("\n[Step 4] Verifying Prediction and Submission...")

    # Predict on test set (debug mode)
    trainer.predict_and_submit(models, load_cached_data=False, debug=True)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created."

    # Validate submission format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print("Head:\n", sub_df.head())

    assert list(sub_df.columns) == [
        "segment_id",
        "time_to_eruption",
    ], "Invalid submission columns."
    assert not sub_df.isnull().values.any(), "Submission contains NaNs."
    assert len(sub_df) > 0, "Submission is empty."

    # In debug mode for test, we expect DEBUG_SAMPLE_SIZE rows (or less if test set is smaller)
    expected_len = min(len(data_utils.load_metadata("test")), config.DEBUG_SAMPLE_SIZE)
    # Note: process_dataset samples randomly, so exact length match depends on implementation details,
    # but it should be close to DEBUG_SAMPLE_SIZE.
    assert (
        len(sub_df) == expected_len
    ), f"Expected {expected_len} predictions, got {len(sub_df)}."

    print("Prediction and submission verified successfully.")

    print("\nAll library components demonstrated and verified successfully!")


if __name__ == "__main__":
    run_demo()
