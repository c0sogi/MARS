import os
import sys
import shutil
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(".")

from library import config, utils, features, model


def clean_cache():
    """
    Removes cached parquet files to ensure the demo runs on the
    freshly configured debug subset.
    """
    if os.path.exists(config.WORKING_DIR):
        for f in os.listdir(config.WORKING_DIR):
            if f.endswith(".parquet"):
                try:
                    os.remove(os.path.join(config.WORKING_DIR, f))
                    print(f"Cleaned cache: {f}")
                except OSError as e:
                    print(f"Error deleting {f}: {e}")


def configure_demo():
    """
    Overwrites configuration for a fast, reproducible debug run.
    """
    print("Configuring for Demo Run...")

    # Enable Debug Mode
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 20  # Small subset for speed

    # Reduce Cross-Validation Folds
    config.N_FOLDS = 2

    # Optimize LightGBM for Speed
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["learning_rate"] = 0.1
    config.LGBM_PARAMS["num_leaves"] = 15
    config.LGBM_PARAMS["early_stopping_rounds"] = 5
    config.LGBM_PARAMS["verbosity"] = -1

    # Set Seeds
    np.random.seed(config.SEED)


def verify_utils():
    print("\n--- Verifying Utils ---")

    # Test Metadata Loading
    train_meta = utils.load_metadata("train")
    print(f"Loaded train metadata shape: {train_meta.shape}")

    assert (
        len(train_meta) == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} samples, got {len(train_meta)}"
    assert "segment_id" in train_meta.columns
    assert "file_path" in train_meta.columns

    # Test Sensor File Reading
    sample_path = train_meta.iloc[0]["file_path"]
    print(f"Reading sample file: {sample_path}")

    sensor_df = utils.read_sensor_file(sample_path)
    print(f"Sensor DataFrame shape: {sensor_df.shape}")

    assert sensor_df.shape == (
        config.SIGNAL_LENGTH,
        config.NUM_SENSORS,
    ), f"Shape mismatch. Expected {(config.SIGNAL_LENGTH, config.NUM_SENSORS)}, got {sensor_df.shape}"

    print("Utils verification passed.")
    return train_meta


def verify_features(sample_meta):
    print("\n--- Verifying Features ---")

    sample_file = sample_meta.iloc[0]["file_path"]

    # Test Single Segment Processing
    print("Processing single segment...")
    feats = features.process_segment(sample_file)

    assert isinstance(feats, dict), "process_segment should return a dictionary"
    assert len(feats) > 0, "Feature dictionary is empty"

    # Check for specific feature existence (e.g., from View A)
    test_key = "sensor_1_trend_mean"
    assert test_key in feats, f"Expected feature '{test_key}' not found."

    print(f"Extracted {len(feats)} features from one segment.")
    print("Features verification passed.")


def verify_model():
    print("\n--- Verifying Model Pipeline ---")

    manager = model.EnsembleManager()

    # Verify parameter override
    assert manager.params["n_estimators"] == 10, "LGBM params not updated correctly"

    # Run Training Loop
    print("Starting Training Loop (this triggers feature generation)...")
    manager.train_loop()

    assert (
        len(manager.models) == config.N_FOLDS
    ), f"Expected {config.N_FOLDS} trained models, found {len(manager.models)}"

    # Run Inference
    print("Starting Inference...")
    manager.predict_average()

    # Verify Submission
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    expected_cols = ["segment_id", "time_to_eruption"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Got {sub_df.columns}"

    # Check that we have rows equal to debug sample size (or total test size if smaller)
    # Note: utils.load_metadata('test') is called inside predict_average with debug=True
    # so it loads DEBUG_SAMPLE_SIZE rows.
    assert (
        len(sub_df) == config.DEBUG_SAMPLE_SIZE
    ), f"Submission length mismatch. Expected {config.DEBUG_SAMPLE_SIZE}, got {len(sub_df)}"

    print("Model pipeline verification passed.")


if __name__ == "__main__":
    try:
        configure_demo()
        clean_cache()

        train_meta = verify_utils()
        verify_features(train_meta)
        verify_model()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        raise e
