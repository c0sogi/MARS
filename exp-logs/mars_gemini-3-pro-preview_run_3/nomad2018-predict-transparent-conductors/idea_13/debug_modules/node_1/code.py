import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb

# Import from the provided library files
import library.config as config
from library.model_trainer import run_training_pipeline, generate_submission_file
from library.geometry_utils import process_single_structure


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Optimize Hyperparameters for Speed (Monkey Patching)
    # The default configuration has 3000 estimators which is too slow for a quick demo.
    # We reduce it significantly to ensure the script completes quickly.
    print("Adjusting configuration for rapid execution...")
    config.XGB_PARAMS["n_estimators"] = 10
    config.EARLY_STOPPING_ROUNDS = 5

    # We will also use a very small subset of data
    SAMPLE_SIZE_TRAIN = 50
    SAMPLE_SIZE_TEST = 20

    # Ensure directories exist
    config.setup_directories()

    # 2. Verify Geometry Processing Logic
    # We manually process one file to ensure the geometry utils are working correctly
    print("\n--- Verifying Geometry Processing ---")
    # Pick a random file from train metadata to test
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_file_rel_path = train_meta.iloc[0]["file_path"]
    sample_file_full_path = os.path.join(config.INPUT_DIR, sample_file_rel_path)

    if os.path.exists(sample_file_full_path):
        print(f"Processing single structure: {sample_file_full_path}")
        features = process_single_structure(sample_file_full_path)

        # Assertions to check feature extraction
        assert features is not None, "Feature extraction returned None"
        assert "phys_volume" in features, "Physical volume missing"
        assert "phys_density" in features, "Physical density missing"
        # Check for RDF features (at least one)
        rdf_keys = [k for k in features.keys() if k.startswith("RDF_")]
        assert len(rdf_keys) > 0, "No RDF features generated"
        # Check for ADF features (at least one)
        adf_keys = [k for k in features.keys() if k.startswith("ADF_")]
        assert len(adf_keys) > 0, "No ADF features generated"

        print(f"Successfully extracted {len(features)} features from sample.")
    else:
        print(
            f"Warning: Sample file {sample_file_full_path} not found. Skipping single file check."
        )

    # 3. Run Training Pipeline
    # This handles data loading, feature extraction (with caching), and model training
    print("\n--- Running Training Pipeline ---")
    # We set load_cached_data=False to force the feature extraction logic to run on our sample subset
    models, feature_cols = run_training_pipeline(
        sample_size=SAMPLE_SIZE_TRAIN, load_cached_data=False
    )

    # Validation of Training Output
    assert len(models) == len(
        config.TARGET_COLS
    ), f"Expected {len(config.TARGET_COLS)} models, got {len(models)}"
    for target in config.TARGET_COLS:
        assert target in models, f"Model for target '{target}' is missing"
        assert isinstance(
            models[target], xgb.XGBRegressor
        ), f"Model for '{target}' is not an XGBRegressor"

    assert len(feature_cols) > 0, "No feature columns identified"
    print("Training pipeline completed successfully.")

    # 4. Generate Submission
    # This handles test data loading, feature extraction, inference, and saving CSV
    print("\n--- Generating Submission ---")
    generate_submission_file(
        models, feature_cols, sample_size=SAMPLE_SIZE_TEST, load_cached_data=False
    )

    # Validation of Submission Output
    if os.path.exists(config.SUBMISSION_PATH):
        submission_df = pd.read_csv(config.SUBMISSION_PATH)

        # Check shape (should match sample size)
        assert (
            len(submission_df) == SAMPLE_SIZE_TEST
        ), f"Submission rows {len(submission_df)} mismatch expected {SAMPLE_SIZE_TEST}"

        # Check columns
        expected_cols = ["id"] + config.TARGET_COLS
        for col in expected_cols:
            assert col in submission_df.columns, f"Column {col} missing from submission"

        # Check for non-null values
        assert not submission_df.isnull().values.any(), "Submission contains NaN values"

        # Check values are positive (since they are energies/bandgaps)
        # Note: Formation energy can theoretically be negative, but bandgap is usually positive.
        # Our inverse transform (expm1) ensures > -1.
        # Let's just check they are numeric.
        assert pd.api.types.is_numeric_dtype(
            submission_df["bandgap_energy_ev"]
        ), "Bandgap predictions are not numeric"

        print("Submission file validated successfully.")
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {config.SUBMISSION_PATH}"
        )

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
