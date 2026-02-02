import os
import sys
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.data import load_and_featurize_data, TargetTransformer
from library.model import DualTargetRegressor
from library.features import StructureFeaturizer


def run_demo():
    print("Starting demonstration of library usage...")

    # 1. Configure for Speed
    # Override Config parameters to ensure the demo runs quickly
    print("Configuring parameters for fast execution...")
    Config.DEBUG_SAMPLE_SIZE = 20  # Process only 20 samples per dataset

    # Reduce XGBoost estimators for speed
    Config.XGB_MODEL_PARAMS["n_estimators"] = 10
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = False  # Suppress XGBoost logs

    # Ensure working directory exists (handled by library, but good practice)
    if not os.path.exists(Config.WORKING_DIR):
        os.makedirs(Config.WORKING_DIR)

    # 2. Demonstrate Feature Extraction Logic (Unit Level)
    # We pick one file from the training set to test StructureFeaturizer directly
    print("\n--- Testing StructureFeaturizer on a single file ---")
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_file_path = train_meta.iloc[0]["file_path"]

    featurizer = StructureFeaturizer()
    features = featurizer.process_single_structure(sample_file_path)

    # Verification
    assert features is not None, "Featurizer returned None for a valid file."
    assert "vol_per_atom" in features, "Physical property 'vol_per_atom' missing."
    assert "density" in features, "Physical property 'density' missing."
    # Check for some RDF keys (assuming Al is present in CATIONS)
    has_rdf = any(k.startswith("RDF_") for k in features.keys())
    assert has_rdf, "No RDF features generated."
    # Check for Steinhardt keys
    has_steinhardt = any(k.startswith("Steinhardt_") for k in features.keys())
    assert has_steinhardt, "No Steinhardt features generated."

    print(f"Successfully extracted {len(features)} features from {sample_file_path}.")

    # 3. Demonstrate Data Loading and Pipeline Featurization
    print("\n--- Running Data Loading and Featurization Pipeline ---")
    # We set load_cached_data=False to force computation
    train_df, val_df, test_df = load_and_featurize_data(
        debug_sample=Config.DEBUG_SAMPLE_SIZE, load_cached_data=False
    )

    # Verification
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train df size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(train_df)}"
    assert (
        len(val_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Val df size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(val_df)}"
    assert (
        len(test_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Test df size mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(test_df)}"

    # Check if targets exist in train/val
    for target in Config.TARGET_COLS:
        assert target in train_df.columns, f"Target {target} missing in train_df"
        assert target in val_df.columns, f"Target {target} missing in val_df"

    # 4. Demonstrate Target Transformation
    print("\n--- Testing TargetTransformer ---")
    transformer = TargetTransformer()
    original_values = np.array([0.0, 1.0, 10.0])
    transformed = transformer.transform(original_values)
    recovered = transformer.inverse_transform(transformed)

    assert np.allclose(
        original_values, recovered
    ), "TargetTransformer failed round-trip conversion."
    print("TargetTransformer logic verified.")

    # 5. Demonstrate Model Training
    print("\n--- Training DualTargetRegressor ---")
    model = DualTargetRegressor()

    # Fit the model
    # Note: The library implementation prints evaluation metrics automatically
    model.fit(train_df, val_df)

    # Verify internal state
    assert len(model.models) == 2, "Model did not train regressors for both targets."
    assert model.feature_cols is not None, "Feature columns were not identified."
    print("Model training completed.")

    # 6. Demonstrate Prediction
    print("\n--- Generating Predictions ---")
    predictions = model.predict(test_df)

    # Verification
    print("Predictions head:")
    print(predictions.head())

    assert "id" in predictions.columns, "Predictions missing 'id' column."
    assert len(predictions) == len(test_df), "Prediction count mismatch."
    for target in Config.TARGET_COLS:
        assert (
            target in predictions.columns
        ), f"Prediction missing target column {target}."
        assert (
            not predictions[target].isnull().any()
        ), f"NaN values found in predictions for {target}."
        # Basic sanity check: energies should be non-negative (mostly) or reasonable
        # Formation energy can be negative, bandgap usually positive.
        # Since we use log1p transform, inverse transform ensures > -1.
        pass

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
