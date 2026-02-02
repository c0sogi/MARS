import os
import sys
import numpy as np
import pandas as pd
import ase
import xgboost as xgb

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

# Import from library
from library import config
from library import utils
from library.data_loader import DataLoader
from library.descriptors import (
    get_global_descriptors,
    get_rdf_fingerprints,
    get_local_sublattice_fingerprints,
)
from library.feature_pipeline import generate_features, clean_features
from library.model import train_model, predict_model


def run_demo():
    print("Starting demonstration...")

    # -------------------------------------------------------------------------
    # 1. Test Utils
    # -------------------------------------------------------------------------
    print("\n--- Testing Utils ---")
    val = 10.0
    log_val = utils.log_transform(val)
    inv_val = utils.inverse_log_transform(log_val)
    assert np.isclose(val, inv_val), f"Log transform inverse failed: {val} vs {inv_val}"

    arr = [1, 2, 3, 4, 5]
    mean_val = utils.safe_mean(arr)
    assert np.isclose(mean_val, 3.0), f"Safe mean failed: {mean_val}"

    empty_arr = []
    assert np.isnan(utils.safe_mean(empty_arr)), "Safe mean of empty should be NaN"
    print("Utils tests passed.")

    # -------------------------------------------------------------------------
    # 2. Test Data Loader
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loader ---")
    loader = DataLoader()
    # Load a small sample using debug mode
    train_meta_sample = loader.get_train_data(debug=True, sample_size=10)
    assert not train_meta_sample.empty, "Train metadata sample is empty"
    assert "file_path" in train_meta_sample.columns, "file_path column missing"

    # Load geometry for the first entry
    first_path = train_meta_sample.iloc[0]["file_path"]
    atoms = loader.load_geometry(first_path)
    assert isinstance(atoms, ase.Atoms), "Failed to load geometry as ase.Atoms"
    print(f"Loaded geometry for {first_path} with {len(atoms)} atoms.")
    print("Data Loader tests passed.")

    # -------------------------------------------------------------------------
    # 3. Test Descriptors
    # -------------------------------------------------------------------------
    print("\n--- Testing Descriptors ---")
    # Global descriptors
    global_desc = get_global_descriptors(atoms)
    assert "Global_Volume" in global_desc
    assert "Global_Density" in global_desc
    assert global_desc["Global_NumAtoms"] > 0

    # RDF descriptors
    rdf_desc = get_rdf_fingerprints(atoms, cutoff=4.0, bins=5)
    # Check if we got some RDF features (keys start with RDF_)
    assert any(
        k.startswith("RDF_") for k in rdf_desc.keys()
    ), "No RDF features generated"

    # Local descriptors
    local_desc = get_local_sublattice_fingerprints(atoms, cutoff=3.0)
    # Check for Cation/Anion features
    assert any(
        k.startswith("Cation_") or k.startswith("Anion_") for k in local_desc.keys()
    ), "No local features generated"

    total_feats = len(global_desc) + len(rdf_desc) + len(local_desc)
    print(f"Generated {total_feats} descriptors for one structure.")
    print("Descriptor tests passed.")

    # -------------------------------------------------------------------------
    # 4. Test Feature Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Testing Feature Pipeline ---")
    # Generate features for train and val (small subset)
    # We use debug=True to limit the number of samples processed

    print("Generating training features (debug mode)...")
    # Force recomputation (load_cached_data=False) to verify logic
    df_train_feats = generate_features(
        split="train", load_cached_data=False, debug=True, sample_size=20
    )

    print("Generating validation features (debug mode)...")
    df_val_feats = generate_features(
        split="val", load_cached_data=False, debug=True, sample_size=10
    )

    print("Generating test features (debug mode)...")
    df_test_feats = generate_features(
        split="test", load_cached_data=False, debug=True, sample_size=10
    )

    # Clean features (drop constant columns based on train)
    df_train_clean, df_val_clean, df_test_clean = clean_features(
        df_train_feats, df_val_feats, df_test_feats
    )

    assert df_train_clean.shape[0] == 20
    assert df_val_clean.shape[0] == 10
    assert df_test_clean.shape[0] == 10

    # Ensure targets are present in train/val
    for target in config.TRAIN_CONFIG["target_cols"]:
        assert (
            target in df_train_clean.columns
        ), f"Target {target} missing from training features"
        assert (
            target in df_val_clean.columns
        ), f"Target {target} missing from validation features"

    print("Feature Pipeline tests passed.")

    # -------------------------------------------------------------------------
    # 5. Test Model Training
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Training ---")
    # The train_model function uses XGB_PARAMS from config.
    # With 20 samples, it should run quickly.
    models = train_model(df_train_clean, df_val_clean)

    assert len(models) == len(
        config.TRAIN_CONFIG["target_cols"]
    ), "Did not train models for all targets"
    for target, model in models.items():
        assert isinstance(
            model, xgb.XGBRegressor
        ), f"Model for {target} is not an XGBRegressor"

    print("Model Training tests passed.")

    # -------------------------------------------------------------------------
    # 6. Test Prediction
    # -------------------------------------------------------------------------
    print("\n--- Testing Prediction ---")

    predictions = predict_model(models, df_test_clean)

    assert "id" in predictions.columns
    for target in config.TRAIN_CONFIG["target_cols"]:
        assert target in predictions.columns

    assert len(predictions) == 10
    print("Predictions head:")
    print(predictions.head())
    print("Prediction tests passed.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
