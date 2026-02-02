import os
import sys
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import TRAIN_METADATA_PATH, TEST_METADATA_PATH, SUBMISSION_DIR
from library.geometry_utils import extract_geometry_features
from library.feature_engineering import build_feature_matrix
from library.model_handler import EnergyPredictor


def run_demo():
    print("Starting demonstration of library modules...")

    # -------------------------------------------------------------------------
    # 1. Prepare Data Subsets
    # -------------------------------------------------------------------------
    print("\n[1] Loading and subsetting metadata...")
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_METADATA_PATH}")

    full_train = pd.read_csv(TRAIN_METADATA_PATH)
    full_test = pd.read_csv(TEST_METADATA_PATH)

    # Create small subsets for demonstration speed
    # 20 samples for training, 10 for validation, 5 for testing
    demo_train = full_train.iloc[:20].copy()
    demo_val = full_train.iloc[20:30].copy()
    demo_test = full_test.iloc[:5].copy()

    print(f"Demo Train size: {len(demo_train)}")
    print(f"Demo Val size: {len(demo_val)}")
    print(f"Demo Test size: {len(demo_test)}")

    # -------------------------------------------------------------------------
    # 2. Demonstrate Geometry Utils
    # -------------------------------------------------------------------------
    print("\n[2] Testing geometry_utils.extract_geometry_features...")
    # Force recompute (load_cached_data=False) to test the extraction logic
    # We use a custom cache name to avoid overwriting the main cache with this subset
    geo_features = extract_geometry_features(
        demo_train, load_cached_data=False, cache_name="demo_geometry_features"
    )

    # Verification
    print("Geometry features shape:", geo_features.shape)
    expected_cols = ["volume", "density", "num_atoms", "mean_bond_Al-O"]
    for col in expected_cols:
        if col not in geo_features.columns:
            raise AssertionError(f"Expected column {col} missing in geometry features.")

    if len(geo_features) != len(demo_train):
        raise AssertionError(
            f"Expected {len(demo_train)} rows, got {len(geo_features)}"
        )

    # Check for valid values (volume should be > 0)
    if (geo_features["volume"] <= 0).any():
        raise AssertionError("Invalid volume detected (<= 0).")

    print("Geometry features extraction successful.")
    print(geo_features.head(2))

    # -------------------------------------------------------------------------
    # 3. Demonstrate Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[3] Testing feature_engineering.build_feature_matrix...")
    # This function combines tabular data, geometry features, and elemental moments
    X_demo = build_feature_matrix(
        demo_train, "demo_train_subset", load_cached_data=False
    )

    print(f"Feature matrix built with shape: {X_demo.shape}")

    # Verification
    # Check for elemental moments (computed in feature_engineering)
    if "mean_electronegativity" not in X_demo.columns:
        raise AssertionError("Elemental moments missing (mean_electronegativity).")

    # Check for spacegroup one-hot encoding (e.g., sg_12 is a common one)
    if "sg_12" not in X_demo.columns:
        raise AssertionError("Spacegroup one-hot encoding missing.")

    # Check that geometry features were merged correctly
    if "density" not in X_demo.columns:
        raise AssertionError(
            "Geometry features (density) not merged into feature matrix."
        )

    # Check for NaNs (should be handled or minimal)
    if X_demo.isnull().any().any():
        print(
            "Warning: NaNs found in feature matrix. This might happen if bond stats are missing for some elements."
        )

    # -------------------------------------------------------------------------
    # 4. Demonstrate Model Handler (Training & Inference)
    # -------------------------------------------------------------------------
    print("\n[4] Testing model_handler.EnergyPredictor...")
    predictor = EnergyPredictor()

    # Optimize for speed: Reduce n_estimators for the XGBoost models
    # We access the internal XGBRegressor objects directly to set parameters
    predictor.model_formation.set_params(n_estimators=5, max_depth=3)
    predictor.model_bandgap.set_params(n_estimators=5, max_depth=3)

    print("Training models on demo subset...")
    # Note: This will create/overwrite 'train_combined_features.parquet' and 'val_combined_features.parquet'
    # in the working directory because the split names are hardcoded in the train method.
    predictor.train(demo_train, demo_val, load_cached_data=False)

    print("Running inference on demo test set...")
    preds = predictor.predict(demo_test, load_cached_data=False)

    # Verification
    if len(preds) != len(demo_test):
        raise AssertionError("Prediction count mismatch.")

    required_targets = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    for tgt in required_targets:
        if tgt not in preds.columns:
            raise AssertionError(f"Missing target column {tgt} in predictions.")
        # Check for non-negative values (physical constraint enforced in predict)
        if (preds[tgt] < 0).any():
            raise AssertionError(f"Negative predictions found for {tgt}.")

    print("Predictions head:")
    print(preds.head())

    # -------------------------------------------------------------------------
    # 5. Generate Submission
    # -------------------------------------------------------------------------
    print("\n[5] Testing generate_submission...")
    demo_submission_path = os.path.join(SUBMISSION_DIR, "demo_submission.csv")

    # This runs predict and saves to CSV
    predictor.generate_submission(
        demo_test, demo_submission_path, load_cached_data=False
    )

    if not os.path.exists(demo_submission_path):
        raise AssertionError(
            f"Submission file was not created at {demo_submission_path}"
        )

    # Verify file content
    sub_df = pd.read_csv(demo_submission_path)
    if sub_df.shape != (5, 3):
        raise AssertionError(
            f"Submission file has incorrect shape: {sub_df.shape}, expected (5, 3)"
        )

    print(f"Demo completed successfully. Output at {demo_submission_path}")


if __name__ == "__main__":
    run_demo()
