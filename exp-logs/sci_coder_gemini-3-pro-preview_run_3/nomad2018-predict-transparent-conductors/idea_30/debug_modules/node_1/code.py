import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

# Import functions and constants from the provided library files
from library.config import RANDOM_SEED, SUBMISSION_PATH, TARGET_COLS
from library.data_handler import load_metadata
from library.feature_extractor import process_structure
from library.model_engine import train_model, generate_predictions

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)


def run_demo():
    print(
        "Starting demonstration of the Formation & Bandgap Energy Prediction Pipeline..."
    )

    # ---------------------------------------------------------
    # 1. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[1/5] Testing Data Handler (Loading Metadata)...")
    try:
        train_meta = load_metadata("train")
        print(f"   - Successfully loaded train metadata. Shape: {train_meta.shape}")

        # Basic assertions to ensure metadata is correct
        assert not train_meta.empty, "Train metadata is empty."
        assert "file_path" in train_meta.columns, "Metadata missing 'file_path' column."
        assert all(
            col in train_meta.columns for col in TARGET_COLS
        ), f"Metadata missing target columns: {TARGET_COLS}"
        print("   - Metadata structure verified.")
    except Exception as e:
        print(f"   - Error loading metadata: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. Feature Extraction Logic Verification
    # ---------------------------------------------------------
    print("\n[2/5] Testing Feature Extraction on a Single Structure...")
    # Pick the first sample from the training set
    sample_file_path = train_meta.iloc[0]["file_path"]
    print(f"   - Processing sample file: {sample_file_path}")

    try:
        # Extract features using the library function
        features = process_structure(sample_file_path)

        # Verify that features were actually extracted
        assert features is not None, "Feature extraction returned None."
        assert len(features) > 0, "Feature extraction returned empty dictionary."

        # Check for key physics-based descriptors
        expected_keys = ["vol_per_atom", "density", "gii"]
        for key in expected_keys:
            assert key in features, f"Missing expected feature key: {key}"

        print(f"   - Extraction successful. Generated {len(features)} features.")
        print(
            f"   - Sample values: Volume/Atom={features['vol_per_atom']:.2f}, Density={features['density']:.2f}, GII={features['gii']:.4f}"
        )

    except Exception as e:
        print(f"   - Error during feature extraction: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 3. Model Training
    # ---------------------------------------------------------
    print("\n[3/5] Running Model Training Pipeline...")
    # We set load_cached_data=False to force the feature extraction code to run on the full dataset.
    # We reduce n_estimators to 50 to ensure the training completes quickly for this demo.
    try:
        models, metrics = train_model(load_cached_data=False, n_estimators=50)

        print("   - Training completed.")
        for target, score in metrics.items():
            print(f"   - Target: {target:<30} | Validation RMSLE: {score:.4f}")
            # Sanity check: RMSLE should be a finite positive number
            assert score >= 0, f"Invalid RMSLE score for {target}"

    except Exception as e:
        print(f"   - Error during model training: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 4. Prediction Generation
    # ---------------------------------------------------------
    print("\n[4/5] Generating Predictions for Test Set...")
    try:
        # Generate predictions using the trained models
        # load_cached_data=False ensures we process test files from scratch
        submission_df = generate_predictions(models, load_cached_data=False)
        print("   - Predictions generated.")

    except Exception as e:
        print(f"   - Error generating predictions: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. Submission Validation
    # ---------------------------------------------------------
    print("\n[5/5] Validating Output Submission File...")
    if not os.path.exists(SUBMISSION_PATH):
        print(f"   - Error: Submission file not found at {SUBMISSION_PATH}")
        sys.exit(1)

    try:
        loaded_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"   - Loaded submission shape: {loaded_sub.shape}")

        # Check column names
        expected_cols = ["id"] + TARGET_COLS
        assert (
            list(loaded_sub.columns) == expected_cols
        ), f"Column mismatch. Expected {expected_cols}, got {list(loaded_sub.columns)}"

        # Check row count against test metadata
        test_meta = load_metadata("test")
        assert len(loaded_sub) == len(
            test_meta
        ), f"Row count mismatch. Expected {len(test_meta)}, got {len(loaded_sub)}"

        # Check for missing values
        assert not loaded_sub.isnull().values.any(), "Submission contains NaN values."

        # Check data types
        assert pd.api.types.is_numeric_dtype(
            loaded_sub["formation_energy_ev_natom"]
        ), "Formation energy is not numeric."
        assert pd.api.types.is_numeric_dtype(
            loaded_sub["bandgap_energy_ev"]
        ), "Bandgap energy is not numeric."

        print("   - Submission file passed all validation checks.")

    except Exception as e:
        print(f"   - Error validating submission: {e}")
        sys.exit(1)

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
