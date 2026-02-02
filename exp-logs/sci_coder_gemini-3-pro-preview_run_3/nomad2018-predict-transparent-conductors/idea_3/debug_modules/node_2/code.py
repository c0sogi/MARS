import os
import sys
import numpy as np
import pandas as pd
import torch
import ase

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_column_wise_rmsle
from library.data_io import get_train_data, get_val_data, get_test_data, read_geometry
from library.descriptors import compute_descriptors, get_volume, get_density
from library.mace_embedding import MACEFeatureExtractor
from library.feature_pipeline import (
    build_feature_matrix,
    transform_targets,
    inverse_transform_targets,
)
from library.model import XGBoostRegressorWrapper, generate_submission_file


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup and Configuration
    # ----------------------------------------------------------------
    print("\n[1] Setting up environment...")
    # Initialize directories
    Config.setup()
    # Set random seed for reproducibility
    set_seed(42)

    # Adjust XGBoost parameters for a quick demo run
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["early_stopping_rounds"] = 5
    Config.VERBOSE_EVAL = False  # Suppress XGBoost logs

    print("Configuration updated for speed (n_estimators=10).")

    # 2. Data Loading & Geometry Reading
    # ----------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")
    # Load a small subset of metadata to verify functionality
    subset_size = 10
    train_meta = get_train_data(limit=subset_size)
    print(f"Loaded {len(train_meta)} training metadata rows.")

    assert len(train_meta) == subset_size, "Failed to limit training metadata."
    assert "file_path" in train_meta.columns, "Metadata missing 'file_path' column."

    # Test reading a geometry file
    sample_path = train_meta.iloc[0]["file_path"]
    atoms = read_geometry(sample_path)
    print(f"Successfully read geometry from {sample_path}")
    print(f"System formula: {atoms.get_chemical_formula()}")
    assert isinstance(
        atoms, ase.Atoms
    ), "read_geometry did not return an ase.Atoms object."

    # 3. Physical Descriptors
    # ----------------------------------------------------------------
    print("\n[3] Demonstrating Physical Descriptors...")
    # Compute descriptors for a single atom object
    vol = get_volume(atoms)
    dens = get_density(atoms)
    print(f"Single sample - Volume: {vol:.2f} A^3, Density: {dens:.2f} amu/A^3")
    assert vol > 0, "Volume should be positive."

    # Batch compute descriptors using the pipeline function (with caching)
    # We use a unique cache name to avoid conflicts with full runs
    desc_df = compute_descriptors(
        train_meta, cache_name="demo_descriptors.parquet", load_cached_data=False
    )
    print(f"Computed descriptors for {len(desc_df)} samples.")
    assert (
        "volume" in desc_df.columns and "density" in desc_df.columns
    ), "Descriptor columns missing."
    assert len(desc_df) == len(train_meta), "Descriptor count mismatch."

    # 4. MACE Embedding Extraction
    # ----------------------------------------------------------------
    print("\n[4] Demonstrating MACE Feature Extraction...")
    extractor = MACEFeatureExtractor(device=Config.DEVICE, hidden_dim=16)

    # Process a single structure
    # Note: If MACE is not installed, this returns a zero vector, which is handled.
    feats = extractor.process_structure(atoms)
    print(f"Extracted feature vector shape: {feats.shape}")

    # Expected shape is 4 * hidden_dim (mean, std, min, max)
    expected_dim = 16 * 4
    assert feats.shape == (
        expected_dim,
    ), f"Expected feature dim {expected_dim}, got {feats.shape[0]}"

    # 5. Full Feature Pipeline
    # ----------------------------------------------------------------
    print("\n[5] Running Full Feature Pipeline...")
    # We will build features for a small train and validation subset
    # This integrates metadata, descriptors, and MACE embeddings

    # Train set
    print("Building training features (limit=50)...")
    train_df = build_feature_matrix("train", load_cached_data=False, limit=50)

    # Validation set
    print("Building validation features (limit=20)...")
    val_df = build_feature_matrix("val", load_cached_data=False, limit=20)

    print(f"Train feature matrix shape: {train_df.shape}")
    print(f"Val feature matrix shape: {val_df.shape}")

    # Check for target columns
    for target in Config.TARGET_COLS:
        assert (
            target in train_df.columns
        ), f"Target {target} missing from train features."

    # 6. Data Preparation for Modeling
    # ----------------------------------------------------------------
    print("\n[6] Preparing Data for Training...")

    # Separate features and targets
    # Features are all columns except ID and targets
    feature_cols = [
        c for c in train_df.columns if c not in Config.TARGET_COLS + [Config.ID_COL]
    ]

    X_train = train_df[feature_cols]
    y_train_raw = train_df[Config.TARGET_COLS]

    X_val = val_df[feature_cols]
    y_val_raw = val_df[Config.TARGET_COLS]

    # Log-transform targets
    y_train_log = transform_targets(y_train_raw)
    y_val_log = transform_targets(y_val_raw)

    print(f"Training with {len(feature_cols)} features.")

    # 7. Model Training (XGBoost)
    # ----------------------------------------------------------------
    print("\n[7] Training XGBoost Models...")
    model_wrapper = XGBoostRegressorWrapper()

    # Fit models
    model_wrapper.fit(X_train, y_train_log, X_val, y_val_log)
    print("Training completed.")

    # 8. Evaluation
    # ----------------------------------------------------------------
    print("\n[8] Evaluating on Validation Set...")
    # Predict (returns log-scale predictions)
    val_preds_log = model_wrapper.predict(X_val)

    # Inverse transform to original scale
    val_preds = inverse_transform_targets(val_preds_log)

    # Compute RMSLE
    rmsle_score, rmsle_details = compute_column_wise_rmsle(y_val_raw, val_preds)
    print(f"Validation RMSLE: {rmsle_score:.4f}")
    print(f"Details: {rmsle_details}")

    # Sanity check: predictions should be non-negative
    assert (val_preds.values >= 0).all(), "Predictions contain negative values."

    # 9. Submission Generation
    # ----------------------------------------------------------------
    print("\n[9] Generating Submission...")
    # Load test data (small subset)
    test_limit = 10
    test_df = build_feature_matrix("test", load_cached_data=False, limit=test_limit)

    # Prepare test features
    X_test = test_df[feature_cols]
    test_ids = test_df[Config.ID_COL]

    # Generate submission file using the model wrapper
    # This function handles prediction, inverse transform, and saving to CSV
    generate_submission_file(model_wrapper, X_test, test_ids)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {sub_df.shape}")
    assert (
        len(sub_df) == test_limit
    ), f"Submission row count mismatch. Expected {test_limit}, got {len(sub_df)}"
    assert all(
        col in sub_df.columns for col in [Config.ID_COL] + Config.TARGET_COLS
    ), "Submission columns mismatch."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
