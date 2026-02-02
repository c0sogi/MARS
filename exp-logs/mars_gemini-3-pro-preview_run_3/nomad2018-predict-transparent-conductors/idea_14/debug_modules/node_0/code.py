import os
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
from ase import Atoms
from library.config import Config
from library import features, data, model

# Set fixed random seeds for reproducibility
np.random.seed(42)


def main():
    print("=== Starting Library Usage Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    # We redirect the working directory to a demo folder to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution/"

    # Clean up demo directory if it exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Modify hyperparameters for speed
    Config.XGB_PARAMS["n_estimators"] = 2
    Config.XGB_PARAMS["max_depth"] = 2
    Config.DEBUG_SAMPLE_SIZE = 10  # Process only 10 samples

    print(
        f"Configuration updated: Working Dir={Config.WORKING_DIR}, Debug Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # -------------------------------------------------------------------------
    # 2. Test Feature Extraction Logic
    # -------------------------------------------------------------------------
    print("\n--- Testing Feature Extraction Functions ---")

    # Create a dummy ASE atoms object (Aluminum and Gallium in a box)
    dummy_atoms = Atoms(
        "AlGa", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[10, 10, 10], pbc=True
    )

    # A. Physical Descriptors
    phys_feats = features.compute_physical_descriptors(dummy_atoms)
    print(f"Physical Features (Vol, Density): {phys_feats}")
    assert len(phys_feats) == 2, "Physical features should have length 2"
    assert phys_feats[0] > 0, "Volume should be positive"

    # B. Radial Distribution Function (RDF)
    # Using default elements [Al, Ga, In, O] -> 4 elements -> 10 pairs
    # 10 pairs * 60 bins = 600 features
    rdf_feats = features.compute_rdf(
        dummy_atoms,
        cutoff=Config.RDF_CUTOFF,
        n_bins=Config.RDF_NUM_BINS,
        elements=Config.ELEMENTS,
    )
    expected_rdf_len = 10 * Config.RDF_NUM_BINS
    print(f"RDF Feature Vector Length: {len(rdf_feats)} (Expected: {expected_rdf_len})")
    assert (
        len(rdf_feats) == expected_rdf_len
    ), f"RDF length mismatch. Got {len(rdf_feats)}, expected {expected_rdf_len}"

    # C. Local Environment Moments (LEM)
    # 4 elements * 4 stats (mean_cn, std_cn, mean_ang, std_ang) = 16 features
    lem_feats = features.compute_local_moments(
        dummy_atoms, cutoff=Config.NEIGHBOR_CUTOFF, elements=Config.ELEMENTS
    )
    expected_lem_len = 4 * 4
    print(f"LEM Feature Vector Length: {len(lem_feats)} (Expected: {expected_lem_len})")
    assert (
        len(lem_feats) == expected_lem_len
    ), f"LEM length mismatch. Got {len(lem_feats)}, expected {expected_lem_len}"

    # -------------------------------------------------------------------------
    # 3. Test Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Pipeline Integration ---")

    # Load Training Data (Debug Mode)
    # This triggers process_dataset -> loads geometry -> computes features -> saves parquet
    print("Building Training Dataset (Debug Mode)...")
    train_df = data.build_dataset("train", load_cached_data=False, debug=True)

    print(f"Train DataFrame Shape: {train_df.shape}")
    assert not train_df.empty, "Train DataFrame is empty"
    assert (
        "target_formation" in train_df.columns
    ), "Target column 'target_formation' missing"
    assert (
        "target_bandgap" in train_df.columns
    ), "Target column 'target_bandgap' missing"

    # Check if cache file was created
    cache_path = os.path.join(Config.WORKING_DIR, Config.TRAIN_FEATURES_FILE)
    assert os.path.exists(cache_path), f"Cache file not created at {cache_path}"

    # Load Validation Data
    print("Building Validation Dataset (Debug Mode)...")
    val_df = data.build_dataset("val", load_cached_data=False, debug=True)
    assert not val_df.empty, "Validation DataFrame is empty"

    # Load Test Data
    print("Building Test Dataset (Debug Mode)...")
    test_df = data.build_dataset("test", load_cached_data=False, debug=True)
    assert not test_df.empty, "Test DataFrame is empty"
    assert "target_formation" not in test_df.columns, "Test set should not have targets"

    # -------------------------------------------------------------------------
    # 4. Test Model Training and Prediction
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Training & Inference ---")

    # Split Features and Targets
    X_train, y_train = data.get_feature_target_split(train_df)
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Training Targets Shape: {y_train.shape}")

    # Instantiate Model
    regressor = model.DualXGBoostRegressor(params=Config.XGB_PARAMS)

    # Fit Model
    print("Fitting model...")
    regressor.fit(train_df, val_df, early_stopping_rounds=2)

    # Predict
    print("Predicting on test set...")
    submission = regressor.predict(test_df)

    print("Prediction Head:")
    print(submission.head())

    # Verify Submission Format
    assert "id" in submission.columns
    assert "formation_energy_ev_natom" in submission.columns
    assert "bandgap_energy_ev" in submission.columns
    assert len(submission) == len(test_df)

    # -------------------------------------------------------------------------
    # 5. Test Full Pipeline Wrapper
    # -------------------------------------------------------------------------
    print("\n--- Testing Full Pipeline Execution ---")
    # This runs the high-level function that orchestrates everything
    # We use load_cached_data=True to verify it picks up the files we just created
    model.run_pipeline(load_cached_data=True, debug=True)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
