import os
import sys
import numpy as np
import pandas as pd
import random
import ase
import shutil

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Import library modules
# We assume the file structure provided in the prompt exists.
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.descriptors as descriptors
import library.feature_processor as feature_processor
import library.model_handler as model_handler


def run_demonstration():
    print("=== Starting Pipeline Demonstration ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Utils
    # ------------------------------------------------------------------------
    print("\n[1] Testing Utilities...")

    # Test log transform and inverse
    original_values = np.array([0.0, 1.0, 10.0, 100.0])
    transformed = utils.log_transform(original_values)
    recovered = utils.inverse_log_transform(transformed)

    print(f"Original: {original_values}")
    print(f"Transformed: {transformed}")
    print(f"Recovered: {recovered}")

    assert np.allclose(
        original_values, recovered
    ), "Inverse log transform failed to recover original values."
    print("Utils validation passed.")

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n[2] Testing Data Loader...")

    # Load a small sample of training metadata
    # debug=True loads DEBUG_SAMPLE_SIZE rows (default 100)
    train_meta = data_loader.load_metadata("train", debug=True)
    print(f"Loaded training metadata sample shape: {train_meta.shape}")

    if len(train_meta) == 0:
        raise RuntimeError("No training metadata loaded. Cannot proceed.")

    # Pick the first example to test geometry reading
    sample_row = train_meta.iloc[0]
    sample_id = sample_row["id"]
    rel_path = sample_row["file_path"]

    print(f"Reading geometry for ID {sample_id} from {rel_path}...")
    atoms = data_loader.read_geometry(rel_path)

    assert isinstance(
        atoms, ase.Atoms
    ), "read_geometry did not return an ase.Atoms object."
    print(f"Successfully loaded atoms. Formula: {atoms.get_chemical_formula()}")

    # ------------------------------------------------------------------------
    # 3. Descriptors
    # ------------------------------------------------------------------------
    print("\n[3] Testing Descriptors...")

    # Macroscopic
    macro = descriptors.calculate_macroscopic(atoms)
    print(f"Macroscopic properties: {macro}")
    assert "volume" in macro and "density" in macro, "Missing macroscopic keys."

    # RDF
    # We reduce bins/cutoff in config if needed, but defaults are fine for single atom
    rdf = descriptors.calculate_rdf(atoms)
    print(f"RDF keys found: {list(rdf.keys())}")
    # Check if any RDF array is returned
    if rdf:
        first_key = list(rdf.keys())[0]
        assert len(rdf[first_key]) == config.RDF_BINS, "RDF bin count mismatch."

    # BVS & ECoN
    bvs_econ = descriptors.calculate_bvs_econ(atoms)
    print(f"BVS/ECoN keys: {list(bvs_econ.keys())}")
    assert len(bvs_econ["bvs"]) == len(atoms), "BVS array length mismatch."

    # Angles
    angles = descriptors.calculate_angles(atoms)
    print(f"Angle keys: {list(angles.keys())}")

    print("Descriptors validation passed.")

    # ------------------------------------------------------------------------
    # 4. Feature Processing
    # ------------------------------------------------------------------------
    print("\n[4] Testing Feature Processor...")

    # Test processing a single structure
    print("Processing single structure features...")
    single_feats = feature_processor.process_single_structure(sample_row)
    assert single_feats is not None, "process_single_structure returned None."
    assert (
        "density" in single_feats
    ), "Feature 'density' missing from processed features."

    # Test processing a dataset (debug mode)
    # We force load_cached_data=False to ensure the code actually runs
    print("Processing dataset (debug mode)...")
    # To save time, we'll just process the 'test' split in debug mode as it might be smaller or same size
    # but let's stick to train since we have targets there for the next step.
    # Note: process_dataset writes to disk.
    train_features_df = feature_processor.process_dataset(
        "train", load_cached_data=False, debug=True
    )

    print(f"Processed training features shape: {train_features_df.shape}")
    assert not train_features_df.empty, "Feature DataFrame is empty."
    assert "id" in train_features_df.columns, "ID column missing from features."

    # Also process validation set for the model training step
    print("Processing validation dataset (debug mode)...")
    val_features_df = feature_processor.process_dataset(
        "val", load_cached_data=False, debug=True
    )

    # ------------------------------------------------------------------------
    # 5. Model Training & Inference
    # ------------------------------------------------------------------------
    print("\n[5] Testing Model Handler (XGBoost)...")

    # OPTIMIZATION: Modify XGB_PARAMS in-place to make training very fast for demonstration
    print("Monkey-patching XGB_PARAMS for speed...")
    config.XGB_PARAMS["n_estimators"] = 2
    config.XGB_PARAMS["max_depth"] = 2
    config.XGB_PARAMS["early_stopping_rounds"] = 1
    # Ensure n_jobs is set
    config.XGB_PARAMS["n_jobs"] = 1

    # Train models
    # We use load_cached_data=True because we just computed and saved them in step 4
    # (process_dataset saves to parquet).
    print("Training models...")
    models = model_handler.train_xgboost(load_cached_data=True, debug=True)

    assert len(models) == len(
        config.TARGET_COLS
    ), "Did not return models for all targets."
    for target in config.TARGET_COLS:
        assert target in models, f"Model for {target} is missing."
        print(f"Model for {target} trained successfully.")

    # Predict on Test Set
    # We need to process test features first.
    # We'll use debug=True to just predict on a subset of test data for speed.
    print("Generating predictions on test set (debug mode)...")

    # We need to make sure test features are generated
    # The predict function calls process_dataset internally.
    # We force re-computation to be safe and demonstrate flow.
    model_handler.predict(models, load_cached_data=False, debug=True)

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Submission file found at {config.SUBMISSION_PATH}")
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission shape: {sub_df.shape}")
        print("Submission columns:", sub_df.columns.tolist())

        expected_cols = ["id"] + config.TARGET_COLS
        for col in expected_cols:
            assert col in sub_df.columns, f"Column {col} missing from submission."

        # Check for non-null values
        assert (
            not sub_df[config.TARGET_COLS].isnull().any().any()
        ), "Submission contains NaNs."
        print("Submission content check passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    try:
        run_demonstration()
    except Exception as e:
        print(f"\n!!! Demonstration Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
