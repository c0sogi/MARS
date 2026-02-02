import os
import shutil
import pandas as pd
import numpy as np
import ase.io
import warnings

# Import library modules
import library.config
import library.descriptors
import library.data_manager
import library.preprocessing
import library.model_wrapper

# Set random seeds
np.random.seed(42)


def run_demonstration():
    print("Starting Library Demonstration...")

    # --- 1. Setup Temporary Workspace and Sample Data ---
    # We create a small subset of the metadata to make the pipeline run fast (e.g., 5 samples)
    working_dir = "./working/demo_execution"
    os.makedirs(working_dir, exist_ok=True)

    # Define paths for sample metadata
    sample_train_path = os.path.join(working_dir, "train_metadata_sample.csv")
    sample_val_path = os.path.join(working_dir, "val_metadata_sample.csv")
    sample_test_path = os.path.join(working_dir, "test_metadata_sample.csv")
    sample_submission_path = os.path.join(working_dir, "demo_submission.csv")

    # Load original metadata
    orig_train = pd.read_csv(library.config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(library.config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(library.config.TEST_METADATA_PATH)

    # Sample 5 rows for speed
    train_sample = orig_train.head(5).copy()
    val_sample = orig_val.head(5).copy()
    test_sample = orig_test.head(5).copy()

    # Save samples
    train_sample.to_csv(sample_train_path, index=False)
    val_sample.to_csv(sample_val_path, index=False)
    test_sample.to_csv(sample_test_path, index=False)

    print(f"Created sample metadata with {len(train_sample)} rows for training.")

    # --- 2. Monkey-patch Library Paths ---
    # We need to point the library modules to our sample data and a temporary cache directory
    # so we don't overwrite or use the full dataset cache.

    # Patch data_manager paths
    library.data_manager.TRAIN_METADATA_PATH = sample_train_path
    library.data_manager.VAL_METADATA_PATH = sample_val_path
    library.data_manager.TEST_METADATA_PATH = sample_test_path
    library.data_manager.CACHE_DIR = working_dir  # Use working dir as cache

    # Patch preprocessing paths
    library.preprocessing.CACHE_DIR = working_dir

    # Patch model_wrapper paths
    library.model_wrapper.SUBMISSION_PATH = sample_submission_path

    # --- 3. Demonstrate StructureDescriptor ---
    print("\n--- Demonstrating StructureDescriptor ---")
    descriptor = library.descriptors.StructureDescriptor()

    # Load one geometry file from the sample
    first_geom_path = os.path.join(
        library.config.INPUT_DIR, train_sample.iloc[0]["file_path"]
    )
    atoms = ase.io.read(first_geom_path, format="aims")

    # Extract features
    features = descriptor.extract(atoms)

    # Verification
    print(f"Extracted {len(features)} features for a single structure.")
    expected_keys = ["macro_density", "comp_frac_Al", "bond_dist_Ga-O_p50"]
    for key in expected_keys:
        if key in features:
            print(f"Feature '{key}': {features[key]:.4f}")
        else:
            # Some keys might be missing if specific bonds don't exist, which is valid,
            # but macro_density should always be there.
            if key == "macro_density":
                raise AssertionError(f"Expected feature {key} missing.")

    # Assert density is positive
    assert features["macro_density"] > 0, "Density should be positive"
    print("StructureDescriptor logic verified.")

    # --- 4. Demonstrate DataManager ---
    print("\n--- Demonstrating MaterialDataset (Feature Extraction) ---")
    dm = library.data_manager.MaterialDataset()

    # Construct feature matrix for the training sample
    # load_cached_data=False forces computation
    print("Extracting features for training sample (forced recompute)...")
    df_features = dm.construct_feature_matrix("train", load_cached_data=False)

    # Verification
    assert len(df_features) == 5, f"Expected 5 rows, got {len(df_features)}"
    assert (
        "macro_volume_per_atom" in df_features.columns
    ), "Features missing macroscopic data"
    print("Feature matrix construction successful.")

    # --- 5. Demonstrate DataPreprocessor ---
    print("\n--- Demonstrating DataPreprocessor ---")
    preprocessor = library.preprocessing.DataPreprocessor()

    # Create a dummy dataframe with a constant column to test cleaning
    dummy_df = df_features.copy()
    dummy_df["constant_col"] = 1.0

    # Fit and transform
    cleaned_df = preprocessor.fit_transform(dummy_df)

    # Verification
    assert "constant_col" not in cleaned_df.columns, "Constant column was not dropped"
    assert preprocessor.fitted is True, "Preprocessor should be fitted"

    # Test log transform logic
    vals = np.array([0.0, 1.0, 10.0])
    log_vals = preprocessor.log_transform(vals)
    inv_vals = preprocessor.inverse_log_transform(log_vals)
    assert np.allclose(vals, inv_vals), "Log transform inverse check failed"

    print(
        "DataPreprocessor logic verified (constant column dropped, log transform reversible)."
    )

    # --- 6. Demonstrate Model Wrapper (Training & Prediction) ---
    print("\n--- Demonstrating DualXGBoostModel ---")
    model = library.model_wrapper.DualXGBoostModel()

    # Train the model
    # We use n_estimators=2 to make it extremely fast
    # load_cached_data=True will pick up the parquet files we generated/cached in step 4 implicitly
    # (though get_preprocessed_dataset handles its own caching logic, we pointed CACHE_DIR to working_dir)
    # Note: get_preprocessed_dataset will look for 'train_cleaned.parquet'.
    # Since we ran construct_feature_matrix but not get_preprocessed_dataset yet, it might compute again or we rely on it.
    # Let's let it run naturally.

    print("Training model with reduced estimators...")
    model.train(load_cached_data=False, n_estimators=2)

    # Predict on test set
    print("Generating predictions on test sample...")
    # This will trigger feature extraction for the test sample (5 rows)
    model.predict(load_cached_data=False)

    # Verification
    if os.path.exists(sample_submission_path):
        sub_df = pd.read_csv(sample_submission_path)
        print(f"Submission generated at {sample_submission_path}")
        print(sub_df.head())
        assert len(sub_df) == 5, "Submission should have 5 rows"
        assert "formation_energy_ev_natom" in sub_df.columns
        assert "bandgap_energy_ev" in sub_df.columns
        # Check values are not NaN
        assert not sub_df.isnull().values.any(), "Submission contains NaNs"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
