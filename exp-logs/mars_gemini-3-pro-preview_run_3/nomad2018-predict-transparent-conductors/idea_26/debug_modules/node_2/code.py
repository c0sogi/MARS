import os
import shutil
import numpy as np
import pandas as pd
import ase
from sklearn.metrics import mean_squared_error

# Import library modules
from library.config import Config
import library.data_loader as dl
import library.features as ft
import library.preprocessing as pp
import library.model as md
import library.pipeline as pl


def main():
    print("=== Starting Demonstration of DMGF Library ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("--- 1. Configuration Setup ---")

    # Override Config for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Enable Debug mode to process only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample for speed

    # Reduce model complexity for speed
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print("Configuration updated for demonstration.\n")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("--- 2. Data Loading ---")

    # Load metadata
    try:
        df_train_meta = dl.load_metadata("train")
        print(f"Loaded Train Metadata: {df_train_meta.shape}")

        # Verify columns
        expected_cols = [
            "id",
            "file_path",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
        ]
        for col in expected_cols:
            assert (
                col in df_train_meta.columns
            ), f"Missing column {col} in train metadata"

        # Test reading a geometry file
        sample_path = df_train_meta.iloc[0]["file_path"]
        atoms = dl.read_geometry(sample_path)
        print(
            f"Successfully read geometry for ID {df_train_meta.iloc[0]['id']}: {atoms}"
        )
        assert isinstance(
            atoms, ase.Atoms
        ), "read_geometry did not return an ase.Atoms object"

    except Exception as e:
        print(f"Data loading failed: {e}")
        raise e
    print("Data loading verification passed.\n")

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("--- 3. Feature Extraction ---")

    # Instantiate Featurizer
    featurizer = ft.GeometricFeaturizer()

    # Test single atom featurization
    print("Featurizing single structure...")
    features_vec = featurizer.featurize(atoms)
    print(f"Feature vector shape: {features_vec.shape}")
    assert len(features_vec) > 0, "Feature vector is empty"
    assert not np.any(np.isnan(features_vec)), "Feature vector contains NaNs"

    # Test batch processing via process_data
    # This will use the Config.DEBUG_SAMPLE_SIZE limit
    print("Running batch feature processing (train)...")
    # Force recompute to ensure logic runs
    if os.path.exists(Config.TRAIN_FEATURES_PATH):
        os.remove(Config.TRAIN_FEATURES_PATH)

    df_train_feats = ft.process_data("train", load_cached_data=False)
    print(f"Processed Train Features DataFrame: {df_train_feats.shape}")

    # Check if targets are merged
    assert "formation_energy_ev_natom" in df_train_feats.columns
    assert "bandgap_energy_ev" in df_train_feats.columns

    # Process validation data as well needed for model training
    print("Running batch feature processing (val)...")
    if os.path.exists(Config.VAL_FEATURES_PATH):
        os.remove(Config.VAL_FEATURES_PATH)
    df_val_feats = ft.process_data("val", load_cached_data=False)
    print(f"Processed Val Features DataFrame: {df_val_feats.shape}")
    print("Feature extraction verification passed.\n")

    # -------------------------------------------------------------------------
    # 4. Preprocessing
    # -------------------------------------------------------------------------
    print("--- 4. Preprocessing ---")

    # Test TargetTransformer
    transformer = pp.TargetTransformer()
    dummy_targets = np.array([0.0, 1.0, 10.0])
    transformed = transformer.transform(dummy_targets)
    inverse = transformer.inverse_transform(transformed)
    print(f"Original: {dummy_targets}")
    print(f"Transformed (log1p): {transformed}")
    print(f"Inversed: {inverse}")
    assert np.allclose(dummy_targets, inverse), "TargetTransformer inverse failed"

    # Test FeatureCleaner
    # Create dummy data with a constant column and a NaN
    X_dummy = pd.DataFrame(
        {
            "feat_1": [1.0, 2.0, 3.0, np.nan],
            "feat_2": [5.0, 5.0, 5.0, 5.0],  # Constant
            "feat_3": [0.1, 0.2, 0.3, 0.4],
        }
    )
    print("Dummy Feature Matrix (before cleaning):")
    print(X_dummy)

    cleaner = pp.FeatureCleaner(constant_threshold=0.0)
    X_clean = cleaner.fit_transform(X_dummy)

    print("Dummy Feature Matrix (after cleaning):")
    print(X_clean)

    # feat_2 should be dropped, NaN in feat_1 filled (mean of 1,2,3 is 2)
    assert "feat_2" not in X_clean.columns, "Constant feature was not removed"
    assert not X_clean.isnull().any().any(), "NaNs still present after cleaning"
    print("Preprocessing verification passed.\n")

    # -------------------------------------------------------------------------
    # 5. Model Training & Prediction
    # -------------------------------------------------------------------------
    print("--- 5. Model Training & Prediction ---")

    # Train model using the small processed datasets
    artifacts = md.train_model(df_train_feats, df_val_feats)

    models = artifacts["models"]
    print(f"Trained models for targets: {list(models.keys())}")

    # Verify both targets have models
    for target in Config.TARGET_COLS:
        assert target in models, f"Model for {target} not trained"

    # Generate dummy test features for prediction test
    # We use the same structure as train features but without targets
    df_test_dummy = df_train_feats.drop(columns=Config.TARGET_COLS).copy()

    print("Generating predictions on dummy test set...")
    preds_df = md.predict_model(artifacts, df_test_dummy)
    print(f"Predictions shape: {preds_df.shape}")
    print(preds_df.head())

    assert "id" in preds_df.columns
    for target in Config.TARGET_COLS:
        assert target in preds_df.columns
        # Energies should be non-negative
        assert (preds_df[target] >= 0).all(), f"Negative predictions found for {target}"

    print("Model training and prediction verification passed.\n")

    # -------------------------------------------------------------------------
    # 6. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("--- 6. Full Pipeline Execution ---")

    # We will run the inference pipeline.
    # The training pipeline was essentially covered in step 5, but let's run the inference function
    # to ensure end-to-end integration works.

    # First, ensure we have test features processed
    if os.path.exists(Config.TEST_FEATURES_PATH):
        os.remove(Config.TEST_FEATURES_PATH)

    # Run inference pipeline (this will process test data, predict, and save)
    # Note: We reuse the 'artifacts' (trained models) from step 5 to save time
    pl.run_inference_pipeline(artifacts, load_cached_data=False)

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Final Submission Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns.tolist()}")

    expected_sub_cols = ["id"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_sub_cols, "Submission columns mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
