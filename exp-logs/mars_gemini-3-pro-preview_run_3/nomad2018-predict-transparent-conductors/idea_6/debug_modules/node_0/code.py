import os
import numpy as np
import pandas as pd
import shutil
import library.config as config
from library.utils import log_transform, inverse_log_transform, calculate_rmsle
from library.data_loader import get_dataset, load_metadata, load_geometry
from library.feature_generator import (
    PhysicalDescriptorExtractor,
    ChemicallyResolvedEmbedder,
    generate_features,
)
from library.model_trainer import GradientBoostingPredictor


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup & Configuration Overrides for Speed
    # Override XGBoost params to make training instant for this demo
    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 3

    # Define a small limit for data processing
    DEMO_LIMIT = 20

    # 2. Verify Utility Functions
    print("\n[1/5] Verifying Utility Functions...")
    y_true = np.array([0.0, 1.0, 10.0])
    y_log = log_transform(y_true)
    y_inv = inverse_log_transform(y_log)

    assert np.allclose(y_true, y_inv), "Log transform -> Inverse transform failed"

    rmsle = calculate_rmsle(np.array([[1.0], [2.0]]), np.array([[1.1], [1.9]]))
    assert isinstance(rmsle, float) and rmsle >= 0, "RMSLE calculation failed"
    print("Utils verified.")

    # 3. Verify Data Loader
    print("\n[2/5] Verifying Data Loader...")
    # Test get_dataset generator
    count = 0
    for row, atoms in get_dataset("train", limit=5):
        count += 1
        assert "id" in row, "Metadata row missing 'id'"
        assert atoms is not None, "Failed to load Atoms object"
        assert len(atoms) > 0, "Atoms object is empty"
    assert count == 5, f"get_dataset yielded {count} items, expected 5"
    print("Data Loader verified.")

    # 4. Verify Feature Generator
    print("\n[3/5] Verifying Feature Generator...")

    # Test individual extractors on a single geometry
    # Load one sample manually
    meta_df = load_metadata("train").head(1)
    sample_path = meta_df.iloc[0]["file_path"]
    sample_atoms = load_geometry(sample_path)

    # Physical Descriptors
    phys_extractor = PhysicalDescriptorExtractor()
    phys_feats = phys_extractor.extract(sample_atoms)
    assert "volume" in phys_feats, "Physical features missing 'volume'"
    assert "density" in phys_feats, "Physical features missing 'density'"

    # Embeddings
    emb_extractor = ChemicallyResolvedEmbedder()
    emb_feats = emb_extractor.extract(sample_atoms)
    # Check if we got a dictionary (might be NaNs if matgl issues, but keys should exist or dict should be valid)
    assert isinstance(emb_feats, dict), "Embedding extraction did not return a dict"

    # Generate features for a subset (Train)
    # We force reload to demonstrate generation logic
    print(f"Generating train features for {DEMO_LIMIT} samples...")
    # Note: We patch the cache path to avoid overwriting the main working file with partial data
    original_train_path = config.TRAIN_COMBINED_FEATURES_PATH
    demo_train_path = os.path.join(config.WORKING_DIR, "demo_train_features.parquet")
    config.TRAIN_COMBINED_FEATURES_PATH = demo_train_path

    df_train_feats = generate_features(
        "train", load_cached_data=False, limit=DEMO_LIMIT
    )
    assert len(df_train_feats) == DEMO_LIMIT, "Feature generation count mismatch"
    assert (
        "formation_energy_ev_natom" in df_train_feats.columns
    ), "Target column missing in train features"

    # Generate features for a subset (Test)
    print(f"Generating test features for {DEMO_LIMIT} samples...")
    original_test_path = config.TEST_COMBINED_FEATURES_PATH
    demo_test_path = os.path.join(config.WORKING_DIR, "demo_test_features.parquet")
    config.TEST_COMBINED_FEATURES_PATH = demo_test_path

    df_test_feats = generate_features("test", load_cached_data=False, limit=DEMO_LIMIT)
    assert len(df_test_feats) == DEMO_LIMIT, "Test feature generation count mismatch"

    # Generate features for a subset (Val) - needed for training
    print(f"Generating val features for {DEMO_LIMIT} samples...")
    original_val_path = config.VAL_COMBINED_FEATURES_PATH
    demo_val_path = os.path.join(config.WORKING_DIR, "demo_val_features.parquet")
    config.VAL_COMBINED_FEATURES_PATH = demo_val_path

    df_val_feats = generate_features("val", load_cached_data=False, limit=DEMO_LIMIT)

    print("Feature Generator verified.")

    # 5. Verify Model Trainer
    print("\n[4/5] Verifying Model Trainer...")
    predictor = GradientBoostingPredictor()

    # Train the model
    # Using the small feature dataframes generated above
    predictor.fit_model(df_train_feats, df_val_feats)

    # Predict on test set
    preds = predictor.predict_values(df_test_feats)
    assert len(preds) == len(df_test_feats), "Prediction count mismatch"
    assert all(
        col in preds.columns for col in config.TARGET_COLS
    ), "Missing target columns in predictions"
    assert (preds.values >= 0).all(), "Predictions contain negative values"

    print("Model Trainer verified.")

    # 6. Verify Submission Generation
    print("\n[5/5] Verifying Submission Generation...")
    # We need to ensure the test_df passed to generate_submission has 'id'
    # The generate_submission function writes to disk.
    predictor.generate_submission(df_test_feats)

    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission generated with shape: {sub_df.shape}")
        # In this demo, we only predicted for DEMO_LIMIT samples.
        # The generate_submission function loads sample_submission.csv and updates it.
        # So the output shape should match sample_submission.csv (240 rows),
        # but only the first DEMO_LIMIT rows would be updated by our model.
        expected_rows = pd.read_csv(config.SAMPLE_SUBMISSION_CSV).shape[0]
        assert (
            len(sub_df) == expected_rows
        ), f"Submission row count {len(sub_df)} != {expected_rows}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("Submission Generation verified.")

    # Cleanup demo files
    if os.path.exists(demo_train_path):
        os.remove(demo_train_path)
    if os.path.exists(demo_test_path):
        os.remove(demo_test_path)
    if os.path.exists(demo_val_path):
        os.remove(demo_val_path)

    # Restore config paths (good practice, though script ends here)
    config.TRAIN_COMBINED_FEATURES_PATH = original_train_path
    config.TEST_COMBINED_FEATURES_PATH = original_test_path
    config.VAL_COMBINED_FEATURES_PATH = original_val_path

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
