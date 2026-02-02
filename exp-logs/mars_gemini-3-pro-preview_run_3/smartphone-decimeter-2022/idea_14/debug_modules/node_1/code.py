import os
import shutil
import pandas as pd
import numpy as np
import sys
import lightgbm as lgb

# Ensure reproducibility
np.random.seed(42)

# Define temporary directories
WORKING_DIR = "./working"
TEMP_META_DIR = os.path.join(WORKING_DIR, "demo_metadata")
TEMP_CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")
TEMP_SUBMISSION_PATH = os.path.join(WORKING_DIR, "demo_submission.csv")

os.makedirs(TEMP_META_DIR, exist_ok=True)
os.makedirs(TEMP_CACHE_DIR, exist_ok=True)

# Import library modules
# We import them now so we can patch their variables
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib
import library.inference as inference_lib


def test_utils():
    print("\n[Test] Verifying Utility Functions...")
    # Test Lat/Lon/Alt -> ECEF -> Lat/Lon/Alt round trip
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = utils.geodetic_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = utils.ecef_to_geodetic(x, y, z)

    assert np.isclose(lat, lat_out, atol=1e-5), f"Lat mismatch: {lat} vs {lat_out}"
    assert np.isclose(lon, lon_out, atol=1e-5), f"Lon mismatch: {lon} vs {lon_out}"
    assert np.isclose(alt, alt_out, atol=1e-3), f"Alt mismatch: {alt} vs {alt_out}"

    # Test Haversine
    # Distance between two close points
    d = utils.haversine_distance(0, 0, 0, 1)  # 1 degree longitude at equator ~ 111km
    assert 111000 < d < 112000, f"Haversine calculation unexpected: {d}"
    print("[Test] Utils verification passed.")


def setup_demo_metadata():
    print("\n[Setup] Creating subsampled metadata for demo...")

    # Load original metadata
    orig_train_meta_path = os.path.join(config.METADATA_DIR, "train_metadata.csv")
    orig_val_meta_path = os.path.join(config.METADATA_DIR, "val_metadata.csv")
    orig_test_meta_path = os.path.join(config.METADATA_DIR, "test_metadata.csv")

    if not os.path.exists(orig_train_meta_path):
        raise FileNotFoundError("Original train metadata not found.")

    # Read and sample
    train_df = pd.read_csv(orig_train_meta_path)
    val_df = pd.read_csv(orig_val_meta_path)
    test_df = pd.read_csv(orig_test_meta_path)

    # Pick 1 drive for train, 1 for val, 1 for test
    train_drives = train_df["drive_id"].unique()
    val_drives = val_df["drive_id"].unique()
    test_drives = test_df["drive_id"].unique()

    # Select small subset (first available)
    demo_train = train_df[train_df["drive_id"] == train_drives[0]].head(50)  # 50 epochs
    demo_val = val_df[val_df["drive_id"] == val_drives[0]].head(20)
    demo_test = test_df[test_df["drive_id"] == test_drives[0]].head(20)

    # Save to temp dir
    demo_train.to_csv(os.path.join(TEMP_META_DIR, "train_metadata.csv"), index=False)
    demo_val.to_csv(os.path.join(TEMP_META_DIR, "val_metadata.csv"), index=False)
    demo_test.to_csv(os.path.join(TEMP_META_DIR, "test_metadata.csv"), index=False)

    print(f"[Setup] Demo metadata created at {TEMP_META_DIR}")
    print(f"  Train: {len(demo_train)} rows")
    print(f"  Val:   {len(demo_val)} rows")
    print(f"  Test:  {len(demo_test)} rows")


def patch_libraries():
    print("\n[Setup] Monkeypatching library paths...")

    # Patch data_loader
    data_loader.METADATA_DIR = TEMP_META_DIR
    data_loader.CACHE_DIR = TEMP_CACHE_DIR

    # Patch train
    train_lib.CACHE_DIR = TEMP_CACHE_DIR
    train_lib.SUBMISSION_PATH = TEMP_SUBMISSION_PATH

    # Patch inference
    inference_lib.CACHE_DIR = TEMP_CACHE_DIR
    inference_lib.SUBMISSION_PATH = TEMP_SUBMISSION_PATH

    # Patch config (though mostly used via imports in other files, good practice)
    config.METADATA_DIR = TEMP_META_DIR
    config.CACHE_DIR = TEMP_CACHE_DIR
    config.SUBMISSION_PATH = TEMP_SUBMISSION_PATH

    print("[Setup] Patching complete.")


def test_data_loader_and_model():
    print("\n[Test] Testing DataLoader and Model Training...")

    # 1. Load Data
    # This triggers _process_trip which does the heavy lifting (feature engineering)
    train_df = data_loader.get_train_data(load_cached_data=False)
    val_df = data_loader.get_val_data(load_cached_data=False)

    assert not train_df.empty, "Train DataFrame is empty"
    assert not val_df.empty, "Val DataFrame is empty"

    # Check for expected features
    expected_cols = config.FEATURES + ["target_E", "target_N"]
    for col in expected_cols:
        assert col in train_df.columns, f"Missing column {col} in train data"

    print(
        f"[Test] Data loaded. Train shape: {train_df.shape}, Val shape: {val_df.shape}"
    )

    # 2. Train Model
    print("[Test] Training LightGBM model on demo data...")
    lgbm_model = model_lib.LGBMModel()

    # Reduce boosting rounds for speed in this test
    lgbm_model.train(
        train_df, val_df, config.FEATURES, num_boost_round=10, early_stopping_rounds=5
    )

    assert lgbm_model.model_e is not None, "East model not trained"
    assert lgbm_model.model_n is not None, "North model not trained"

    # 3. Predict
    pred_e, pred_n = lgbm_model.predict(val_df, config.FEATURES)
    assert len(pred_e) == len(val_df)
    assert len(pred_n) == len(val_df)

    mae = np.mean(np.abs(val_df["target_E"] - pred_e))
    print(f"[Test] Validation MAE (East): {mae:.4f}")
    print("[Test] Model training and prediction passed.")


def test_pipeline():
    print("\n[Test] Running Full Cross-Validation and Inference Pipeline...")

    # Run CV (Training)
    # We use n_folds=2 for speed on this tiny dataset
    train_lib.run_cross_validation(load_cached_data=True, n_folds=2)

    # Check if models were saved
    model_path = os.path.join(TEMP_CACHE_DIR, "models", "lgbm_east_fold_0.txt")
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    # Run Inference (Submission Generation)
    inference_lib.generate_submission(load_cached_data=False, n_folds=2)

    # Check submission
    assert os.path.exists(TEMP_SUBMISSION_PATH), "Submission file not generated"
    sub_df = pd.read_csv(TEMP_SUBMISSION_PATH)
    assert not sub_df.empty, "Submission file is empty"
    assert "LatitudeDegrees" in sub_df.columns
    assert "LongitudeDegrees" in sub_df.columns

    print(f"[Test] Pipeline completed. Submission generated at {TEMP_SUBMISSION_PATH}")
    print(sub_df.head())


if __name__ == "__main__":
    try:
        # 1. Verify Utils
        test_utils()

        # 2. Prepare Environment
        setup_demo_metadata()
        patch_libraries()

        # 3. Test Components
        test_data_loader_and_model()

        # 4. Test Full Pipeline
        test_pipeline()

        print("\nSUCCESS: All demonstrations and tests passed.")

    except Exception as e:
        print(f"\nFAILURE: An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
