import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import logging

# Filter warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import library components
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.feature_engineering import FeatureEngineer
from library.data_manager import DataManager
from library.models import LGBMWrapper, XGBWrapper, Ensemble
from library.pipeline import Pipeline


def create_mini_dataset(
    original_meta_path, original_track_path, dest_meta_path, dest_track_path, n_plays=2
):
    """
    Helper to create a small subset of data for demonstration purposes.
    """
    print(f"Creating mini dataset from {os.path.basename(original_meta_path)}...")

    # Load full metadata
    df_meta = pd.read_csv(original_meta_path)

    # Select specific plays
    unique_plays = df_meta["game_play"].unique()
    if len(unique_plays) > n_plays:
        selected_plays = unique_plays[:n_plays]
    else:
        selected_plays = unique_plays

    df_mini_meta = df_meta[df_meta["game_play"].isin(selected_plays)].copy()

    # Load full tracking (only columns needed to save time/memory if possible, but reading full is safer for structure)
    # To speed up reading tracking, we can't easily filter before read without iterating chunks.
    # Given the constraints, we assume we can read the tracking file (1.2M rows is manageable).
    df_track = pd.read_csv(original_track_path)
    df_mini_track = df_track[df_track["game_play"].isin(selected_plays)].copy()

    # Save
    df_mini_meta.to_csv(dest_meta_path, index=False)
    df_mini_track.to_csv(dest_track_path, index=False)

    print(
        f"  Saved {len(df_mini_meta)} meta rows and {len(df_mini_track)} tracking rows."
    )
    return selected_plays


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # --------------------------------------------------------------------------
    print("\n=== Setting up Demo Environment ===")

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to point to our demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "data_cache")
    Config.MODEL_DIR = os.path.join(DEMO_DIR, "models")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Create necessary subdirectories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Define paths for mini datasets
    mini_train_meta = os.path.join(DEMO_DIR, "mini_train_metadata.csv")
    mini_train_track = os.path.join(DEMO_DIR, "mini_train_tracking.csv")
    mini_val_meta = os.path.join(DEMO_DIR, "mini_val_metadata.csv")
    mini_val_track = os.path.join(
        DEMO_DIR, "mini_train_tracking.csv"
    )  # Reuse train tracking for val demo
    mini_test_meta = os.path.join(DEMO_DIR, "mini_test_metadata.csv")
    mini_test_track = os.path.join(DEMO_DIR, "mini_test_tracking.csv")

    # Override Config Data Paths
    Config.TRAIN_METADATA_PATH = mini_train_meta
    Config.TRAIN_TRACKING_PATH = mini_train_track
    Config.VAL_METADATA_PATH = mini_val_meta
    # Note: Val usually shares tracking file with train in this dataset structure,
    # but we point to the file we created.

    Config.TEST_METADATA_PATH = mini_test_meta
    Config.TEST_TRACKING_PATH = mini_test_track

    # Override Model Hyperparameters for Speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["verbose"] = -1

    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3
    Config.XGB_PARAMS["device"] = "cpu"  # Use CPU for tiny demo data to avoid overhead
    Config.XGB_PARAMS["tree_method"] = "hist"

    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = False  # Disable logging

    # Set Seed
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Create Mini Datasets
    # --------------------------------------------------------------------------
    print("\n=== Creating Mini Datasets ===")

    # Create Train (2 plays)
    # We need to find valid paths. The original Config points to ./metadata/train_metadata.csv
    orig_train_meta = "./metadata/train_metadata.csv"
    orig_train_track = "./input/train_player_tracking.csv"

    if not os.path.exists(orig_train_meta):
        raise FileNotFoundError(f"Original metadata not found at {orig_train_meta}")

    # Create Train
    create_mini_dataset(
        orig_train_meta, orig_train_track, mini_train_meta, mini_train_track, n_plays=2
    )

    # Create Val (1 play)
    # We'll just take the head of val metadata, assuming tracking is in train_player_tracking
    orig_val_meta = "./metadata/val_metadata.csv"
    create_mini_dataset(
        orig_val_meta, orig_train_track, mini_val_meta, mini_train_track, n_plays=1
    )

    # Create Test (1 play)
    orig_test_meta = "./metadata/test_metadata.csv"
    orig_test_track = "./input/test_player_tracking.csv"
    create_mini_dataset(
        orig_test_meta, orig_test_track, mini_test_meta, mini_test_track, n_plays=1
    )

    # --------------------------------------------------------------------------
    # 3. Component Validation: Feature Engineer
    # --------------------------------------------------------------------------
    print("\n=== Testing Feature Engineer ===")
    fe = FeatureEngineer()

    # Test process_tracking_data
    print("Processing tracking data...")
    df_track_processed = fe.process_tracking_data(
        mini_train_track, load_cached_data=False
    )

    # Assertions
    assert (
        "jerk" in df_track_processed.columns
    ), "Feature 'jerk' missing from processed tracking."
    assert any(
        col.startswith("grid_") for col in df_track_processed.columns
    ), "Grid features missing."
    print("Tracking processing successful.")

    # Test generate_dataset
    print("Generating training dataset...")
    X, y, meta = fe.generate_dataset(
        mini_train_meta, mini_train_track, mode="train", load_cached_data=False
    )

    # Assertions
    assert not X.empty, "Feature matrix X is empty."
    assert len(X) == len(y), "Mismatch between X and y length."
    assert "distance" in X.columns, "Distance feature missing."
    print(f"Dataset generation successful. Shape: {X.shape}")

    # --------------------------------------------------------------------------
    # 4. Component Validation: Data Manager
    # --------------------------------------------------------------------------
    print("\n=== Testing Data Manager ===")
    dm = DataManager()

    # Test Scout Dataset
    X_scout, y_scout = dm.get_scout_dataset(load_cached_data=False)
    assert len(X_scout) > 0, "Scout dataset is empty."
    print(f"Scout dataset loaded. Rows: {len(X_scout)}")

    # Test Mining Candidates
    X_mine, y_mine, meta_mine = dm.get_mining_candidates(
        load_cached_data=True
    )  # Use cache from prev step
    assert len(X_mine) >= len(
        X_scout
    ), "Mining candidates should be superset of scout sample."
    print("Mining candidates loaded.")

    # --------------------------------------------------------------------------
    # 5. Component Validation: Models
    # --------------------------------------------------------------------------
    print("\n=== Testing Models ===")

    # Prepare Val Data
    X_val, y_val, _ = dm.get_val_dataset(load_cached_data=False)

    # Test LightGBM
    lgbm = LGBMWrapper("test_lgbm")
    lgbm.fit(X_scout, y_scout, X_val, y_val)
    preds_lgbm = lgbm.predict(X_val)
    assert len(preds_lgbm) == len(X_val), "LGBM prediction shape mismatch."
    assert (
        0 <= preds_lgbm.min() and preds_lgbm.max() <= 1
    ), "LGBM predictions out of probability range."
    print("LightGBM training and prediction successful.")

    # Test XGBoost
    xgb_model = XGBWrapper("test_xgb")
    xgb_model.fit(X_scout, y_scout, X_val, y_val)
    preds_xgb = xgb_model.predict(X_val)
    assert len(preds_xgb) == len(X_val), "XGB prediction shape mismatch."
    print("XGBoost training and prediction successful.")

    # --------------------------------------------------------------------------
    # 6. Integration Test: Full Pipeline
    # --------------------------------------------------------------------------
    print("\n=== Testing Full Pipeline ===")
    pipeline = Pipeline()

    # Execute Pipeline
    # We force load_cached_data=True to use the features we just computed/cached in steps above
    # where possible, but the pipeline has its own logic.
    pipeline.execute(load_cached_data=True)

    # --------------------------------------------------------------------------
    # 7. Final Verification
    # --------------------------------------------------------------------------
    print("\n=== Final Verification ===")

    # Check Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file found. Rows: {len(df_sub)}")

    # Check content
    assert (
        "contact_id" in df_sub.columns and "contact" in df_sub.columns
    ), "Submission columns missing."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values."

    # Check Model Artifacts
    expected_models = [
        "scout_lgbm.joblib",
        "expert_lgbm.joblib",
        "expert_xgb.joblib",
        "best_threshold.npy",
    ]
    for m in expected_models:
        path = os.path.join(Config.MODEL_DIR, m)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact {m} not found at {path}")

    print("All verification checks passed.")
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
