import os
import sys
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from provided library
from library.config import Config
from library.data_manager import DataManager
from library.training_core import TrainingCore
from library.evaluation import Evaluator


def create_mini_dataset(
    metadata_path, tracking_path, out_meta_path, out_track_path, n_plays=2
):
    """
    Creates a mini dataset by sampling a few plays to ensure speed.
    """
    print(f"Creating mini dataset from {os.path.basename(metadata_path)}...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)
    unique_plays = df_meta["game_play"].unique()

    if len(unique_plays) > n_plays:
        selected_plays = unique_plays[:n_plays]
        df_mini_meta = df_meta[df_meta["game_play"].isin(selected_plays)].copy()
    else:
        df_mini_meta = df_meta.copy()
        selected_plays = unique_plays

    # Save mini metadata
    df_mini_meta.to_csv(out_meta_path, index=False)

    # Load tracking
    # Tracking files can be large, so we load, filter, and save.
    # We assume the tracking file fits in memory for this environment (120GB+ RAM available).
    print(f"Filtering tracking data for {len(selected_plays)} plays...")
    df_track = pd.read_csv(tracking_path)

    # Filter tracking for the selected plays
    # Ensure game_play is string for matching
    df_track["game_play"] = df_track["game_play"].astype(str)
    selected_plays_str = [str(p) for p in selected_plays]

    df_mini_track = df_track[df_track["game_play"].isin(selected_plays_str)].copy()

    # Save mini tracking
    df_mini_track.to_csv(out_track_path, index=False)

    return len(df_mini_meta), len(df_mini_track)


def run_demo():
    # =========================================================================
    # 1. Configuration Override for Speed and Isolation
    # =========================================================================
    print("Setting up configuration for demo execution...")

    # Define a specific working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_DIR = os.path.join(DEMO_DIR, "models")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce Model Complexity for Speed
    Config.NUM_BOOST_ROUND = 10
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = -1

    # LGBM Params override
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 31

    # XGB Params override
    Config.XGB_PARAMS["n_estimators"] = 10

    # HistGB Params override
    Config.HISTGB_PARAMS["max_iter"] = 10

    # Ensure directories exist
    Config.setup()

    # =========================================================================
    # 2. Create Mini Datasets
    # =========================================================================
    print("\n--- Generating Mini Datasets ---")

    # Paths for mini datasets
    mini_train_meta = os.path.join(DEMO_DIR, "mini_train_metadata.csv")
    mini_train_track = os.path.join(DEMO_DIR, "mini_train_tracking.csv")

    mini_val_meta = os.path.join(DEMO_DIR, "mini_val_metadata.csv")
    # Val uses train tracking file usually, but we'll create a specific filtered one
    # Actually, Config uses TRAIN_TRACKING_PATH for both train and val features.
    # So we just need to update TRAIN_TRACKING_PATH to a file containing both train and val plays.

    mini_test_meta = os.path.join(DEMO_DIR, "mini_test_metadata.csv")
    mini_test_track = os.path.join(DEMO_DIR, "mini_test_tracking.csv")

    # Sample Train Metadata (2 plays)
    # We need to ensure the tracking file covers these plays.
    # We will sample plays from train_metadata, save to mini_train_meta.
    # We will sample plays from val_metadata, save to mini_val_meta.
    # Then we will create a combined mini_tracking file covering both sets of plays.

    # Load full train and val metadata to pick plays
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    train_plays = df_train_full["game_play"].unique()[:2]
    df_mini_train = df_train_full[df_train_full["game_play"].isin(train_plays)].copy()
    df_mini_train.to_csv(mini_train_meta, index=False)

    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    val_plays = df_val_full["game_play"].unique()[:1]
    df_mini_val = df_val_full[df_val_full["game_play"].isin(val_plays)].copy()
    df_mini_val.to_csv(mini_val_meta, index=False)

    # Create combined tracking for train/val
    all_needed_plays = np.concatenate([train_plays, val_plays])
    print(f"Filtering train tracking for {len(all_needed_plays)} plays...")
    df_track_full = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    # Ensure string comparison
    df_track_full["game_play"] = df_track_full["game_play"].astype(str)
    all_needed_plays_str = [str(x) for x in all_needed_plays]
    df_mini_track_tv = df_track_full[
        df_track_full["game_play"].isin(all_needed_plays_str)
    ].copy()
    df_mini_track_tv.to_csv(mini_train_track, index=False)

    # Sample Test Metadata (1 play)
    create_mini_dataset(
        Config.TEST_METADATA_PATH,
        Config.TEST_TRACKING_PATH,
        mini_test_meta,
        mini_test_track,
        n_plays=1,
    )

    # Update Config Paths to point to mini datasets
    Config.TRAIN_METADATA_PATH = mini_train_meta
    Config.VAL_METADATA_PATH = mini_val_meta
    Config.TRAIN_TRACKING_PATH = (
        mini_train_track  # Used for both train and val features
    )
    Config.TEST_METADATA_PATH = mini_test_meta
    Config.TEST_TRACKING_PATH = mini_test_track

    print("Mini datasets created and Config updated.")

    # =========================================================================
    # 3. Feature Engineering Verification
    # =========================================================================
    print("\n--- Verifying Feature Engineering ---")
    dm = DataManager()

    # Generate Train Features
    df_train_feats = dm.get_train_features(load_cached_data=False)
    print(f"Train Features Shape: {df_train_feats.shape}")

    # Assertions
    assert not df_train_feats.empty, "Train features DataFrame is empty."
    assert "distance" in df_train_feats.columns, "Feature 'distance' missing."
    assert (
        "radial_velocity" in df_train_feats.columns
    ), "Feature 'radial_velocity' missing."

    # Generate Val Features
    df_val_feats = dm.get_val_features(load_cached_data=False)
    print(f"Val Features Shape: {df_val_feats.shape}")
    assert not df_val_feats.empty, "Val features DataFrame is empty."

    # =========================================================================
    # 4. Training Pipeline (VDAM-E)
    # =========================================================================
    print("\n--- Running Training Pipeline ---")
    trainer = TrainingCore()

    # Execute the full run (Scouts -> Mining -> Experts)
    # We pass load_cached_data=True, but since we changed paths/hashes, it will recompute/re-train
    trainer.run(load_cached_data=True)

    # Verifications
    scout_files = os.listdir(trainer.scout_dir)
    expert_files = os.listdir(trainer.expert_dir)
    cache_files = os.listdir(trainer.cache_dir)

    print(f"Scout Models: {scout_files}")
    print(f"Expert Models: {expert_files}")

    assert len(scout_files) >= 3, "Not all scout models were saved."
    assert len(expert_files) >= 3, "Not all expert models were saved."
    assert (
        "hard_negative_indices.npy" in cache_files
    ), "Hard negative indices not cached."

    # =========================================================================
    # 5. Evaluation Pipeline
    # =========================================================================
    print("\n--- Running Evaluation Pipeline ---")
    evaluator = Evaluator()

    # Run optimization and submission generation
    evaluator.run(load_cached_data=True)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head())

    # Check format
    assert list(df_sub.columns) == [
        "contact_id",
        "contact",
    ], "Submission columns incorrect."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary values."

    # Check if predictions are not all zeros (unless dataset is tiny and has no contacts)
    # With only 1 test play, it's possible all are 0, but we just check structural validity here.

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    run_demo()
