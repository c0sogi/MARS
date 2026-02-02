import os
import pandas as pd
import numpy as np
import shutil
import sys

# Import the provided library modules
from library import config, utils, features, data_loader, models, train, inference


def create_mini_dataset(
    source_meta_path, source_track_path, dest_meta_prefix, dest_track_prefix, n_plays=2
):
    """
    Creates a mini dataset by sampling a few plays from the source files
    and saving them to the working directory.
    """
    print(f"Creating mini dataset from {source_meta_path}...")

    # Load full metadata
    df_meta = pd.read_csv(source_meta_path)

    # Ensure game_play is string to avoid type mismatches
    df_meta["game_play"] = df_meta["game_play"].astype(str)

    # Sample unique plays
    unique_plays = df_meta["game_play"].unique()
    if len(unique_plays) > n_plays:
        selected_plays = unique_plays[:n_plays]
    else:
        selected_plays = unique_plays

    # Filter metadata
    df_mini_meta = df_meta[df_meta["game_play"].isin(selected_plays)].copy()

    # Save mini metadata
    mini_meta_path = os.path.join(
        config.WORKING_DIR, f"{dest_meta_prefix}_metadata.csv"
    )
    df_mini_meta.to_csv(mini_meta_path, index=False)

    # Load full tracking (only for the selected plays to save time/memory)
    # We read the whole file then filter. For a demo, this is acceptable overhead.
    print(f"Loading tracking data from {source_track_path}...")
    # Using chunks to avoid OOM if file is huge, though for this env it fits in RAM.
    # Simple read is fine given the constraints.
    df_track = pd.read_csv(source_track_path)

    # Ensure game_play is string to avoid type mismatches
    if "game_play" in df_track.columns:
        df_track["game_play"] = df_track["game_play"].astype(str)
    else:
        # Fallback if game_play is missing (construct it)
        print("Warning: game_play column missing in tracking. Constructing from keys.")
        df_track["game_play"] = (
            df_track["game_key"].astype(str)
            + "_"
            + df_track["play_id"].astype(str).str.zfill(6)
        )

    # Filter tracking
    df_mini_track = df_track[df_track["game_play"].isin(selected_plays)].copy()

    # Verify we actually got data
    if df_mini_track.empty:
        print(f"Error: No tracking data found for plays: {selected_plays}")
        print(f"Tracking game_play sample: {df_track['game_play'].head().tolist()}")

    # Save mini tracking
    mini_track_path = os.path.join(
        config.WORKING_DIR, f"{dest_track_prefix}_tracking.csv"
    )
    df_mini_track.to_csv(mini_track_path, index=False)

    return mini_meta_path, mini_track_path


def override_config_for_demo(
    mini_train_meta, mini_train_track, mini_val_meta, mini_test_meta, mini_test_track
):
    """
    Overrides configuration paths and hyperparameters for the demo run.
    """
    # Override Paths
    config.TRAIN_METADATA_PATH = mini_train_meta
    config.VAL_METADATA_PATH = mini_val_meta
    config.TEST_METADATA_PATH = mini_test_meta

    config.TRAIN_TRACKING_PATH = mini_train_track
    config.TEST_TRACKING_PATH = mini_test_track

    # Override Model Hyperparameters for Speed
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 16

    config.XGB_PARAMS["n_estimators"] = 10
    config.XGB_PARAMS["max_depth"] = 4

    # Adjust Gating/Mining for small data
    config.HARD_NEGATIVE_THRESHOLD = (
        0.01  # Lower threshold to ensure we find some negatives in small data
    )

    print("Configuration overridden for demo execution.")


def verify_features(df, split_name):
    """
    Verifies that the feature dataframe has the expected structure.
    """
    print(f"Verifying {split_name} features...")

    if df.empty:
        raise AssertionError(f"{split_name} features dataframe is empty!")

    # Check for core columns
    expected_cols = [
        "contact_id",
        "game_play",
        "contact",
        "distance",
        "speed_p1",
        "speed_p2",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in {split_name} features"

    # Check for lag/lead columns (Temporal Windowing)
    # WINDOW_SIZE is 10 in config
    if (
        split_name != "test"
    ):  # Test might not have enough history at edges, but should still have cols
        lag_col = f"distance_lag_{config.WINDOW_SIZE}"
        lead_col = f"distance_lead_{config.WINDOW_SIZE}"
        assert lag_col in df.columns, f"Missing lag column {lag_col}"
        assert lead_col in df.columns, f"Missing lead column {lead_col}"

    # Check for context features
    assert (
        "min_dist_3rd_party" in df.columns
    ), "Missing context feature min_dist_3rd_party"

    print(f"{split_name} features verified. Shape: {df.shape}")


def main():
    # 1. Setup
    utils.seed_everything()
    utils.setup_logging("demo_execution.log")

    # Ensure working directory exists (handled by config, but good to be sure)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print("=== Starting Demo Execution ===")

    # 2. Prepare Mini Datasets
    # We use the existing metadata files in ./metadata and tracking in ./input
    # to create small subsets in ./working

    # Train Subset (2 plays)
    mini_train_meta, mini_train_track = create_mini_dataset(
        os.path.join(config.METADATA_DIR, "train_metadata.csv"),
        os.path.join(config.INPUT_DIR, "train_player_tracking.csv"),
        "mini_train",
        "mini_train",
        n_plays=2,
    )

    # Val Subset (1 play)
    mini_val_meta, mini_val_track = create_mini_dataset(
        os.path.join(config.METADATA_DIR, "val_metadata.csv"),
        os.path.join(
            config.INPUT_DIR, "train_player_tracking.csv"
        ),  # Val uses train tracking file
        "mini_val",
        "mini_val",
        n_plays=1,
    )

    # Merge Train and Val tracking into one file (simulating the full train_player_tracking.csv)
    # This is necessary because library.features expects a single file for both splits.
    print("Merging mini train and val tracking data...")
    df_tr = pd.read_csv(mini_train_track)
    df_va = pd.read_csv(mini_val_track)
    df_combined = pd.concat([df_tr, df_va], axis=0, ignore_index=True)
    df_combined.to_csv(mini_train_track, index=False)

    # Test Subset (1 play)
    mini_test_meta, mini_test_track = create_mini_dataset(
        os.path.join(config.METADATA_DIR, "test_metadata.csv"),
        os.path.join(config.INPUT_DIR, "test_player_tracking.csv"),
        "mini_test",
        "mini_test",
        n_plays=1,
    )

    # 3. Override Config
    override_config_for_demo(
        mini_train_meta,
        mini_train_track,
        mini_val_meta,
        mini_test_meta,
        mini_test_track,
    )

    # 4. Feature Generation Verification
    # We explicitly call the data loader to verify feature generation works
    print("\n--- Testing Feature Generation ---")
    df_train = data_loader.DatasetBuilder().load_data("train", load_cached=False)
    verify_features(df_train, "train")

    df_val = data_loader.DatasetBuilder().load_data("val", load_cached=False)
    verify_features(df_val, "val")

    # 5. Training Pipeline
    print("\n--- Running Training Pipeline ---")
    # This runs Scout -> Mining -> Expert -> Threshold Opt
    expert_lgbm, expert_xgb, best_thresh = train.run_training_pipeline(
        load_cached_features=True,  # Use the features we just generated/cached
        load_cached_mining=False,  # Force mining on new mini data
    )

    # Verify Models exist
    models_dir = os.path.join(config.WORKING_DIR, "models")
    assert os.path.exists(
        os.path.join(models_dir, "expert_lgbm.joblib")
    ), "Expert LGBM model not saved"
    assert os.path.exists(
        os.path.join(models_dir, "expert_xgb.joblib")
    ), "Expert XGB model not saved"
    assert os.path.exists(
        os.path.join(models_dir, "best_threshold.npy")
    ), "Threshold file not saved"

    print(f"Training complete. Best Threshold: {best_thresh}")

    # 6. Inference and Submission
    print("\n--- Running Inference ---")
    # We use the mini test set we created
    submission = inference.create_submission(load_cached_features=False)

    # Verify Submission
    assert "contact_id" in submission.columns
    assert "contact" in submission.columns
    assert submission["contact"].dtype == int or submission["contact"].dtype == np.int64

    # Check if submission file exists on disk
    sub_file_path = "./submission/submission.csv"
    assert os.path.exists(
        sub_file_path
    ), f"Submission file not found at {sub_file_path}"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
