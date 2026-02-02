import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    TRAIN_HELMETS_PATH,
    TEST_HELMETS_PATH,
    TRAIN_META_CSV,
    VAL_META_CSV,
    TEST_META_CSV,
    WORKING_DIR,
    FPS,
    TRACKING_FREQ,
)


def _merge_data_sources(meta_df, tracking_df, helmets_df):
    """
    Merges metadata (labels) with tracking and helmet data.
    Performs a Left Join of Helmets onto Tracking, and then Tracking onto Labels.
    """
    # 1. Filter Tracking and Helmets to relevant plays to reduce memory usage
    relevant_plays = meta_df["game_play"].unique()
    tracking_df = tracking_df[tracking_df["game_play"].isin(relevant_plays)].copy()
    helmets_df = helmets_df[helmets_df["game_play"].isin(relevant_plays)].copy()

    # 2. Standardize IDs to string for consistent merging
    # Labels have 'G' for ground, so we treat all IDs as strings.
    meta_df["nfl_player_id_1"] = meta_df["nfl_player_id_1"].astype(str)
    meta_df["nfl_player_id_2"] = meta_df["nfl_player_id_2"].astype(str)
    tracking_df["nfl_player_id"] = tracking_df["nfl_player_id"].astype(str)
    helmets_df["nfl_player_id"] = helmets_df["nfl_player_id"].astype(str)

    # 3. Prepare Helmets (Pivot View)
    # We need to flatten Sideline and Endzone views into a single row per player/frame
    bbox_cols = ["left", "width", "top", "height"]

    # Filter to known views
    helmets_df = helmets_df[helmets_df["view"].isin(["Sideline", "Endzone"])].copy()

    # Split by view
    helmets_sideline = helmets_df[helmets_df["view"] == "Sideline"].copy()
    helmets_endzone = helmets_df[helmets_df["view"] == "Endzone"].copy()

    # Rename columns for flattened format
    rename_map_sl = {c: f"sideline_{c}" for c in bbox_cols}
    helmets_sideline = helmets_sideline.rename(columns=rename_map_sl)

    rename_map_ez = {c: f"endzone_{c}" for c in bbox_cols}
    helmets_endzone = helmets_endzone.rename(columns=rename_map_ez)

    # Select key columns + features
    key_cols = ["game_play", "nfl_player_id", "frame"]
    helmets_sideline = helmets_sideline[key_cols + list(rename_map_sl.values())]
    helmets_endzone = helmets_endzone[key_cols + list(rename_map_ez.values())]

    # Outer merge to keep data even if a player is only visible in one view
    helmets_wide = pd.merge(helmets_sideline, helmets_endzone, on=key_cols, how="outer")

    # 4. Enrich Tracking with Helmets
    # Calculate estimated frame for tracking to join with visual data
    # Snap (step 0) is at frame 300 (5 seconds * 59.94 FPS)
    # Ratio: 59.94 frames / 10 steps = 5.994 frames/step
    frames_per_step = FPS / TRACKING_FREQ
    tracking_df["frame_est"] = (
        (300 + tracking_df["step"] * frames_per_step).round().astype(int)
    )

    # Left Join: Keep all tracking data. If no helmet detected, we get NaNs (to be filled later)
    tracking_enriched = pd.merge(
        tracking_df,
        helmets_wide,
        left_on=["game_play", "nfl_player_id", "frame_est"],
        right_on=["game_play", "nfl_player_id", "frame"],
        how="left",
    )

    # Drop the redundant 'frame' column from helmets
    if "frame" in tracking_enriched.columns:
        tracking_enriched = tracking_enriched.drop(columns=["frame"])

    # 5. Merge Enriched Tracking to Metadata (Labels)
    # We attach this data for Player 1 and Player 2 separately.

    # Define columns to transfer
    track_cols = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]
    visual_cols = [
        c for c in tracking_enriched.columns if "sideline_" in c or "endzone_" in c
    ]
    cols_to_merge = track_cols + visual_cols

    # Prepare P1 dataframe
    p1_df = tracking_enriched[
        ["game_play", "step", "nfl_player_id"] + cols_to_merge
    ].copy()
    p1_df.columns = ["game_play", "step", "nfl_player_id"] + [
        f"p1_{c}" for c in cols_to_merge
    ]

    # Prepare P2 dataframe
    p2_df = tracking_enriched[
        ["game_play", "step", "nfl_player_id"] + cols_to_merge
    ].copy()
    p2_df.columns = ["game_play", "step", "nfl_player_id"] + [
        f"p2_{c}" for c in cols_to_merge
    ]

    # Merge P1
    merged = pd.merge(
        meta_df,
        p1_df,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    merged = merged.drop(columns=["nfl_player_id"])

    # Merge P2
    merged = merged.merge(
        p2_df,
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    merged = merged.drop(columns=["nfl_player_id"])

    # 6. Fill Missing Values with Sentinel
    # This covers:
    # - Missing helmet detections (visual_cols)
    # - Ground contacts where P2 is 'G' (all p2_ cols)
    # - Missing tracking data
    feature_cols = [
        c for c in merged.columns if c.startswith("p1_") or c.startswith("p2_")
    ]
    merged[feature_cols] = merged[feature_cols].fillna(-999)

    return merged


def load_dataset(mode="train", load_cached_data=True, sample_size=None):
    """
    Loads the dataset for the specified mode (train/validation/test).

    Args:
        mode (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        sample_size (int, optional): If set, samples the metadata before merging for debugging.

    Returns:
        pd.DataFrame: The unified raw dataframe containing labels and merged tracking/helmet data.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache path
    cache_filename = f"{mode}_merged_raw"
    if sample_size:
        cache_filename += f"_sample{sample_size}"
    cache_path = os.path.join(WORKING_DIR, f"{cache_filename}.parquet")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} dataset from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Generating {mode} dataset from scratch...")

    # Determine input files based on mode
    if mode == "train":
        meta_path = TRAIN_META_CSV
        track_path = TRAIN_TRACKING_PATH
        helmet_path = TRAIN_HELMETS_PATH
    elif mode == "validation":
        meta_path = VAL_META_CSV
        # Validation is a subset of the training source files
        track_path = TRAIN_TRACKING_PATH
        helmet_path = TRAIN_HELMETS_PATH
    elif mode == "test":
        meta_path = TEST_META_CSV
        track_path = TEST_TRACKING_PATH
        helmet_path = TEST_HELMETS_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Load Metadata
    print(f"Loading metadata from {meta_path}...")
    meta_df = pd.read_csv(meta_path)

    # Apply sampling if requested (useful for debugging pipeline)
    if sample_size is not None and sample_size < len(meta_df):
        print(f"Sampling {sample_size} rows from metadata...")
        meta_df = meta_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Load Raw Data
    print(f"Loading tracking data from {track_path}...")
    tracking_df = pd.read_csv(track_path)

    print(f"Loading helmet data from {helmet_path}...")
    helmets_df = pd.read_csv(helmet_path)

    # Merge Data Sources
    print("Merging data sources (Tracking + Helmets + Labels)...")
    merged_df = _merge_data_sources(meta_df, tracking_df, helmets_df)

    # Save to cache
    print(f"Saving to cache {cache_path}...")
    merged_df.to_parquet(cache_path, index=False)

    return merged_df
