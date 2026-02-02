import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    WORKING_DIR,
    GATING_DISTANCE,
    SEED,
)


def _merge_tracking(df_meta, df_tracking):
    """
    Helper function to merge tracking data onto metadata for both players.
    """
    # Ensure join keys are consistent
    df_meta["game_play"] = df_meta["game_play"].astype(str)
    df_meta["step"] = df_meta["step"].astype(int)
    df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

    # Handle mixed type in nfl_player_id_2 (contains ints and 'G')
    # Create a numeric join key, 'G' becomes NaN
    df_meta["p2_join_key"] = pd.to_numeric(df_meta["nfl_player_id_2"], errors="coerce")

    df_tracking["game_play"] = df_tracking["game_play"].astype(str)
    df_tracking["step"] = df_tracking["step"].astype(int)
    df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(int)

    # Select relevant tracking columns to merge
    # We exclude game_play, step, nfl_player_id from the values to avoid duplication after merge
    tracking_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]
    df_track_sub = df_tracking[tracking_cols].copy()

    # -------------------------
    # Merge Player 1 Tracking
    # -------------------------
    df_merged = pd.merge(
        df_meta,
        df_track_sub,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename Player 1 columns
    rename_p1 = {
        col: f"{col}_p1"
        for col in tracking_cols
        if col not in ["game_play", "step", "nfl_player_id"]
    }
    df_merged.rename(columns=rename_p1, inplace=True)
    df_merged.drop(columns=["nfl_player_id"], inplace=True)  # Drop redundant join col

    # -------------------------
    # Merge Player 2 Tracking
    # -------------------------
    # We use the numeric p2_join_key. Rows with 'G' will have NaN join key and thus NaN tracking data.
    df_merged = pd.merge(
        df_merged,
        df_track_sub,
        left_on=["game_play", "step", "p2_join_key"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename Player 2 columns
    rename_p2 = {
        col: f"{col}_p2"
        for col in tracking_cols
        if col not in ["game_play", "step", "nfl_player_id"]
    }
    df_merged.rename(columns=rename_p2, inplace=True)
    df_merged.drop(columns=["nfl_player_id", "p2_join_key"], inplace=True)

    return df_merged


def apply_geometric_gating(df):
    """
    Filters the dataset based on Euclidean distance between players.
    Keeps:
    1. All Player-Ground interactions (nfl_player_id_2 == 'G').
    2. Player-Player interactions where distance <= GATING_DISTANCE.
    """
    # Calculate Euclidean distance
    # Note: If p2 is 'G', x_position_p2 is NaN, so dist is NaN.
    # If p2 is a player but tracking is missing, dist is NaN.

    dist_sq = (df["x_position_p1"] - df["x_position_p2"]) ** 2 + (
        df["y_position_p1"] - df["y_position_p2"]
    ) ** 2
    df["calculated_distance"] = np.sqrt(dist_sq)

    # Condition 1: Ground contact
    is_ground = df["nfl_player_id_2"] == "G"

    # Condition 2: Player-Player within distance
    # We must handle NaNs in distance. If distance is NaN for a P-P pair,
    # it means tracking is missing. We usually drop these for training
    # as we can't compute features.
    is_close = df["calculated_distance"] <= GATING_DISTANCE

    # Combine masks
    # Keep if Ground OR (Not Ground AND Close)
    # Note: is_close is False if NaN.
    mask = is_ground | is_close

    df_filtered = df[mask].copy()

    # Clean up temporary column
    df_filtered.drop(columns=["calculated_distance"], inplace=True)

    return df_filtered


def load_metadata_and_tracking(split="train", load_cached_data=True):
    """
    Loads metadata and tracking data, merges them, and caches the result.

    Args:
        split (str): 'train' or 'val'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    cache_path = os.path.join(WORKING_DIR, f"merged_{split}.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading merged {split} data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch...")

    # 2. Identify Paths
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = VAL_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 3. Load Raw Data
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not os.path.exists(TRAIN_TRACKING_PATH):
        raise FileNotFoundError(f"Tracking file not found: {TRAIN_TRACKING_PATH}")

    df_meta = pd.read_csv(meta_path)
    df_tracking = pd.read_csv(TRAIN_TRACKING_PATH)

    # 4. Merge
    df_merged = _merge_tracking(df_meta, df_tracking)

    # 5. Save Cache
    print(f"Saving merged {split} data to cache: {cache_path}")
    df_merged.to_parquet(cache_path, index=False)

    return df_merged


def prepare_test_data(load_cached_data=True):
    """
    Loads test metadata (from sample submission) and test tracking, merges them.
    No gating is applied to test data.
    """
    cache_path = os.path.join(WORKING_DIR, "merged_test.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading merged test data from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print("Processing test data from scratch...")

    # 2. Load Raw Data
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found: {TEST_METADATA_PATH}")
    if not os.path.exists(TEST_TRACKING_PATH):
        raise FileNotFoundError(f"Test tracking not found: {TEST_TRACKING_PATH}")

    df_meta = pd.read_csv(TEST_METADATA_PATH)
    df_tracking = pd.read_csv(TEST_TRACKING_PATH)

    # 3. Merge
    df_merged = _merge_tracking(df_meta, df_tracking)

    # 4. Save Cache
    print(f"Saving merged test data to cache: {cache_path}")
    df_merged.to_parquet(cache_path, index=False)

    return df_merged
