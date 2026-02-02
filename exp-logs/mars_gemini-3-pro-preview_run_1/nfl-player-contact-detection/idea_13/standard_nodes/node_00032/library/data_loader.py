import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    GATING_DISTANCE,
    WORKING_DIR,
    SEED,
)
from library.utils import reduce_mem_usage, generate_cache_key


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata for the specified split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)
    # Ensure consistent types
    df["game_play"] = df["game_play"].astype(str)
    df["step"] = df["step"].astype(int)
    df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)
    # nfl_player_id_2 is mixed (int and 'G'), keep as object/str
    df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

    return df


def load_tracking(split: str) -> pd.DataFrame:
    """
    Loads and optimizes the tracking data for the specified split.
    Note: 'train' and 'val' splits both use the training tracking file.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The optimized tracking dataframe.
    """
    if split in ["train", "val"]:
        path = TRAIN_TRACKING_PATH
    elif split == "test":
        path = TEST_TRACKING_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Tracking file not found at {path}")

    df = pd.read_csv(path)
    df = reduce_mem_usage(df, verbose=False)

    # Ensure join keys are correct types
    df["game_play"] = df["game_play"].astype(str)
    df["step"] = df["step"].astype(int)
    df["nfl_player_id"] = df["nfl_player_id"].astype(int)

    return df


def merge_tracking_data(
    df_meta: pd.DataFrame, df_tracking: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges tracking data onto the metadata for both Player 1 and Player 2.

    Args:
        df_meta (pd.DataFrame): Metadata dataframe.
        df_tracking (pd.DataFrame): Tracking dataframe.

    Returns:
        pd.DataFrame: Merged dataframe with _p1 and _p2 suffixes.
    """
    # Columns to merge from tracking
    # We exclude game_play, step, nfl_player_id from the values to avoid duplicates,
    # but we need them for the join keys.
    track_cols = [
        c
        for c in df_tracking.columns
        if c not in ["game_play", "step", "nfl_player_id"]
    ]

    # --- Merge Player 1 ---
    df_merged = pd.merge(
        df_meta,
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename P1 columns
    rename_map_p1 = {col: f"{col}_p1" for col in track_cols}
    df_merged = df_merged.rename(columns=rename_map_p1)
    df_merged = df_merged.drop(columns=["nfl_player_id"])  # Drop the redundant join key

    # --- Merge Player 2 ---
    # Handle 'G' in nfl_player_id_2.
    # We create a temporary numeric column for joining. 'G' becomes NaN.
    df_merged["p2_join_id"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    df_merged = pd.merge(
        df_merged,
        df_tracking,
        left_on=["game_play", "step", "p2_join_id"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename P2 columns
    rename_map_p2 = {col: f"{col}_p2" for col in track_cols}
    df_merged = df_merged.rename(columns=rename_map_p2)
    df_merged = df_merged.drop(columns=["nfl_player_id", "p2_join_id"])

    return df_merged


def apply_geometric_gating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies geometric gating to filter out trivial non-contact pairs.
    Logic:
        - Keep ALL Player-Ground interactions (nfl_player_id_2 == 'G').
        - Keep Player-Player interactions where distance <= GATING_DISTANCE.
        - Keep interactions where distance is NaN (missing tracking data) to be safe.

    Args:
        df (pd.DataFrame): Dataframe containing merged tracking data (x_position_p1, etc.).

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    # Calculate Euclidean distance
    # Ensure coordinates are floats
    x1 = df["x_position_p1"].astype(float)
    y1 = df["y_position_p1"].astype(float)
    x2 = df["x_position_p2"].astype(float)
    y2 = df["y_position_p2"].astype(float)

    dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
    distance = np.sqrt(dist_sq)

    # Define masks
    is_ground = df["nfl_player_id_2"] == "G"
    is_close = distance <= GATING_DISTANCE
    is_nan_dist = distance.isna()

    # Gating condition: Ground OR Close OR Unknown Distance
    mask = is_ground | is_close | is_nan_dist

    filtered_df = df[mask].copy()
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def get_data(
    split: str, load_cached_data: bool = True, apply_gating: bool = True
) -> pd.DataFrame:
    """
    Main entry point to retrieve data for a specific split.
    Handles caching, merging, and optional geometric gating.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        apply_gating (bool): If True, applies geometric gating (The Sieve).
                             Typically True for Train/Val, False for Test.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Generate cache key based on parameters
    cache_params = {
        "split": split,
        "apply_gating": apply_gating,
        "gating_distance": GATING_DISTANCE,
    }
    cache_key = generate_cache_key(cache_params)
    cache_filename = f"data_{split}_{cache_key}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data for split '{split}' from {cache_path}...")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing data for split '{split}' (Gating={apply_gating})...")

    # Load raw data
    df_meta = load_metadata(split)
    df_tracking = load_tracking(split)

    # Merge
    print("Merging tracking data...")
    df_merged = merge_tracking_data(df_meta, df_tracking)

    # Gate
    if apply_gating:
        print(f"Applying geometric gating (Threshold={GATING_DISTANCE} yards)...")
        initial_len = len(df_merged)
        df_merged = apply_geometric_gating(df_merged)
        print(f"Gating complete. Rows: {initial_len} -> {len(df_merged)}")

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    df_merged.to_parquet(cache_path, index=False)

    return df_merged
