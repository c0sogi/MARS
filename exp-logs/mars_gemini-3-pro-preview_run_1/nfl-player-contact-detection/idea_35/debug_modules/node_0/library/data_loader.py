import os
import pandas as pd
import numpy as np
import gc
from library.config import Config
from library.utils import setup_logger, ensure_dir

# Initialize Logger
logger = setup_logger("data_loader")


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    logger.info(f"Loading {split} metadata from {path}...")
    df = pd.read_csv(path)

    # Ensure game_play is string to match tracking data
    df["game_play"] = df["game_play"].astype(str)

    return df


def load_tracking(split: str) -> pd.DataFrame:
    """
    Loads the player tracking data relevant to the split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The tracking dataframe.
    """
    # Validation set is a subset of the training plays, so it uses train tracking data.
    if split in ["train", "val"]:
        path = Config.TRAIN_TRACKING_PATH
    elif split == "test":
        path = Config.TEST_TRACKING_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    logger.info(f"Loading tracking data for {split} from {path}...")
    df = pd.read_csv(path)

    # Ensure game_play is string
    df["game_play"] = df["game_play"].astype(str)

    # Optimize memory types
    df["nfl_player_id"] = df["nfl_player_id"].astype(int)
    df["step"] = df["step"].astype(int)

    return df


def merge_tracking_data(df_meta: pd.DataFrame, df_track: pd.DataFrame) -> pd.DataFrame:
    """
    Merges tracking data onto the metadata for both Player 1 and Player 2.

    Args:
        df_meta (pd.DataFrame): Metadata containing contact pairs.
        df_track (pd.DataFrame): Player tracking data.

    Returns:
        pd.DataFrame: Merged dataframe with _p1 and _p2 tracking columns.
    """
    logger.info("Merging tracking data...")

    # Columns to merge from tracking
    track_cols = [
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

    # Filter tracking data to only include columns we need + keys
    df_track_sub = df_track[track_cols].copy()

    # --- Merge Player 1 ---
    # Player 1 is always an integer ID
    logger.info("Merging Player 1 tracking data...")
    df_merged = pd.merge(
        df_meta,
        df_track_sub,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename P1 columns
    rename_p1 = {
        c: f"{c}_p1"
        for c in track_cols
        if c not in ["game_play", "step", "nfl_player_id"]
    }
    df_merged = df_merged.rename(columns=rename_p1)
    df_merged = df_merged.drop(
        columns=["nfl_player_id"]
    )  # Drop the join key from tracking

    # --- Merge Player 2 ---
    # Player 2 can be 'G' (Ground) or an integer ID.
    # We need to handle the type mismatch.
    logger.info("Merging Player 2 tracking data...")

    # Create a temporary numeric join key for P2. 'G' becomes NaN.
    df_merged["nfl_player_id_2_numeric"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    df_merged = pd.merge(
        df_merged,
        df_track_sub,
        left_on=["game_play", "step", "nfl_player_id_2_numeric"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )

    # Rename P2 columns
    rename_p2 = {
        c: f"{c}_p2"
        for c in track_cols
        if c not in ["game_play", "step", "nfl_player_id"]
    }
    df_merged = df_merged.rename(columns=rename_p2)

    # Cleanup
    df_merged = df_merged.drop(columns=["nfl_player_id", "nfl_player_id_2_numeric"])

    # Memory management
    del df_track_sub
    gc.collect()

    logger.info(f"Merge complete. Result shape: {df_merged.shape}")
    return df_merged


def load_dataset(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main entry point to load data for a specific split.
    Handles caching of the merged dataframe to avoid repeated expensive joins.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The fully merged dataframe ready for feature engineering.
    """
    # Construct cache path
    cache_filename = f"merged_{split}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {split} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Successfully loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Proceeding to re-process.")

    # 2. Process from scratch
    logger.info(f"Processing {split} data from scratch...")

    # Load raw files
    df_meta = load_metadata(split)
    df_track = load_tracking(split)

    # Merge
    df_merged = merge_tracking_data(df_meta, df_track)

    # 3. Save to cache
    ensure_dir(cache_path)
    logger.info(f"Saving merged {split} data to cache: {cache_path}")
    df_merged.to_parquet(cache_path, index=False)

    return df_merged
