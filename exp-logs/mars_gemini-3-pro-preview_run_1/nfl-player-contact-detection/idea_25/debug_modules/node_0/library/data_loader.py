import os
import pandas as pd
import numpy as np
from library.config import Config


def load_metadata(dataset_type="train"):
    """
    Loads the metadata CSV file for the specified dataset type.

    Args:
        dataset_type (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if dataset_type == "train":
        path = Config.TRAIN_METADATA_PATH
    elif dataset_type == "val":
        path = Config.VAL_METADATA_PATH
    elif dataset_type == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    print(f"Loading {dataset_type} metadata from {path}...")
    return pd.read_csv(path)


def load_tracking(dataset_type="train"):
    """
    Loads the player tracking CSV file.

    Args:
        dataset_type (str): One of 'train', 'test'. Note that 'val' uses 'train' tracking.

    Returns:
        pd.DataFrame: The loaded tracking data.
    """
    if dataset_type in ["train", "val"]:
        path = Config.TRAIN_TRACKING_PATH
    elif dataset_type == "test":
        path = Config.TEST_TRACKING_PATH
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    print(f"Loading tracking data from {path}...")
    return pd.read_csv(path)


def merge_tracking_data(df_meta, df_tracking, dataset_type, load_cached_data=True):
    """
    Merges Player 1 and Player 2 tracking data onto the metadata dataframe.
    Implements parameter-aware caching using Parquet.

    Args:
        df_meta (pd.DataFrame): The metadata dataframe.
        df_tracking (pd.DataFrame): The tracking dataframe.
        dataset_type (str): 'train', 'val', or 'test' (used for cache naming).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The merged dataframe with tracking columns for p1 and p2.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache filename
    cache_filename = f"merged_{dataset_type}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading merged {dataset_type} data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Error loading cache ({e}). Recomputing...")

    print(f"Computing merge for {dataset_type}...")

    # 2. Prepare Tracking Data
    # Select only necessary columns to optimize memory
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]
    # Filter columns
    df_track_sub = df_tracking[track_cols].copy()

    # Ensure join keys match types
    df_track_sub["game_play"] = df_track_sub["game_play"].astype(str)
    df_track_sub["step"] = df_track_sub["step"].astype(int)
    df_track_sub["nfl_player_id"] = df_track_sub["nfl_player_id"].astype(int)

    df_meta["game_play"] = df_meta["game_play"].astype(str)
    df_meta["step"] = df_meta["step"].astype(int)
    df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

    # 3. Merge Player 1 Tracking
    # Rename map for P1
    p1_rename = {
        c: f"{c}_p1"
        for c in track_cols
        if c not in ["game_play", "step", "nfl_player_id"]
    }

    # Left join
    df_merged = pd.merge(
        df_meta,
        df_track_sub,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    # Rename and drop redundant key
    df_merged = df_merged.rename(columns=p1_rename)
    df_merged = df_merged.drop(columns=["nfl_player_id"])

    # 4. Merge Player 2 Tracking
    # Handle Ground ('G') vs Player
    mask_ground = df_merged["nfl_player_id_2"] == "G"

    # Split
    df_ground = df_merged[mask_ground].copy()
    df_players = df_merged[~mask_ground].copy()

    if not df_players.empty:
        # Convert ID to int for merge
        df_players["nfl_player_id_2"] = df_players["nfl_player_id_2"].astype(int)

        # Rename map for P2
        p2_rename = {
            c: f"{c}_p2"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }

        # Left join
        df_players = pd.merge(
            df_players,
            df_track_sub,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        # Rename and drop redundant key
        df_players = df_players.rename(columns=p2_rename)
        df_players = df_players.drop(columns=["nfl_player_id"])

    # Concatenate (df_ground will have NaNs for _p2 columns)
    df_final = pd.concat([df_players, df_ground], axis=0, ignore_index=True)

    # 5. Save to Cache
    print(f"Saving merged {dataset_type} data to cache: {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    return df_final
