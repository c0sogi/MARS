import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    SENTINEL_VALUE,
    WORKING_DIR,
    SEED,
)


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)
    return df


def load_tracking(split="train"):
    """
    Loads the tracking data CSV.

    Args:
        split (str): 'train' (covers train/val) or 'test'.

    Returns:
        pd.DataFrame: The loaded tracking data.
    """
    if split in ["train", "val"]:
        path = TRAIN_TRACKING_PATH
    elif split == "test":
        path = TEST_TRACKING_PATH
    else:
        raise ValueError(f"Invalid split for tracking: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Tracking file not found at {path}")

    # Load tracking data
    # We specify types to optimize memory and ensure merge consistency
    dtype_map = {
        "game_play": "object",
        "game_key": "int32",
        "play_id": "int32",
        "nfl_player_id": "int32",
        "step": "int32",
        "x_position": "float32",
        "y_position": "float32",
        "speed": "float32",
        "distance": "float32",
        "direction": "float32",
        "orientation": "float32",
        "acceleration": "float32",
        "sa": "float32",
    }

    # Read only necessary columns if possible, but for safety read all then filter
    df = pd.read_csv(path, dtype=dtype_map)
    return df


def _handle_ground_contact(df):
    """
    Internal helper to apply the Sentinel Value Strategy for Ground interactions.

    Args:
        df (pd.DataFrame): Dataframe containing 'nfl_player_id_2' and 'distance'.

    Returns:
        pd.DataFrame: Updated dataframe with sentinel values applied.
    """
    # Identify Ground interactions
    ground_mask = df["nfl_player_id_2"] == "G"

    # Assign Sentinel Value to distance
    if "distance" in df.columns:
        df.loc[ground_mask, "distance"] = SENTINEL_VALUE

    return df


def merge_tracking_data(
    metadata_df, tracking_df, cache_file=None, load_cached_data=True
):
    """
    Merges tracking data onto the metadata (contact pairs).
    Handles caching and the specific logic for Player-Ground interactions.

    Args:
        metadata_df (pd.DataFrame): The contact pairs.
        tracking_df (pd.DataFrame): The player tracking data.
        cache_file (str): Path to save/load the parquet cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The merged dataframe with tracking features for p1 and p2.
    """
    # 1. Cache Check
    if load_cached_data and cache_file and os.path.exists(cache_file):
        print(f"Loading merged data from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print("Merging tracking data (Cache miss or force reload)...")

    # 2. Prepare Tracking Data
    # Select relevant columns to merge
    track_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "sa",
    ]
    # Filter tracking df to relevant columns
    track_sub = tracking_df[track_cols].copy()

    # Ensure merge keys match types
    metadata_df["game_play"] = metadata_df["game_play"].astype(str)
    metadata_df["step"] = metadata_df["step"].astype(int)
    metadata_df["nfl_player_id_1"] = metadata_df["nfl_player_id_1"].astype(int)

    # 3. Split Metadata into Player-Player and Player-Ground
    # This is necessary because 'nfl_player_id_2' is 'G' (str) for ground, but int for players.
    mask_ground = metadata_df["nfl_player_id_2"] == "G"
    df_ground = metadata_df[mask_ground].copy()
    df_players = metadata_df[~mask_ground].copy()

    # ---------------------------------------------------------
    # 4. Process Player-Player Interactions
    # ---------------------------------------------------------
    if not df_players.empty:
        df_players["nfl_player_id_2"] = df_players["nfl_player_id_2"].astype(int)

        # Merge Player 1
        df_players = pd.merge(
            df_players,
            track_sub,
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
        df_players.rename(columns=rename_p1, inplace=True)
        df_players.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        # Merge Player 2
        df_players = pd.merge(
            df_players,
            track_sub,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        # Rename P2 columns
        rename_p2 = {
            c: f"{c}_p2"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_players.rename(columns=rename_p2, inplace=True)
        df_players.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        # Calculate Euclidean Distance
        # We calculate it here to serve as the base feature
        df_players["distance"] = np.sqrt(
            (df_players["x_position_p1"] - df_players["x_position_p2"]) ** 2
            + (df_players["y_position_p1"] - df_players["y_position_p2"]) ** 2
        )

    # ---------------------------------------------------------
    # 5. Process Player-Ground Interactions
    # ---------------------------------------------------------
    if not df_ground.empty:
        # Merge Player 1 only
        df_ground = pd.merge(
            df_ground,
            track_sub,
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
        df_ground.rename(columns=rename_p1, inplace=True)
        df_ground.drop(columns=["nfl_player_id"], inplace=True, errors="ignore")

        # Create P2 columns filled with NaN (since there is no player 2)
        p2_cols = [
            "x_position_p2",
            "y_position_p2",
            "speed_p2",
            "acceleration_p2",
            "direction_p2",
            "orientation_p2",
            "sa_p2",
        ]
        for col in p2_cols:
            df_ground[col] = np.nan

        # Apply Sentinel Value Strategy for Distance
        df_ground = _handle_ground_contact(df_ground)

    # ---------------------------------------------------------
    # 6. Combine and Save
    # ---------------------------------------------------------
    merged_df = pd.concat([df_players, df_ground], axis=0, ignore_index=True)

    # Sort to maintain some order (optional but good for consistency)
    merged_df.sort_values(by=["game_play", "step"], inplace=True)

    # Save to cache
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        print(f"Saving merged data to cache: {cache_file}")
        merged_df.to_parquet(cache_file, index=False)

    return merged_df
