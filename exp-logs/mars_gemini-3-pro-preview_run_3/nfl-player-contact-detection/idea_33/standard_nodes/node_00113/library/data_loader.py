import pandas as pd
import numpy as np
import os
from library.config import Config


def load_metadata(split_name, sample_ratio=1.0):
    """
    Loads the metadata for the specified split (train, validation, or test).

    Args:
        split_name (str): The name of the split ('train', 'validation', 'test').
        sample_ratio (float): Fraction of data to load (0.0 to 1.0). Useful for debugging.

    Returns:
        pd.DataFrame: The loaded metadata dataframe.

    Raises:
        ValueError: If split_name is invalid.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If required columns are missing.
    """
    if split_name == "train":
        path = Config.TRAIN_META_PATH
    elif split_name == "validation":
        path = Config.VAL_META_PATH
    elif split_name == "test":
        path = Config.TEST_META_PATH
    else:
        raise ValueError(
            f"Invalid split_name: {split_name}. Must be 'train', 'validation', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    # Strict Schema Validation
    required_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
    ]
    if split_name != "test":
        required_cols.append("contact")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")

    # Ensure IDs are strings for consistent merging
    df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)
    df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

    # Sampling for debugging/development
    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=Config.SEED).reset_index(
            drop=True
        )

    return df


def load_tracking(split_name):
    """
    Loads the player tracking data corresponding to the split.

    Args:
        split_name (str): 'train', 'validation', or 'test'.

    Returns:
        pd.DataFrame: The tracking data.

    Raises:
        RuntimeError: If schema validation fails or features are zero-filled.
    """
    # Map validation split to the training tracking file
    if split_name in ["train", "validation"]:
        path = Config.TRAIN_TRACKING_PATH
    elif split_name == "test":
        path = Config.TEST_TRACKING_PATH
    else:
        raise ValueError(f"Invalid split_name: {split_name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Tracking file not found at {path}")

    df = pd.read_csv(path)

    # Strict Schema Validation
    # Ensure all features required by Config.STREAM_A/B_FEATURES are present
    required_cols = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Tracking data missing required columns: {missing}")

    # Check for zero-filled features (integrity check)
    # If a dynamic feature like speed or acceleration is strictly all zeros, data is likely corrupt.
    for col in ["speed", "acceleration"]:
        if df[col].sum() == 0 and df[col].max() == 0:
            raise RuntimeError(
                f"Tracking feature '{col}' appears to be zero-filled/empty."
            )

    # Ensure IDs are strings
    df["nfl_player_id"] = df["nfl_player_id"].astype(str)

    return df


def load_helmets(split_name):
    """
    Loads the baseline helmet detection data.

    Args:
        split_name (str): 'train', 'validation', or 'test'.

    Returns:
        pd.DataFrame: The helmet data.
    """
    if split_name in ["train", "validation"]:
        path = Config.TRAIN_HELMETS_PATH
    elif split_name == "test":
        path = Config.TEST_HELMETS_PATH
    else:
        raise ValueError(f"Invalid split_name: {split_name}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Helmets file not found at {path}")

    df = pd.read_csv(path)

    # Strict Schema Validation
    required_cols = [
        "game_play",
        "frame",
        "nfl_player_id",
        "left",
        "width",
        "top",
        "height",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Helmets data missing required columns: {missing}")

    df["nfl_player_id"] = df["nfl_player_id"].astype(str)

    return df


def merge_data(df_labels, df_tracking, df_helmets=None):
    """
    Merges the labels dataframe with tracking data for both Player 1 and Player 2.

    Args:
        df_labels (pd.DataFrame): The metadata/labels dataframe.
        df_tracking (pd.DataFrame): The player tracking dataframe.
        df_helmets (pd.DataFrame, optional): The helmets dataframe.
            Note: This function primarily handles tracking merges based on 'step'.
            Helmet merging requires frame-to-step mapping (video metadata) which is
            handled in downstream feature engineering.

    Returns:
        pd.DataFrame: The merged dataframe with tracking features suffixed by _p1 and _p2.
    """
    if df_labels.empty:
        raise RuntimeError("Labels DataFrame is empty.")
    if df_tracking.empty:
        raise RuntimeError("Tracking DataFrame is empty.")

    # Identify tracking feature columns (exclude join keys and metadata)
    exclude_cols = {
        "game_play",
        "step",
        "nfl_player_id",
        "datetime",
        "game_key",
        "play_id",
        "time",
        "team",
        "jersey_number",
        "position",
    }
    # We include 'position' in exclude list if we don't want it, but Config uses it?
    # Config uses 'position' (football position) in categorical analysis but not explicitly in feature lists
    # except maybe for filtering. The feature lists in Config are numerical.
    # Let's keep all numerical columns plus necessary categoricals if needed.
    # Based on Config, we need x, y, speed, accel, dir, orient.

    # Dynamically select numeric columns + specific features
    track_feats = [c for c in df_tracking.columns if c not in exclude_cols]

    # --- Prepare Player 1 Tracking ---
    # Select keys + features
    p1_cols = ["game_play", "step", "nfl_player_id"] + track_feats
    df_p1 = df_tracking[p1_cols].copy()

    # Rename for merge
    rename_map_p1 = {c: f"{c}_p1" for c in track_feats}
    rename_map_p1["nfl_player_id"] = "nfl_player_id_1"
    df_p1 = df_p1.rename(columns=rename_map_p1)

    # --- Prepare Player 2 Tracking ---
    p2_cols = ["game_play", "step", "nfl_player_id"] + track_feats
    df_p2 = df_tracking[p2_cols].copy()

    # Rename for merge
    rename_map_p2 = {c: f"{c}_p2" for c in track_feats}
    rename_map_p2["nfl_player_id"] = "nfl_player_id_2"
    df_p2 = df_p2.rename(columns=rename_map_p2)

    # --- Merge ---
    # Merge Player 1 (Left join: we expect P1 to always be a valid player)
    df_merged = pd.merge(
        df_labels, df_p1, on=["game_play", "step", "nfl_player_id_1"], how="left"
    )

    # Merge Player 2 (Left join: P2 can be a player or 'G'. If 'G', tracking will be NaN)
    df_merged = pd.merge(
        df_merged, df_p2, on=["game_play", "step", "nfl_player_id_2"], how="left"
    )

    return df_merged
