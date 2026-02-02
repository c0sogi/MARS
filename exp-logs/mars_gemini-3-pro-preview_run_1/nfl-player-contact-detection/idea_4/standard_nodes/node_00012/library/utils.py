import os
import hashlib
import json
import pandas as pd
import numpy as np
from library.config import Config


def generate_cache_key():
    """
    Generates a unique MD5 hash based on the feature engineering configuration
    defined in the Config class. This ensures that if parameters like window size
    or feature flags change, a new cache file is generated.

    Returns:
        str: A hexadecimal hash string representing the current configuration.
    """
    # Extract relevant configuration parameters
    config_dict = {
        "WINDOW_HALF_SIZE": Config.WINDOW_HALF_SIZE,
        "USE_SPATIAL_DENSITY": Config.USE_SPATIAL_DENSITY,
        "USE_IMPACT_PHYSICS": Config.USE_IMPACT_PHYSICS,
        "SEED": Config.SEED,
    }

    # Serialize to JSON with sorted keys to ensure determinism
    config_str = json.dumps(config_dict, sort_keys=True)

    # Compute MD5 hash
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def load_raw_data(split="train", load_tracking=True):
    """
    Loads the metadata and optionally the tracking data for a specific split.

    Args:
        split (str): One of 'train', 'val', or 'test'.
        load_tracking (bool): Whether to load the associated player tracking data.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame or None)
            - metadata dataframe
            - tracking dataframe (or None if load_tracking is False)
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        # Training split uses the main train tracking file
        track_path = Config.TRAIN_TRACKING_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        # Validation split is a subset of the training games, so it uses the train tracking file
        track_path = Config.TRAIN_TRACKING_PATH
    else:  # test
        meta_path = Config.TEST_METADATA_PATH
        track_path = Config.TEST_TRACKING_PATH

    # Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at: {meta_path}")

    # Efficiently read CSV
    # Specifying types for memory efficiency where possible
    df_meta = pd.read_csv(meta_path)

    # Load Tracking Data if requested
    df_track = None
    if load_tracking:
        if not os.path.exists(track_path):
            raise FileNotFoundError(f"Tracking file not found at: {track_path}")

        # Tracking data can be large, read efficiently
        # We can optimize types:
        # - game_play: string/category
        # - nfl_player_id: int (if no NaNs) or float
        # - step: int
        # - position, team: category
        dtype_map = {
            "game_play": "object",
            "game_key": "int32",
            "play_id": "int32",
            "nfl_player_id": "float32",  # Can contain NaNs sometimes? usually int but safe with float then cast
            "step": "int16",
            "position": "category",
            "team": "category",
            "jersey_number": "float32",
            "x_position": "float32",
            "y_position": "float32",
            "speed": "float32",
            "distance": "float32",
            "orientation": "float32",
            "direction": "float32",
            "acceleration": "float32",
            "sa": "float32",
        }

        # Note: read_csv might warn if columns are missing from dtype_map,
        # but tracking files are standard. We'll use standard read and then downcast
        # if memory is an issue, but standard read is usually fine for 1-2GB on this machine.
        df_track = pd.read_csv(track_path)

        # Convert datetime to datetime objects for accurate time operations
        if "datetime" in df_track.columns:
            df_track["datetime"] = pd.to_datetime(
                df_track["datetime"], format="mixed", errors="coerce"
            )

    return df_meta, df_track


def save_to_parquet(df, filename):
    """
    Saves a dataframe to a parquet file in the working directory defined in Config.

    Args:
        df (pd.DataFrame): Data to save.
        filename (str): Name of the file (e.g., 'train_features.parquet').
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    path = os.path.join(Config.WORKING_DIR, filename)
    df.to_parquet(path, index=False)
    print(f"Saved cache to {path}")


def load_from_parquet(filename):
    """
    Loads a dataframe from a parquet file in the working directory.

    Args:
        filename (str): Name of the file.

    Returns:
        pd.DataFrame or None: The loaded dataframe, or None if it doesn't exist.
    """
    path = os.path.join(Config.WORKING_DIR, filename)
    if os.path.exists(path):
        print(f"Loading cache from {path}")
        return pd.read_parquet(path)
    return None
