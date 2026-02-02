import os
import pandas as pd
import numpy as np
from library.config import Config


def load_metadata(split="train"):
    """
    Loads the metadata for the specified split (train, val, or test).

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)

    # In debug mode, sample the data to speed up development
    if Config.DEBUG:
        sample_size = min(len(df), Config.DEBUG_SAMPLE_SIZE)
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
        print(f"[DEBUG] Sampled {sample_size} rows from {split} metadata.")

    # Ensure consistent data types
    # game_play should be string
    if "game_play" in df.columns:
        df["game_play"] = df["game_play"].astype(str)

    # nfl_player_id_1 is int
    if "nfl_player_id_1" in df.columns:
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)

    # nfl_player_id_2 can be 'G', so keep as object/str, but if it's numeric, it will be loaded as such.
    # We force it to string to handle 'G' consistently if mixed.
    if "nfl_player_id_2" in df.columns:
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

    return df


def load_tracking(dataset_type="train", load_cached_data=True):
    """
    Loads the player tracking data.

    Args:
        dataset_type (str): 'train' (for training/validation) or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The tracking data.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if dataset_type == "train":
        raw_path = Config.TRAIN_TRACKING_PATH
        cache_filename = "train_tracking_processed.parquet"
    elif dataset_type == "test":
        raw_path = Config.TEST_TRACKING_PATH
        cache_filename = "test_tracking_processed.parquet"
    else:
        raise ValueError(
            f"Invalid dataset_type: {dataset_type}. Must be 'train' or 'test'."
        )

    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from raw CSV."
            )

    # 2. Load from raw CSV
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Tracking data not found at {raw_path}")

    df = pd.read_csv(raw_path)

    # 3. Process Data Types
    # game_play -> string
    df["game_play"] = df["game_play"].astype(str)

    # nfl_player_id -> int
    df["nfl_player_id"] = df["nfl_player_id"].astype(int)

    # step -> int
    df["step"] = df["step"].astype(int)

    # datetime -> datetime object
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Optimize numeric types to save memory
    float_cols = [
        "x_position",
        "y_position",
        "speed",
        "distance",
        "orientation",
        "direction",
        "acceleration",
        "sa",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    # 4. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df
