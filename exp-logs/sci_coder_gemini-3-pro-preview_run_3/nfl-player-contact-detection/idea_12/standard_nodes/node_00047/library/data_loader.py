import os
import pandas as pd
import numpy as np
from library.config import Config

# Ensure working directory exists for caching
os.makedirs(Config.WORKING_DIR, exist_ok=True)


def load_metadata(split: str = "train") -> pd.DataFrame:
    """
    Loads the metadata for the specified split (train, validation, or test) from the
    pre-generated metadata CSVs. Standardizes ID columns to strings for consistency.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_META_PATH
    elif split == "validation":
        path = Config.VAL_META_PATH
    elif split == "test":
        path = Config.TEST_META_PATH
    else:
        raise ValueError(
            f"Invalid split: {split}. Must be 'train', 'validation', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    df = pd.read_csv(path)

    # Standardize ID columns to string to ensure consistent merging
    # nfl_player_id_1 is typically int in raw data, convert to string
    if "nfl_player_id_1" in df.columns:
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)

    # nfl_player_id_2 can be 'G' (ground) or int, convert all to string
    if "nfl_player_id_2" in df.columns:
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)

    # Ensure game_play is string
    if "game_play" in df.columns:
        df["game_play"] = df["game_play"].astype(str)

    return df


def load_tracking(
    dataset_type: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads player tracking data. Implements caching to Parquet to speed up subsequent runs.

    Args:
        dataset_type (str): 'train' (covers both train/val splits) or 'test'.
        load_cached_data (bool): If True, attempts to load from the local cache first.

    Returns:
        pd.DataFrame: The tracking dataframe with standardized types.
    """
    if dataset_type not in ["train", "test"]:
        raise ValueError("dataset_type must be 'train' or 'test'")

    cache_filename = f"tracking_{dataset_type}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cached tracking data: {e}. Reloading from raw source."
            )

    # 2. Process from raw CSV
    if dataset_type == "train":
        raw_path = Config.TRAIN_TRACKING_PATH
    else:
        raw_path = Config.TEST_TRACKING_PATH

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw tracking file not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    # Type casting and standardization
    if "nfl_player_id" in df.columns:
        df["nfl_player_id"] = df["nfl_player_id"].astype(str)

    if "game_play" in df.columns:
        df["game_play"] = df["game_play"].astype(str)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Save to cache for future use
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save tracking cache to {cache_path}: {e}")

    return df


def load_helmets(
    dataset_type: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads baseline helmet predictions. Implements caching to Parquet.

    Args:
        dataset_type (str): 'train' or 'test'.
        load_cached_data (bool): If True, attempts to load from the local cache first.

    Returns:
        pd.DataFrame: The helmets dataframe.
    """
    if dataset_type not in ["train", "test"]:
        raise ValueError("dataset_type must be 'train' or 'test'")

    cache_filename = f"helmets_{dataset_type}.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cached helmet data: {e}. Reloading from raw source.")

    # 2. Process from raw CSV
    if dataset_type == "train":
        raw_path = Config.TRAIN_HELMETS_PATH
    else:
        raw_path = Config.TEST_HELMETS_PATH

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw helmets file not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    # Type casting
    if "nfl_player_id" in df.columns:
        # Handle potential NaNs in imperfect predictions by filling with sentinel
        # Convert to int first to remove decimals (e.g. 123.0 -> 123), then to string
        df["nfl_player_id"] = df["nfl_player_id"].fillna(-999).astype(int).astype(str)

    if "game_play" in df.columns:
        df["game_play"] = df["game_play"].astype(str)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save helmets cache to {cache_path}: {e}")

    return df


def load_video_metadata(dataset_type: str = "train") -> pd.DataFrame:
    """
    Loads video metadata CSVs.

    Args:
        dataset_type (str): 'train' or 'test'.

    Returns:
        pd.DataFrame: Video metadata.
    """
    if dataset_type == "train":
        path = Config.TRAIN_VIDEO_META_PATH
    elif dataset_type == "test":
        path = Config.TEST_VIDEO_META_PATH
    else:
        raise ValueError("dataset_type must be 'train' or 'test'")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Video metadata file not found at: {path}")

    df = pd.read_csv(path)

    if "game_play" in df.columns:
        df["game_play"] = df["game_play"].astype(str)

    return df


def load_sample_submission() -> pd.DataFrame:
    """
    Loads the raw sample submission file.

    Returns:
        pd.DataFrame: The sample submission dataframe.
    """
    path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample submission file not found at: {path}")
    return pd.read_csv(path)
