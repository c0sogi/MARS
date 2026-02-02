import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("data_loader")


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if (
            col_type != object
            and col_type.name != "category"
            and "datetime" not in col_type.name
        ):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(
                        np.float32
                    )  # float16 has low precision, using float32 is safer
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(
        f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
    )
    return df


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

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

    logger.info(f"Loading {split} metadata from {path}...")
    df = pd.read_csv(path)

    # Ensure standard types for key columns
    if "step" in df.columns:
        df["step"] = df["step"].astype(int)
    if "nfl_player_id_1" in df.columns:
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)
    # nfl_player_id_2 can contain 'G', so it remains object/str

    logger.info(f"Loaded {len(df)} rows for {split} metadata.")
    return df


def load_tracking(split="train", load_cached_data=True):
    """
    Loads the player tracking data. Uses caching to speed up subsequent loads.

    Args:
        split (str): 'train' or 'test'. Note that validation data usually comes from the train tracking file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The tracking data.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if split == "train":
        raw_path = Config.TRAIN_TRACKING_PATH
        cache_filename = "train_tracking_processed.parquet"
    elif split == "test":
        raw_path = Config.TEST_TRACKING_PATH
        cache_filename = "test_tracking_processed.parquet"
    else:
        raise ValueError(
            f"Invalid split for tracking data: {split}. Must be 'train' or 'test'."
        )

    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {split} tracking data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Successfully loaded {len(df)} rows from cache.")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Falling back to raw CSV.")

    # 2. Load from raw CSV
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw tracking file not found at {raw_path}")

    logger.info(f"Loading raw {split} tracking data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 3. Preprocessing and Optimization
    logger.info("Optimizing memory usage...")

    # Convert datetime to datetime object if present
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Downcast numeric columns
    df = reduce_mem_usage(df)

    # 4. Save to cache
    logger.info(f"Saving processed {split} tracking data to cache: {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df
