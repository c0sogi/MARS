import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_TRACKING_PATH,
    TEST_TRACKING_PATH,
    TRAIN_HELMETS_PATH,
    TEST_HELMETS_PATH,
    WORKING_DIR,
)
from library.utils import seed_everything


def _load_file_with_cache(
    csv_path: str, cache_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Generic function to load a CSV file with Parquet caching.

    Args:
        csv_path: Path to the source CSV file.
        cache_name: Name of the cache file (e.g., 'train_tracking.parquet').
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded data.
    """
    cache_path = os.path.join(WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading cached data from {cache_path}...")
            return pd.read_parquet(cache_path)
        except Exception:
            # If load fails, fall through to re-compute
            pass

    # 2. Compute (Read CSV)
    # print(f"Reading raw data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Optimization: Downcast types if possible to save memory,
    # though specific type enforcement is better handled in feature engineering.
    # Here we just ensure we save it.

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved cache to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


def load_tracking_data(
    mode: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads player tracking data for the specified mode.

    Args:
        mode: 'train', 'validation', or 'test'.
        load_cached_data: Whether to use cached parquet files.
    """
    if mode in ["train", "validation"]:
        source_path = TRAIN_TRACKING_PATH
        cache_name = "train_tracking.parquet"
    elif mode == "test":
        source_path = TEST_TRACKING_PATH
        cache_name = "test_tracking.parquet"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return _load_file_with_cache(source_path, cache_name, load_cached_data)


def load_helmet_data(
    mode: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads helmet baseline data for the specified mode.

    Args:
        mode: 'train', 'validation', or 'test'.
        load_cached_data: Whether to use cached parquet files.
    """
    if mode in ["train", "validation"]:
        source_path = TRAIN_HELMETS_PATH
        cache_name = "train_helmets.parquet"
    elif mode == "test":
        source_path = TEST_HELMETS_PATH
        cache_name = "test_helmets.parquet"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return _load_file_with_cache(source_path, cache_name, load_cached_data)


def load_labels(mode: str = "train") -> pd.DataFrame:
    """
    Loads the metadata/labels file for the specified mode.

    Args:
        mode: 'train', 'validation', or 'test'.
    """
    if mode == "train":
        path = TRAIN_META_PATH
    elif mode == "validation":
        path = VAL_META_PATH
    elif mode == "test":
        path = TEST_META_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata file not found at {path}. Ensure metadata generation step was run."
        )

    return pd.read_csv(path)


def load_dataset(mode: str = "train", load_cached_data: bool = True):
    """
    High-level function to load all necessary data for a given pipeline stage.

    Args:
        mode: 'train', 'validation', or 'test'.
        load_cached_data: Whether to use cached data for heavy files (tracking/helmets).

    Returns:
        tuple: (labels_df, tracking_df, helmets_df)
    """
    # Load Labels (Metadata)
    labels_df = load_labels(mode)

    # Load Tracking Data
    tracking_df = load_tracking_data(mode, load_cached_data)

    # Load Helmet Data
    helmets_df = load_helmet_data(mode, load_cached_data)

    return labels_df, tracking_df, helmets_df


def partition_streams(df: pd.DataFrame):
    """
    Splits the dataframe into two streams based on the contact type.

    Stream A: Player-Player interactions (nfl_player_id_2 != 'G')
    Stream B: Player-Ground interactions (nfl_player_id_2 == 'G')

    Args:
        df: Input dataframe containing 'nfl_player_id_2'.

    Returns:
        tuple: (df_stream_a, df_stream_b)
    """
    # Ensure nfl_player_id_2 is treated as string to handle 'G' correctly
    # In some pandas versions, mixed types can cause issues.
    if "nfl_player_id_2" not in df.columns:
        raise KeyError("Column 'nfl_player_id_2' not found in dataframe.")

    # Create mask for Ground contact
    is_ground = df["nfl_player_id_2"].astype(str) == "G"

    # Split
    df_stream_b = df[is_ground].copy()
    df_stream_a = df[~is_ground].copy()

    return df_stream_a, df_stream_b


def prepare_test_skeleton() -> pd.DataFrame:
    """
    Loads the test skeleton data.
    This uses the generated metadata/test.csv which is derived from sample_submission.csv
    and contains parsed columns (game_play, step, player ids).

    Returns:
        pd.DataFrame: The test skeleton dataframe.
    """
    return load_labels(mode="test")
