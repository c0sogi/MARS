import pandas as pd
import logging
from library.config import Config
from library.utils import load_or_process_data


def _process_raw_data(path, is_test=False):
    """
    Reads a raw Parquet file, enforces column allow-listing to prevent leakage,
    and returns the cleaned DataFrame.

    Args:
        path (str): Path to the raw metadata Parquet file.
        is_test (bool): Whether the dataset is the test set (excludes target).

    Returns:
        pd.DataFrame: The cleaned DataFrame with only allow-listed columns.
    """
    logging.info(f"Reading raw data from {path}")
    df = pd.read_parquet(path)

    # Define strict allow-list of columns
    cols_to_keep = (
        [Config.ID_COL]
        + Config.TEXT_COLS
        + [Config.COMMUNITY_COL]
        + Config.METADATA_COLS
    )

    # Add target column for train/val sets
    if not is_test:
        cols_to_keep.append(Config.TARGET_COL)

    # Verify all columns exist before selection
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {path}: {missing_cols}")

    # Select only allow-listed columns (implicitly drops _at_retrieval and other leakage)
    df_clean = df[cols_to_keep].copy()

    # Explicit leakage check (redundant due to allow-list, but good for safety)
    leakage_cols = [c for c in df_clean.columns if "_at_retrieval" in c]
    if leakage_cols:
        raise ValueError(
            f"Leakage detected! Columns contain '_at_retrieval': {leakage_cols}"
        )

    return df_clean


def load_datasets(load_cache=True):
    """
    Loads the Train, Validation, and Test datasets.
    Uses caching to store cleaned versions of the data.

    Args:
        load_cache (bool): If True, attempts to load processed files from the working directory.

    Returns:
        tuple: (train_df, val_df, test_df)
    """

    # Load Train
    train_df = load_or_process_data(
        file_name="train_cleaned.parquet",
        process_fn=_process_raw_data,
        load_cache=load_cache,
        file_type="parquet",
        path=Config.TRAIN_PATH,
        is_test=False,
    )

    # Load Validation
    val_df = load_or_process_data(
        file_name="val_cleaned.parquet",
        process_fn=_process_raw_data,
        load_cache=load_cache,
        file_type="parquet",
        path=Config.VAL_PATH,
        is_test=False,
    )

    # Load Test
    test_df = load_or_process_data(
        file_name="test_cleaned.parquet",
        process_fn=_process_raw_data,
        load_cache=load_cache,
        file_type="parquet",
        path=Config.TEST_PATH,
        is_test=True,
    )

    logging.info(
        f"Datasets loaded. Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}"
    )
    return train_df, val_df, test_df
