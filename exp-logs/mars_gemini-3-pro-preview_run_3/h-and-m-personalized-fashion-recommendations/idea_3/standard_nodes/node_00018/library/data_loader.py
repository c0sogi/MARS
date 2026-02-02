import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional

from library import config
from library import utils


def load_articles(
    load_cached_data: bool = True, debug_nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads the articles dataset.

    Args:
        load_cached_data: If True, attempts to load from a processed parquet cache.
        debug_nrows: If set, returns only the first N rows. Cache is not saved if this is set.

    Returns:
        pd.DataFrame: The articles data.
    """
    cache_path = config.WORKING_DIR / "articles_processed.parquet"

    # 1. Try Loading Cache
    if load_cached_data and debug_nrows is None and cache_path.exists():
        with utils.Timer("Load Articles (Cache)"):
            df = pd.read_parquet(cache_path)
            return df

    # 2. Load Raw Data
    with utils.Timer("Load Articles (Raw)"):
        # read_csv supports nrows, which is efficient for debugging
        df = pd.read_csv(config.ARTICLES_CSV_PATH, nrows=debug_nrows)

    # 3. Process Data
    with utils.Timer("Process Articles"):
        # Optimize memory usage
        df = utils.reduce_mem_usage(df, verbose=True)

        # Ensure article_id is treated as int32/64 but usually int64 in raw
        # reduce_mem_usage might have downcast it, which is fine as long as precision holds

    # 4. Save Cache (Only if processing full dataset)
    if debug_nrows is None:
        with utils.Timer("Save Articles Cache"):
            # Ensure directory exists
            config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)

    return df


def load_customers(
    load_cached_data: bool = True, debug_nrows: Optional[int] = None
) -> pd.DataFrame:
    """
    Loads the customers dataset.

    Args:
        load_cached_data: If True, attempts to load from a processed parquet cache.
        debug_nrows: If set, returns only the first N rows. Cache is not saved if this is set.

    Returns:
        pd.DataFrame: The customers data.
    """
    cache_path = config.WORKING_DIR / "customers_processed.parquet"

    # 1. Try Loading Cache
    if load_cached_data and debug_nrows is None and cache_path.exists():
        with utils.Timer("Load Customers (Cache)"):
            df = pd.read_parquet(cache_path)
            return df

    # 2. Load Raw Data
    with utils.Timer("Load Customers (Raw)"):
        df = pd.read_csv(config.CUSTOMERS_CSV_PATH, nrows=debug_nrows)

    # 3. Process Data
    with utils.Timer("Process Customers"):
        df = utils.reduce_mem_usage(df, verbose=True)

    # 4. Save Cache (Only if processing full dataset)
    if debug_nrows is None:
        with utils.Timer("Save Customers Cache"):
            config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)

    return df


def load_transactions(
    split: str = "train",
    load_cached_data: bool = True,
    debug_nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads the transaction dataset (train or val).

    Args:
        split: 'train' or 'val'.
        load_cached_data: If True, attempts to load from a processed parquet cache.
        debug_nrows: If set, slices the dataframe after loading. Cache is not saved if this is set.

    Returns:
        pd.DataFrame: The transaction data.
    """
    if split not in ["train", "val"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    cache_path = config.WORKING_DIR / f"transactions_{split}_processed.parquet"

    # 1. Try Loading Cache
    if load_cached_data and debug_nrows is None and cache_path.exists():
        with utils.Timer(f"Load Transactions {split} (Cache)"):
            df = pd.read_parquet(cache_path)
            return df

    # 2. Load Metadata (Raw for this stage)
    source_path = config.TRAIN_DATA_PATH if split == "train" else config.VAL_DATA_PATH

    with utils.Timer(f"Load Transactions {split} (Metadata)"):
        # Metadata is already parquet, but t_dat is string
        df = pd.read_parquet(source_path)

        if debug_nrows is not None:
            df = df.head(debug_nrows)

    # 3. Process Data
    with utils.Timer(f"Process Transactions {split}"):
        # Convert t_dat to datetime
        if "t_dat" in df.columns:
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        # Optimize memory
        df = utils.reduce_mem_usage(df, verbose=True)

    # 4. Save Cache (Only if processing full dataset)
    if debug_nrows is None:
        with utils.Timer(f"Save Transactions {split} Cache"):
            config.WORKING_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)

    return df


def load_sample_submission(debug_nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads the sample submission (test customers).

    Args:
        debug_nrows: If set, returns only the first N rows.

    Returns:
        pd.DataFrame: The sample submission data.
    """
    with utils.Timer("Load Sample Submission"):
        df = pd.read_parquet(config.TEST_DATA_PATH)

        if debug_nrows is not None:
            df = df.head(debug_nrows)

    return df
