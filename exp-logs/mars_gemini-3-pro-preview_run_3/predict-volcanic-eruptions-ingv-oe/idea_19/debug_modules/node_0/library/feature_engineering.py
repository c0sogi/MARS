import os
import pandas as pd
from library.config import process_dataset, METADATA_DIR
from library.utils import reduce_mem_usage


def process_train_data(
    load_cached_data: bool = True, debug_size: int = None
) -> pd.DataFrame:
    """
    Generates or loads features for the training set using the Dense-Quantile Wavelet strategy.

    Args:
        load_cached_data (bool): Whether to load from parquet cache if available.
        debug_size (int): If set, processes only a subset of the data for debugging.

    Returns:
        pd.DataFrame: Optimized DataFrame containing features and target 'time_to_eruption'.
    """
    metadata_path = os.path.join(METADATA_DIR, "train.csv")
    df = process_dataset(
        metadata_path=metadata_path,
        output_filename="train_features.parquet",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
    return reduce_mem_usage(df, verbose=False)


def process_val_data(
    load_cached_data: bool = True, debug_size: int = None
) -> pd.DataFrame:
    """
    Generates or loads features for the validation set using the Dense-Quantile Wavelet strategy.

    Args:
        load_cached_data (bool): Whether to load from parquet cache if available.
        debug_size (int): If set, processes only a subset of the data for debugging.

    Returns:
        pd.DataFrame: Optimized DataFrame containing features and target 'time_to_eruption'.
    """
    metadata_path = os.path.join(METADATA_DIR, "val.csv")
    df = process_dataset(
        metadata_path=metadata_path,
        output_filename="val_features.parquet",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
    return reduce_mem_usage(df, verbose=False)


def process_test_data(
    load_cached_data: bool = True, debug_size: int = None
) -> pd.DataFrame:
    """
    Generates or loads features for the test set using the Dense-Quantile Wavelet strategy.

    Args:
        load_cached_data (bool): Whether to load from parquet cache if available.
        debug_size (int): If set, processes only a subset of the data for debugging.

    Returns:
        pd.DataFrame: Optimized DataFrame containing features (no target).
    """
    metadata_path = os.path.join(METADATA_DIR, "test.csv")
    df = process_dataset(
        metadata_path=metadata_path,
        output_filename="test_features.parquet",
        load_cached_data=load_cached_data,
        debug_size=debug_size,
    )
    return reduce_mem_usage(df, verbose=False)
