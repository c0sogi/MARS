import pandas as pd
from library.config import Config
from library.utils import load_data as _load_single_dataset


def load_datasets(debug: bool = Config.DEBUG, load_cached_data: bool = True):
    """
    Loads the Train, Validation, and Test datasets for the Insult Detection task.

    This function utilizes the utility function from library.utils to handle
    decoding of unicode-escaped text and caching of processed dataframes.

    Args:
        debug (bool): If True, limits the number of samples loaded (defined in Config).
                      Useful for rapid prototyping and debugging.
        load_cached_data (bool): If True, attempts to load from Parquet cache.
                                 If False or cache missing, re-processes from raw CSVs.

    Returns:
        tuple: A tuple containing three pandas DataFrames: (train_df, val_df, test_df).
    """
    # Determine max samples based on debug flag
    max_samples = Config.MAX_TRAIN_SAMPLES if debug else None

    # Load Train Data
    # The utility function handles reading from metadata, decoding text, and caching.
    train_df = _load_single_dataset(
        "train", load_cached_data=load_cached_data, max_samples=max_samples
    )

    # Load Validation Data
    val_df = _load_single_dataset(
        "val", load_cached_data=load_cached_data, max_samples=max_samples
    )

    # Load Test Data
    test_df = _load_single_dataset(
        "test", load_cached_data=load_cached_data, max_samples=max_samples
    )

    # Final consistency check for text columns
    # Ensure no NaN values exist in the text column and force string type
    # This acts as a safety net for downstream feature extraction.
    text_col = Config.TEXT_COL

    if train_df is not None and text_col in train_df.columns:
        train_df[text_col] = train_df[text_col].fillna("").astype(str)

    if val_df is not None and text_col in val_df.columns:
        val_df[text_col] = val_df[text_col].fillna("").astype(str)

    if test_df is not None and text_col in test_df.columns:
        test_df[text_col] = test_df[text_col].fillna("").astype(str)

    return train_df, val_df, test_df
