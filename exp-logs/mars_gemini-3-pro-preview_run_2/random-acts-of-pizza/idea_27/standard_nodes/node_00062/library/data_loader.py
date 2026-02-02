import os
import pandas as pd
import numpy as np
import library.config as config
import library.utils as utils


def prepare_text_data(df):
    """
    Concatenates request title and body into a single text column.

    Args:
        df (pd.DataFrame): The dataframe containing raw text columns.

    Returns:
        pd.Series: A series containing the concatenated text.
    """
    # Ensure we use the columns defined in config
    title_col = config.TEXT_COLS[0]  # request_title
    body_col = config.TEXT_COLS[1]  # request_text_edit_aware

    # Fill NaNs with empty string to allow concatenation
    titles = df[title_col].fillna("").astype(str)
    bodies = df[body_col].fillna("").astype(str)

    # Concatenate with a space separator
    combined_text = titles + " " + bodies
    return combined_text


def extract_numerical_metadata(df):
    """
    Extracts the specific numerical columns defined in the configuration.

    Args:
        df (pd.DataFrame): The dataframe containing all features.

    Returns:
        pd.DataFrame: A dataframe containing only the selected numerical features.
    """
    # Select columns defined in config.NUMERICAL_COLS
    # We use intersection to avoid KeyErrors if a column is missing (though it shouldn't be)
    available_cols = [c for c in config.NUMERICAL_COLS if c in df.columns]

    if len(available_cols) != len(config.NUMERICAL_COLS):
        missing = set(config.NUMERICAL_COLS) - set(available_cols)
        print(f"Warning: The following numerical columns were not found: {missing}")

    return df[available_cols].copy()


def get_processed_data(split, debug_size=None, load_cached_data=True):
    """
    Loads the dataset, processes text and numerical features, and returns a clean DataFrame.
    Implements caching to parquet to speed up subsequent runs.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug_size (int, optional): Number of samples to load for debugging.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe with 'text_combined', numerical cols, and target.
    """
    # Construct cache filename
    debug_suffix = f"_debug_{debug_size}" if debug_size is not None else ""
    cache_filename = f"{split}_processed{debug_suffix}.parquet"
    cache_path = os.path.join(config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading processed {split} data from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from scratch...")

    # Load raw data using library utility
    df_raw = utils.load_dataset(split, debug_size=debug_size)

    # Prepare features
    df_processed = pd.DataFrame()

    # Identity
    if "request_id" in df_raw.columns:
        df_processed["request_id"] = df_raw["request_id"]

    # Target (if available)
    if config.TARGET_COL in df_raw.columns:
        df_processed[config.TARGET_COL] = df_raw[config.TARGET_COL]

    # Text Processing
    df_processed["text_combined"] = prepare_text_data(df_raw)

    # Numerical Processing
    df_num = extract_numerical_metadata(df_raw)
    df_processed = pd.concat([df_processed, df_num], axis=1)

    # 3. Save to cache
    try:
        # Ensure directory exists (redundant if config handles it, but safe)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_processed.to_parquet(cache_path, index=False)
        print(f"Saved processed {split} data to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return df_processed
