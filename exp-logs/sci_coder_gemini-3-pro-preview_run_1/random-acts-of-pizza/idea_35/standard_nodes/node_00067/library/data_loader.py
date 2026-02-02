import os
import pandas as pd
from library.config import Config, ensure_dir, parse_list_col


def get_common_columns(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str = None
) -> list:
    """
    Identifies the intersection of columns between training and test sets to prevent leakage.
    Strictly enforces that only features present in both datasets are used.

    Args:
        df_train (pd.DataFrame): Training dataframe.
        df_test (pd.DataFrame): Test dataframe.
        target_col (str, optional): The name of the target column to exclude from the feature list.

    Returns:
        list: A sorted list of column names common to both dataframes.
    """
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    # Calculate intersection
    common_cols = list(train_cols.intersection(test_cols))

    # Remove target column if specified and present
    if target_col and target_col in common_cols:
        common_cols.remove(target_col)

    return sorted(common_cols)


def load_dataset(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the dataset for a specific split (train, val, test).
    Handles caching using Parquet format for efficiency and strictly parses list columns.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to use cached data if available.

    Returns:
        pd.DataFrame: The loaded and processed dataframe.
    """
    # Validate split argument
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Determine source CSV path based on split
    if split == "train":
        csv_path = Config.TRAIN_PATH
    elif split == "val":
        csv_path = Config.VAL_PATH
    else:
        csv_path = Config.TEST_PATH

    # Define cache path
    ensure_dir(Config.CACHE_DIR)
    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_parsed.parquet")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            # Fallback to loading from source if cache is corrupt
            pass

    # Load from source CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse stringified list columns
    # 'requester_subreddits_at_request' is stored as a string representation of a list in the CSV
    list_cols = ["requester_subreddits_at_request"]
    for col in list_cols:
        if col in df.columns:
            df[col] = parse_list_col(df[col])

    # Clean text columns (Fill NaNs with empty strings)
    text_cols = ["request_text", "request_text_edit_aware", "request_title"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Save processed dataframe to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        # Proceed without caching if write fails
        pass

    return df
