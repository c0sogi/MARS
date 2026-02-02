import os
import pandas as pd
import numpy as np
import ast
from library import config


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleaning on the dataframe:
    - Fills missing values for text and categorical columns.
    - Converts boolean target to integer.
    - Parses stringified list columns.
    """
    df = df.copy()

    # Handle missing values in text columns
    text_cols = ["request_text", "request_title", "request_text_edit_aware"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Handle missing values in categorical columns
    if "requester_user_flair" in df.columns:
        df["requester_user_flair"] = df["requester_user_flair"].fillna("None")

    if "giver_username_if_known" in df.columns:
        df["giver_username_if_known"] = df["giver_username_if_known"].fillna("N/A")

    # Convert target to integer if present
    if "requester_received_pizza" in df.columns:
        df["requester_received_pizza"] = df["requester_received_pizza"].astype(int)

    # Parse stringified lists (CSVs save lists as strings like "['a', 'b']")
    list_cols = ["requester_subreddits_at_request"]
    for col in list_cols:
        if col in df.columns:
            # Use ast.literal_eval to safely parse the string representation of lists
            # Handle cases where it might already be a list (if loaded from parquet previously not strictly needed but good for safety)
            # or if it is a string representation.
            def parse_list(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        return []
                return []

            df[col] = df[col].apply(parse_list)

    return df


def load_dataset(split: str = "train", load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads the dataset for the specified split.
    Implements caching using Parquet files.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Determine file paths
    if split == "train":
        input_path = config.TRAIN_PATH
    elif split == "val":
        input_path = config.VAL_PATH
    elif split == "test":
        input_path = config.TEST_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Define cache path
    cache_filename = f"{split}_cleaned.parquet"
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Parquet preserves data types, including lists, so no need to re-parse
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(
                f"Failed to load cache from {cache_path}: {e}. Reloading from source."
            )

    # Load from source CSV
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Apply cleaning
    df = clean_dataframe(df)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df


def get_feature_intersection(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Identifies the intersection of columns between train and test dataframes,
    filtering out specified leakage columns.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.

    Returns:
        list: A list of valid feature column names.
    """
    # Get common columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)

    # Remove leakage columns
    valid_features = [col for col in common_cols if col not in config.LEAKAGE_COLUMNS]

    # Sort for deterministic order
    valid_features.sort()

    return valid_features
