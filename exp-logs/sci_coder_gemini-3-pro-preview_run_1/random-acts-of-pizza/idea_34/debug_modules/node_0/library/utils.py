import os
import random
import numpy as np
import torch
import pandas as pd
import ast
from library.config import Config


def set_seed(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_STATE.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(split="train", load_cached_data=True):
    """
    Loads the dataset for a specific split (train, val, or test).
    Handles parsing of list columns and caching to Parquet for performance.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache first.

    Returns:
        pd.DataFrame: The loaded and parsed dataframe.
    """
    # Determine file paths based on split
    if split == "train":
        source_path = Config.TRAIN_PATH
    elif split == "val":
        source_path = Config.VAL_PATH
    elif split == "test":
        source_path = Config.TEST_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    # Define cache path
    cache_filename = f"{split}_parsed.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If load fails, fall back to processing from scratch
            pass

    # Load from source CSV
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)

    # Parse list columns
    # The subreddit column is stored as a string representation of a list in the CSV
    if Config.SUBREDDIT_COL in df.columns:
        # Use ast.literal_eval to safely parse stringified lists
        # Handle potential NaNs by converting them to empty lists
        df[Config.SUBREDDIT_COL] = (
            df[Config.SUBREDDIT_COL]
            .fillna("[]")
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        )

    # Save to cache for future use
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache for {split}: {e}")

    return df


def get_common_columns(train_df, test_df):
    """
    Identifies feature columns common to both train and test dataframes,
    excluding targets and leakage columns.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.

    Returns:
        list: A list of column names safe for use as features.
    """
    # 1. Identify intersection
    common_cols = set(train_df.columns).intersection(set(test_df.columns))

    # 2. Exclude Target
    if Config.TARGET_COL in common_cols:
        common_cols.remove(Config.TARGET_COL)

    # 3. Exclude Leakage and Artifacts
    for col in Config.LEAKAGE_COLUMNS:
        if col in common_cols:
            common_cols.remove(col)

    # Sort for deterministic order
    return sorted(list(common_cols))


def save_submission(request_ids, probabilities, filename="submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        request_ids (array-like): List or array of request IDs.
        probabilities (array-like): List or array of predicted probabilities.
        filename (str): Name of the output file.
    """
    submission_df = pd.DataFrame(
        {"request_id": request_ids, Config.TARGET_COL: probabilities}
    )

    output_path = os.path.join(Config.SUBMISSION_DIR, filename)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
