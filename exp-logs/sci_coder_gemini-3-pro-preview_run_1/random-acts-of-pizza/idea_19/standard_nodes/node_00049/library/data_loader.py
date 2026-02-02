import pandas as pd
import numpy as np
import ast
import os
from library.config import TRAIN_PATH, VAL_PATH, TEST_PATH


def filter_leakage(train_df, val_df, test_df, target_col="requester_received_pizza"):
    """
    Restricts the columns in the datasets to only those present in both the training
    and test sets (intersection), plus the target column for the training/validation sets.

    This prevents the model from training on features that are not available at inference time
    (e.g., retrieval-time statistics that might exist in train but not test).

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        target_col (str): The name of the target variable.

    Returns:
        tuple: (cleaned_train_df, cleaned_val_df, cleaned_test_df)
    """
    # Identify columns present in both train and test
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # The intersection of features available in both
    common_features = list(train_cols.intersection(test_cols))

    # Ensure specific identifier columns are preserved if they are in the intersection
    # (They should be, but order matters for readability/debugging)
    priority_cols = [
        "request_id",
        "request_title",
        "request_text",
        "request_text_edit_aware",
    ]
    sorted_features = [c for c in priority_cols if c in common_features] + [
        c for c in common_features if c not in priority_cols
    ]

    # Select features for Test
    cleaned_test = test_df[sorted_features].copy()

    # Select features + target for Train/Val
    # We ensure target_col is not duplicated if it happened to be in common_features (unlikely for blind test)
    train_features = [c for c in sorted_features if c != target_col] + [target_col]

    cleaned_train = train_df[train_features].copy()
    cleaned_val = val_df[train_features].copy()

    print(f"Feature Intersection Analysis:")
    print(f"  - Original Train Cols: {len(train_cols)}")
    print(f"  - Original Test Cols:  {len(test_cols)}")
    print(f"  - Common Features:     {len(sorted_features)}")
    print(f"  - Removed from Train:  {len(train_cols) - len(set(train_features))}")

    return cleaned_train, cleaned_val, cleaned_test


def parse_list_columns(df):
    """
    Parses columns that contain stringified lists (e.g. "['a', 'b']") into actual Python lists.
    """
    # The primary list column in this dataset is 'requester_subreddits_at_request'
    list_cols = ["requester_subreddits_at_request"]

    for col in list_cols:
        if col in df.columns:
            # Use ast.literal_eval to safely parse the string representation of the list
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

            # Fill NaNs with empty lists to avoid iteration errors later
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    return df


def load_data(nrows=None):
    """
    Loads the training, validation, and test datasets from the metadata CSVs.
    Performs initial cleaning including list parsing and leakage filtering.

    Args:
        nrows (int, optional): Number of rows to load for debugging/testing.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("Loading data from metadata CSVs...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")

    # Load CSVs
    train_df = pd.read_csv(TRAIN_PATH, nrows=nrows)
    val_df = pd.read_csv(VAL_PATH, nrows=nrows)
    test_df = pd.read_csv(TEST_PATH, nrows=nrows)

    # Parse stringified lists
    print("Parsing list columns...")
    train_df = parse_list_columns(train_df)
    val_df = parse_list_columns(val_df)
    test_df = parse_list_columns(test_df)

    # Filter for leakage (keep only intersection of columns)
    print("Filtering leakage (keeping column intersection)...")
    train_df, val_df, test_df = filter_leakage(train_df, val_df, test_df)

    print(f"Data Loaded Successfully.")
    print(f"  - Train shape: {train_df.shape}")
    print(f"  - Val shape:   {val_df.shape}")
    print(f"  - Test shape:  {test_df.shape}")

    return train_df, val_df, test_df
