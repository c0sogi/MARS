import os
import pandas as pd
import numpy as np
import ast
import random
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(return_val=True, parse_list_cols=None, debug_size=None):
    """
    Loads train, validation, and test datasets from the paths defined in config.

    Args:
        return_val (bool): Whether to return the validation set.
        parse_list_cols (list of str): List of column names containing stringified lists
                                       to be parsed back into Python lists.
        debug_size (int, optional): If provided, subsamples the datasets to this size
                                    for debugging purposes.

    Returns:
        tuple: (train_df, val_df, test_df) if return_val is True,
               else (train_df, test_df).
    """
    # Load CSVs from metadata directory
    if not os.path.exists(config.TRAIN_PATH):
        raise FileNotFoundError(f"Training data not found at {config.TRAIN_PATH}")

    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    val_df = None
    if return_val:
        if not os.path.exists(config.VAL_PATH):
            raise FileNotFoundError(f"Validation data not found at {config.VAL_PATH}")
        val_df = pd.read_csv(config.VAL_PATH)

    # Subsample for debugging if requested
    if debug_size is not None:
        train_df = train_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]
        if val_df is not None:
            val_df = val_df.iloc[:debug_size]

    # Parse stringified list columns
    if parse_list_cols:
        for col in parse_list_cols:
            # Helper to safely evaluate string lists
            def safe_eval(x):
                if isinstance(x, str):
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        return []
                return x

            if col in train_df.columns:
                train_df[col] = train_df[col].apply(safe_eval)
            if col in test_df.columns:
                test_df[col] = test_df[col].apply(safe_eval)
            if val_df is not None and col in val_df.columns:
                val_df[col] = val_df[col].apply(safe_eval)

    if return_val:
        return train_df, val_df, test_df
    return train_df, test_df


def get_feature_intersection(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test dataframes
    to prevent feature leakage, excluding specified columns.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        exclude_cols (list of str, optional): Columns to strictly exclude
                                              (e.g., target, IDs).

    Returns:
        list: Sorted list of common column names.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    common_cols = train_cols.intersection(test_cols)

    if exclude_cols:
        common_cols = common_cols - set(exclude_cols)

    return sorted(list(common_cols))


def arcsinh_transform(df, columns=None):
    """
    Applies the arcsinh transformation to specified columns in a dataframe
    or to the entire array if input is numpy.

    Args:
        df (pd.DataFrame or np.ndarray): Input data.
        columns (list of str, optional): List of columns to transform if input is DataFrame.
                                         If None and input is DataFrame, transforms all
                                         numeric columns.

    Returns:
        pd.DataFrame or np.ndarray: Transformed data.
    """
    if isinstance(df, pd.DataFrame):
        df_transformed = df.copy()
        if columns is None:
            columns = df_transformed.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in df_transformed.columns:
                df_transformed[col] = np.arcsinh(df_transformed[col])
        return df_transformed
    else:
        # Assume numpy array
        return np.arcsinh(df)
