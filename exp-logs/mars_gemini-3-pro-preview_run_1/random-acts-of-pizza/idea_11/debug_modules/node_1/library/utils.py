import os
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def arcsinh_transform(df, columns):
    """
    Applies the arcsinh transformation (inverse hyperbolic sine) to specified columns
    in a DataFrame. This is useful for normalizing heavy-tailed distributions
    (e.g., vote counts, karma) while handling zero and negative values gracefully,
    unlike log(x) or log(x+1).

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to transform.

    Returns:
        pd.DataFrame: A new DataFrame with transformed columns.
    """
    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns:
            # np.arcsinh is defined for all real numbers
            df_transformed[col] = np.arcsinh(df_transformed[col])
    return df_transformed


def get_common_columns(train_df, test_df, exclude_cols=None):
    """
    Identifies the intersection of columns between train and test DataFrames
    to ensure the model only uses features present in both sets, preventing leakage.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        exclude_cols (list, optional): List of columns to explicitly exclude
                                       from the intersection (e.g., target variables).

    Returns:
        list: A sorted list of column names present in both DataFrames.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    common_cols = train_cols.intersection(test_cols)

    if exclude_cols:
        common_cols = common_cols - set(exclude_cols)

    return sorted(list(common_cols))
