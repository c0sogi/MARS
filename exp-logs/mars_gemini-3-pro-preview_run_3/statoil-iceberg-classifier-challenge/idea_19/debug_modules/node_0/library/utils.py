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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def impute_inc_angle(df, train_median=None):
    """
    Imputes missing incidence angle values in the dataframe.

    Converts the 'inc_angle' column to numeric, coercing errors (like 'na') to NaN,
    and then fills NaNs with the provided training median. If no median is provided,
    it calculates it from the current dataframe (assumed to be the training set).

    Args:
        df (pd.DataFrame): The dataframe containing the 'inc_angle' column.
        train_median (float, optional): The median value from the training set to use for imputation.
                                        If None, the median is calculated from the input df.

    Returns:
        pd.DataFrame: The dataframe with imputed 'inc_angle' values.
        float: The median value used for imputation.
    """
    # Work on a copy to avoid modifying the original dataframe in place
    df_imputed = df.copy()

    # Ensure inc_angle is numeric, coercing 'na' strings to NaN
    df_imputed["inc_angle"] = pd.to_numeric(df_imputed["inc_angle"], errors="coerce")

    # Calculate median if not provided
    if train_median is None:
        train_median = df_imputed["inc_angle"].median()

    # Fill missing values
    df_imputed["inc_angle"] = df_imputed["inc_angle"].fillna(train_median)

    return df_imputed, train_median
