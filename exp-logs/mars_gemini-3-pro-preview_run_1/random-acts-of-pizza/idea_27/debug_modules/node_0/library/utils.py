import os
import random
import ast
import numpy as np
import pandas as pd
import torch
from library import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the value in config.py.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed for reproducible dictionary iterations
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_list_column(series: pd.Series) -> list:
    """
    Parses a pandas Series containing stringified lists (e.g., "['a', 'b']")
    back into actual Python lists. Handles NaNs by returning empty lists.

    Args:
        series (pd.Series): Series containing string representations of lists.

    Returns:
        list: A list of Python lists.
    """
    parsed_data = []
    for item in series:
        if pd.isna(item):
            parsed_data.append([])
        elif isinstance(item, str):
            try:
                # Safely evaluate the string literal
                parsed_data.append(ast.literal_eval(item))
            except (ValueError, SyntaxError):
                # In case of malformed strings, return empty list
                parsed_data.append([])
        elif isinstance(item, list):
            parsed_data.append(item)
        else:
            # Fallback for unexpected types
            parsed_data.append([])
    return parsed_data


def arcsinh_scale(data: np.ndarray) -> np.ndarray:
    """
    Applies the inverse hyperbolic sine transformation to the input data.
    This transformation is effective for scaling features with heavy tails
    or zero-inflation, similar to log but handles zeros and negative values.

    Args:
        data (np.ndarray): Input numerical data (array-like).

    Returns:
        np.ndarray: Transformed data.
    """
    # Convert to numpy array if not already
    data_arr = np.array(data)
    return np.arcsinh(data_arr)


def load_dataset(path: str, list_columns: list = None) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file and optionally parses specified columns
    as lists.

    Args:
        path (str): Path to the CSV file.
        list_columns (list, optional): List of column names to parse as lists.

    Returns:
        pd.DataFrame: The loaded and processed dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    df = pd.read_csv(path)

    if list_columns:
        for col in list_columns:
            if col in df.columns:
                df[col] = parse_list_column(df[col])

    return df
