import ast
import os
import numpy as np
import pandas as pd
from library.config import set_seed, Config


def parse_list_string(x):
    """
    Parses a string representation of a list into a numpy array.
    Handles cases where the input is already a list or numpy array.

    Args:
        x (str, list, or np.ndarray): The input value to parse.

    Returns:
        np.ndarray: The parsed numpy array of floats. Returns empty array on failure.
    """
    if isinstance(x, str):
        try:
            # literal_eval is safe for parsing python literals like lists
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except (ValueError, SyntaxError):
            return np.array([], dtype=np.float32)
    elif isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.array([], dtype=np.float32)


def parse_dataframe_columns(df, columns):
    """
    Parses specified columns in a DataFrame from stringified lists to numpy arrays.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to parse.

    Returns:
        pd.DataFrame: The DataFrame with parsed columns.
    """
    df_parsed = df.copy()
    for col in columns:
        if col in df_parsed.columns:
            df_parsed[col] = df_parsed[col].apply(parse_list_string)
    return df_parsed


def get_parsed_metadata(mode="train", data_dir=Config.METADATA_DIR, sample_size=None):
    """
    Loads the metadata CSV for a given mode (train/val/test) and parses target columns.

    Args:
        mode (str): 'train', 'val', or 'test'.
        data_dir (str): Directory containing the CSV files.
        sample_size (int, optional): Number of samples to load. Useful for debugging/testing.

    Returns:
        pd.DataFrame: The loaded and parsed DataFrame.
    """
    filepath = os.path.join(data_dir, f"{mode}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Optional sampling for debugging
    if sample_size is not None and sample_size < len(df):
        # Ensure reproducibility in sampling
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Columns that typically contain stringified lists in this dataset
    # We include both targets and errors
    potential_list_cols = [
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
    ]

    # Only parse columns that actually exist in the dataframe
    cols_to_parse = [c for c in potential_list_cols if c in df.columns]
    df = parse_dataframe_columns(df, cols_to_parse)

    return df
