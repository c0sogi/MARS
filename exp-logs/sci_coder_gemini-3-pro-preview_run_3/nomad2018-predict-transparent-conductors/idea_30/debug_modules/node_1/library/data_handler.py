import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TARGET_COLS,
)


def load_metadata(split: str = "train") -> pd.DataFrame:
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata containing material IDs, features, and file paths.
    """
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    df = pd.read_csv(path)
    return df


def transform_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts target columns from the DataFrame and applies log(1+y) transformation.
    This transformation is used to normalize the distribution of energy values
    and improve model convergence.

    Args:
        df (pd.DataFrame): DataFrame containing the target columns.

    Returns:
        pd.DataFrame: Log-transformed target variables.
    """
    # Verify that target columns exist in the dataframe
    missing_cols = [col for col in TARGET_COLS if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Target columns {missing_cols} not found in DataFrame. "
            "Ensure you are passing the training or validation set."
        )

    # Apply log1p transformation: z = log(1 + y)
    # Based on data analysis, minimum values are >= 0, so this is safe.
    y = df[TARGET_COLS].copy()
    y_transformed = np.log1p(y)

    return y_transformed


def inverse_transform_targets(y_pred: np.ndarray) -> np.ndarray:
    """
    Applies the inverse transformation exp(z) - 1 to predictions to return them
    to the original physical scale (eV).

    Args:
        y_pred (np.ndarray): Log-transformed predictions.

    Returns:
        np.ndarray: Predictions in the original scale.
    """
    # Apply expm1 transformation: y = exp(z) - 1
    return np.expm1(y_pred)
