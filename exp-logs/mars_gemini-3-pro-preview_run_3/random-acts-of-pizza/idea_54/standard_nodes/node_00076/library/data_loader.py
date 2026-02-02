import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from library.config import METADATA_DIR, TARGET_COL, SEED, N_FOLDS


def load_raw_dataset(split: str = "full_train"):
    """
    Loads the raw dataset from the metadata directory.

    Args:
        split (str): One of 'train', 'val', 'test', or 'full_train'.
                     'full_train' combines train and val for cross-validation.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    if split not in ["train", "val", "test", "full_train"]:
        raise ValueError(
            f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'full_train'."
        )

    # Helper to load specific parquet file
    def _load_parquet(name):
        path = os.path.join(METADATA_DIR, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return pd.read_parquet(path)

    if split == "test":
        return _load_parquet("test")

    if split == "train":
        return _load_parquet("train")

    if split == "val":
        return _load_parquet("val")

    if split == "full_train":
        df_train = _load_parquet("train")
        df_val = _load_parquet("val")
        # Concatenate train and val for full cross-validation
        df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)
        return df_full


def clean_dataset(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Performs basic cleaning and strictly removes leakage features.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): Whether this is the test set (target column might be missing).

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()

    # 1. Leakage Prevention: Drop all columns suffixed with '_at_retrieval'
    # These features are collected at the time the dataset was scraped,
    # which is after the request outcome is known.
    retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
    if retrieval_cols:
        df.drop(columns=retrieval_cols, inplace=True)

    # 2. Target Processing
    if not is_test and TARGET_COL in df.columns:
        # Ensure target is integer (0 or 1)
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 3. Type Consistency
    # Ensure boolean columns are treated consistently if necessary,
    # though most models handle int representations better.
    bool_cols = df.select_dtypes(include=["bool"]).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df


def get_stratified_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED):
    """
    Generates stratified K-Fold indices.

    Args:
        df (pd.DataFrame): DataFrame containing the target variable.
        n_folds (int): Number of folds.
        seed (int): Random seed.

    Returns:
        list: A list of tuples (train_index, val_index).
    """
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in DataFrame for stratification."
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = df[TARGET_COL]

    # Return list of folds
    return list(skf.split(df, y))
