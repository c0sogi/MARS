import os
import pandas as pd
import numpy as np
from library.config import Config
from library.feature_extraction import generate_features


def load_raw_segment(file_path):
    """
    Loads a raw sensor segment from a CSV file and performs mean imputation.

    Args:
        file_path (str): Relative path to the CSV file (e.g., "train/1000015382.csv").

    Returns:
        pd.DataFrame: The loaded and imputed sensor data, or None if file not found.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        return None

    # Load data with float32 to handle potential NaNs and optimize memory
    df = pd.read_csv(full_path, dtype="float32")

    # Imputation: Fill NaNs with column mean to preserve DC offsets
    # If a column is entirely NaN, the mean is NaN, so we fill with 0 subsequently.
    df = df.fillna(df.mean())
    df = df.fillna(0)

    return df


def build_feature_dataset(load_cached_data=True, debug=False):
    """
    Constructs the feature datasets for Train, Validation, and Test sets.

    This function utilizes the `library.feature_extraction` module to:
    1. Load metadata.
    2. Check for existing cached parquet files.
    3. If not cached, execute parallel feature extraction using the High-Resolution Hybrid-Transform pipeline.
    4. Save generated features to cache.
    5. Return consolidated DataFrames.

    Args:
        load_cached_data (bool): If True, attempts to load features from parquet cache.
                                 If False, forces regeneration of features.
        debug (bool): If True, processes only a small subset of the data (defined in Config).

    Returns:
        tuple: (train_df, val_df, test_df)
            - train_df (pd.DataFrame): Training features and target.
            - val_df (pd.DataFrame): Validation features and target.
            - test_df (pd.DataFrame): Test features (no target).
    """
    # Delegate the heavy lifting to the provided library function which handles
    # parallelization, caching, and the specific signal processing pipeline.
    train_df, val_df, test_df = generate_features(
        load_cached_data=load_cached_data, debug=debug
    )

    return train_df, val_df, test_df
