import os
import numpy as np
import pandas as pd
from library.config import Config
from library.features import generate_features


def load_csv_data(debug=False, debug_sample_size=100):
    """
    Loads the metadata CSV files for training, validation, and testing.

    Args:
        debug (bool): If True, samples the datasets for rapid iteration.
        debug_sample_size (int): Number of samples to load in debug mode.

    Returns:
        tuple: (train_meta_df, val_meta_df, test_meta_df)
    """
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )
    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(f"Val metadata not found at {Config.VAL_METADATA_PATH}")
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        print(f"Debug mode enabled: Sampling {debug_sample_size} rows per dataset.")
        train_df = train_df.head(debug_sample_size).copy()
        val_df = val_df.head(debug_sample_size).copy()
        test_df = test_df.head(debug_sample_size).copy()

    return train_df, val_df, test_df


def transform_targets(df, target_cols=None):
    """
    Applies log(1+y) transformation to the specified target columns.

    Args:
        df (pd.DataFrame): DataFrame containing target columns.
        target_cols (list): List of column names to transform.
                            Defaults to Config.TARGET_COLS if None.

    Returns:
        pd.DataFrame: DataFrame with transformed target columns.
    """
    if target_cols is None:
        target_cols = Config.TARGET_COLS

    df_transformed = df.copy()
    for col in target_cols:
        if col in df_transformed.columns:
            # Apply log1p: log(1 + x)
            df_transformed[col] = np.log1p(df_transformed[col])

    return df_transformed


def inverse_transform_targets(y_pred):
    """
    Applies exp(y) - 1 transformation to reverse the log1p operation.

    Args:
        y_pred (np.array or pd.Series): Log-transformed predictions.

    Returns:
        np.array: Predictions in the original scale.
    """
    # Apply expm1: exp(x) - 1
    return np.expm1(y_pred)


def build_dataset(load_cached_data=True, debug=False, debug_sample_size=100):
    """
    Orchestrates the data loading and feature engineering process.

    1. Loads metadata CSVs.
    2. Generates (or loads cached) physics-informed features using library.features.
    3. Merges features with tabular metadata.
    4. Transforms target variables in training and validation sets.

    Args:
        load_cached_data (bool): Whether to attempt loading features from parquet cache.
        debug (bool): If True, uses a small subset of data.
        debug_sample_size (int): Size of subset in debug mode.

    Returns:
        tuple: (train_df, val_df, test_df) containing features and (transformed) targets.
    """
    # 1. Load Metadata
    train_meta, val_meta, test_meta = load_csv_data(debug, debug_sample_size)

    # Define cache paths. If debugging, use a separate cache file to avoid
    # overwriting the full dataset cache with a partial one.
    if debug:
        train_cache = Config.TRAIN_FEATURES_PATH.replace(".parquet", "_debug.parquet")
        val_cache = Config.VAL_FEATURES_PATH.replace(".parquet", "_debug.parquet")
        test_cache = Config.TEST_FEATURES_PATH.replace(".parquet", "_debug.parquet")
    else:
        train_cache = Config.TRAIN_FEATURES_PATH
        val_cache = Config.VAL_FEATURES_PATH
        test_cache = Config.TEST_FEATURES_PATH

    # 2. Generate Features (Geometric, Chemical Disorder, Electrostatic)
    # The generate_features function handles iteration, extraction, and caching.
    print("Building Training Set...")
    train_df = generate_features(
        train_meta, train_cache, load_cached_data=load_cached_data
    )

    print("Building Validation Set...")
    val_df = generate_features(val_meta, val_cache, load_cached_data=load_cached_data)

    print("Building Test Set...")
    test_df = generate_features(
        test_meta, test_cache, load_cached_data=load_cached_data
    )

    # 3. Transform Targets
    # Apply log(1+y) to targets for regression stability and metric alignment
    print("Transforming targets (log1p)...")
    train_df = transform_targets(train_df, Config.TARGET_COLS)
    val_df = transform_targets(val_df, Config.TARGET_COLS)

    # Test set does not have targets, so no transformation needed there.

    return train_df, val_df, test_df
