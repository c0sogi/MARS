import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("feature_engineering")


def _compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper to compute raw scalar features from a dataframe containing
    'response_a' and 'response_b' columns.

    Args:
        df (pd.DataFrame): Input dataframe with response text columns.

    Returns:
        pd.DataFrame: Dataframe containing only the computed scalar features.
    """
    # Ensure text columns are strings and handle NaNs
    resp_a = df["response_a"].fillna("").astype(str)
    resp_b = df["response_b"].fillna("").astype(str)

    # Pre-compute basic stats to avoid redundant calculations
    # 1. Character Length
    len_char_a = resp_a.str.len()
    len_char_b = resp_b.str.len()

    # 2. Word Count (simple whitespace split)
    # Using numpy for potentially faster execution on list comprehension if needed,
    # but pandas str.split().str.len() is readable and sufficient for this scale.
    len_word_a = resp_a.apply(lambda x: len(x.split()))
    len_word_b = resp_b.apply(lambda x: len(x.split()))

    # 3. Newline Count
    newline_a = resp_a.apply(lambda x: x.count("\n"))
    newline_b = resp_b.apply(lambda x: x.count("\n"))

    # Compute Interaction Features
    features = pd.DataFrame(index=df.index)

    # Differences (A - B)
    features["len_diff_char"] = len_char_a - len_char_b
    features["len_diff_word"] = len_word_a - len_word_b
    features["newline_diff"] = newline_a - newline_b

    # Ratios (A / B)
    # Add epsilon=1 to avoid division by zero and handle empty responses gracefully
    features["len_ratio_char"] = (len_char_a + 1) / (len_char_b + 1)
    features["len_ratio_word"] = (len_word_a + 1) / (len_word_b + 1)
    features["newline_ratio"] = (newline_a + 1) / (newline_b + 1)

    # Ensure we only return columns defined in Config
    # This acts as a filter and an order enforcer
    missing_cols = [c for c in Config.scalar_feature_cols if c not in features.columns]
    if missing_cols:
        raise ValueError(f"Failed to compute required columns: {missing_cols}")

    return features[Config.scalar_feature_cols]


def prepare_scalar_features(load_cached_data: bool = True):
    """
    Main entry point to generate, normalize, and cache scalar features for all splits.

    Logic:
    1. Checks if cached parquet files exist.
    2. If valid cache found and load_cached_data=True, loads and returns them.
    3. Otherwise:
       - Loads metadata for Train, Val, Test.
       - Computes raw features.
       - Fits StandardScaler on TRAIN only.
       - Transforms Train, Val, and Test.
       - Saves to cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from disk.

    Returns:
        tuple: (train_features_df, val_features_df, test_features_df)
    """
    # Define cache paths from Config
    train_cache = Config.train_features_path
    val_cache = Config.val_features_path
    test_cache = Config.test_features_path

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    # 1. Try Loading Cache
    if load_cached_data and cache_exists:
        logger.info("Loading scalar features from cache...")
        try:
            train_feats = pd.read_parquet(train_cache)
            val_feats = pd.read_parquet(val_cache)
            test_feats = pd.read_parquet(test_cache)
            logger.info("Successfully loaded features from cache.")
            return train_feats, val_feats, test_feats
        except Exception as e:
            logger.warning(f"Failed to load cache ({e}). Recomputing features...")

    # 2. Compute from Scratch
    logger.info("Computing scalar features from scratch...")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Load Metadata
    logger.info(
        f"Loading metadata from {Config.train_path}, {Config.val_path}, {Config.test_path}"
    )
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # Debug mode: subset data if configured
    if hasattr(Config, "debug") and Config.debug:
        logger.info("Debug mode enabled: Subsetting data.")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # Extract Raw Features
    logger.info("Extracting raw structural features...")
    raw_train = _compute_raw_features(df_train)
    raw_val = _compute_raw_features(df_val)
    raw_test = _compute_raw_features(df_test)

    # Normalize Features
    # Critical: Fit scaler ONLY on training data to prevent leakage
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(raw_train)

    logger.info("Transforming all splits...")
    # Transform returns numpy array, convert back to DataFrame to preserve column names
    train_feats = pd.DataFrame(
        scaler.transform(raw_train), columns=raw_train.columns, index=raw_train.index
    )
    val_feats = pd.DataFrame(
        scaler.transform(raw_val), columns=raw_val.columns, index=raw_val.index
    )
    test_feats = pd.DataFrame(
        scaler.transform(raw_test), columns=raw_test.columns, index=raw_test.index
    )

    # Save to Cache
    logger.info(f"Saving features to {Config.working_dir}...")
    train_feats.to_parquet(train_cache)
    val_feats.to_parquet(val_cache)
    test_feats.to_parquet(test_cache)

    logger.info("Feature engineering complete.")
    return train_feats, val_feats, test_feats
