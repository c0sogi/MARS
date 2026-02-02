import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import logging
from library.config import PathConfig, TrainConfig
from library.features import create_features
from library.utils import setup_logging, save_dataframe, load_dataframe

# Initialize logging
setup_logging()


def apply_temporal_smoothing(df, sigma=1.0):
    """
    Applies Gaussian smoothing to the binary 'contact' labels along the time dimension
    for each unique player pair (or player-ground pair) within a play.

    This creates 'soft targets' that represent the approach dynamics and helps
    mitigate timestamp noise in the labels.

    Args:
        df (pd.DataFrame): DataFrame containing 'game_play', 'step', 'nfl_player_id_1',
                           'nfl_player_id_2', and 'contact'.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        pd.DataFrame: DataFrame with the 'contact' column replaced by smoothed probabilities.
    """
    logging.info(f"Applying Temporal Label Smoothing (sigma={sigma})...")

    # Ensure data is sorted by step for correct time-series smoothing
    # We sort by the grouping keys + step
    sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # Define the grouping key
    # We group by game_play and the player pair
    group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

    # Function to apply to each group's contact series
    def smooth_group(x):
        # x is the Series of 'contact' labels for one pair over time
        # mode='nearest' handles boundaries by replicating the edge value
        return gaussian_filter1d(x.astype(float), sigma=sigma, mode="nearest")

    # Apply transformation
    # Using transform is generally faster than apply for maintaining shape
    df["contact"] = df.groupby(group_cols)["contact"].transform(smooth_group)

    return df


def prepare_training_data(load_cached=True):
    """
    Prepares the training dataset.

    Steps:
    1. Check for cached smoothed training data.
    2. If not found, generate raw features (via FeatureEngineer).
    3. Apply Temporal Label Smoothing.
    4. Cache the result.

    Args:
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed training DataFrame with soft targets.
    """
    # Define a specific cache path for the smoothed training data
    # The base features are cached by FeatureEngineer, but we want to cache the smoothed version too.
    smoothed_cache_path = os.path.join(
        PathConfig.WORKING_DIR, "train_data_smoothed.parquet"
    )

    # 1. Try Loading Cache
    if load_cached and os.path.exists(smoothed_cache_path):
        logging.info(
            f"Loading cached smoothed training data from {smoothed_cache_path}"
        )
        return load_dataframe(smoothed_cache_path)

    # 2. Generate Base Features
    # This delegates to library.features.FeatureEngineer, which handles:
    # - Loading Metadata/Tracking
    # - Relaxed Quadratic Gating
    # - Vector-Aligned Feature Generation
    # - Caching of the base feature set
    logging.info("Generating/Loading base training features...")
    df_train = create_features(mode="train", load_cached=load_cached)

    # 3. Apply Temporal Label Smoothing
    # Only apply if 'contact' column exists (it should for train)
    if "contact" in df_train.columns:
        df_train = apply_temporal_smoothing(
            df_train, sigma=TrainConfig.LABEL_SMOOTHING_SIGMA
        )
    else:
        logging.warning("Contact column missing in training data. Skipping smoothing.")

    # 4. Save Cache
    save_dataframe(df_train, smoothed_cache_path)

    return df_train


def prepare_validation_data(load_cached=True):
    """
    Prepares the validation dataset.

    For validation, we typically do NOT smooth the labels because we want to measure
    performance (MCC) against the ground truth binary labels.

    Args:
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed validation DataFrame.
    """
    logging.info("Preparing validation data...")
    # FeatureEngineer handles caching of the validation feature set internally
    df_val = create_features(mode="val", load_cached=load_cached)
    return df_val


def prepare_test_data(load_cached=True):
    """
    Prepares the test dataset for inference.

    Args:
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed test DataFrame.
    """
    logging.info("Preparing test data...")
    # FeatureEngineer handles caching of the test feature set internally
    df_test = create_features(mode="test", load_cached=load_cached)
    return df_test


def get_data_split(split_name, load_cached=True):
    """
    Unified accessor for data splits.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached (bool): Whether to use caching.

    Returns:
        pd.DataFrame: The requested dataset.
    """
    if split_name == "train":
        return prepare_training_data(load_cached=load_cached)
    elif split_name == "val":
        return prepare_validation_data(load_cached=load_cached)
    elif split_name == "test":
        return prepare_test_data(load_cached=load_cached)
    else:
        raise ValueError(f"Unknown split name: {split_name}")
