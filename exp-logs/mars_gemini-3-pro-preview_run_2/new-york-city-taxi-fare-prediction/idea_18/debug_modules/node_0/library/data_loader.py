import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import clamp_coordinates, haversine_distance


def load_and_clean_data(filepath: str, clamp: bool = True) -> pd.DataFrame:
    """
    Loads data from a parquet file and optionally clamps coordinates to the NYC bounding box.

    Args:
        filepath (str): Path to the parquet file.
        clamp (bool): Whether to apply coordinate clamping.

    Returns:
        pd.DataFrame: The loaded and processed dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load data
    df = pd.read_parquet(filepath)

    # Apply clamping if requested
    if clamp:
        df = clamp_coordinates(df)

    return df


def create_dual_sets(df: pd.DataFrame):
    """
    Splits the dataframe into a 'Wisdom Set' for statistics and a 'Learner Set' for training.

    Wisdom Set: Strict filtering to ensure high-quality statistical priors.
    Learner Set: Loose filtering to allow the model to learn heavy tails, with subsampling.

    Args:
        df (pd.DataFrame): The full training dataframe (clamped).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (wisdom_df, learner_df)
    """
    # ---------------------------------------------------------
    # 1. Create Wisdom Set (Strict Filters)
    # ---------------------------------------------------------
    # Fare Amount Filter
    wisdom_mask_fare = (df["fare_amount"] >= Config.WISDOM_MIN_FARE) & (
        df["fare_amount"] <= Config.WISDOM_MAX_FARE
    )

    # Fare per Km Filter (Sanity Check)
    # Calculate distance for filtering logic
    dist_km = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    # Avoid division by zero
    fare_per_km = df["fare_amount"] / (dist_km + 1e-6)
    wisdom_mask_rate = fare_per_km < Config.WISDOM_MAX_FARE_PER_KM

    wisdom_df = df[wisdom_mask_fare & wisdom_mask_rate].copy()

    # ---------------------------------------------------------
    # 2. Create Learner Set (Loose Filters + Subsampling)
    # ---------------------------------------------------------
    # Loose Fare Filter
    learner_mask = df["fare_amount"] >= Config.LEARNER_MIN_FARE
    learner_df = df[learner_mask].copy()

    # Subsampling
    if len(learner_df) > Config.LEARNER_SUBSAMPLE_SIZE:
        learner_df = learner_df.sample(
            n=Config.LEARNER_SUBSAMPLE_SIZE, random_state=Config.RANDOM_SEED
        )

    return wisdom_df, learner_df


def load_training_data(load_cached_data: bool = True):
    """
    Orchestrates the loading, cleaning, and splitting of training data.
    Implements caching to avoid re-processing the large dataset.

    Args:
        load_cached_data (bool): If True, attempts to load from ./working/idea_18/

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (wisdom_df, learner_df)
    """
    # Define cache paths
    cache_wisdom_path = os.path.join(Config.WORKING_DIR, "cached_wisdom_set.parquet")
    cache_learner_path = os.path.join(Config.WORKING_DIR, "cached_learner_set.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(cache_wisdom_path)
        and os.path.exists(cache_learner_path)
    ):
        print(f"Loading cached training data from {Config.WORKING_DIR}...")
        wisdom_df = pd.read_parquet(cache_wisdom_path)
        learner_df = pd.read_parquet(cache_learner_path)
        return wisdom_df, learner_df

    print("Cache not found or ignored. Processing training data from scratch...")

    # Load and Clean
    print(f"Loading raw data from {Config.TRAIN_DATA_PATH}...")
    full_df = load_and_clean_data(Config.TRAIN_DATA_PATH, clamp=True)

    # Split
    print("Creating Dual-Hygiene sets (Wisdom & Learner)...")
    wisdom_df, learner_df = create_dual_sets(full_df)

    # Save to cache
    print(f"Caching processed sets to {Config.WORKING_DIR}...")
    wisdom_df.to_parquet(cache_wisdom_path, index=False)
    learner_df.to_parquet(cache_learner_path, index=False)

    return wisdom_df, learner_df


def load_validation_data(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads and cleans the validation set. Caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        pd.DataFrame: Processed validation set.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cached_val_set.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached validation data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Processing validation data...")
    val_df = load_and_clean_data(Config.VAL_DATA_PATH, clamp=True)

    print(f"Caching validation data to {cache_path}...")
    val_df.to_parquet(cache_path, index=False)

    return val_df


def load_test_data(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads and cleans the test set. Caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        pd.DataFrame: Processed test set.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "cached_test_set.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Processing test data...")
    test_df = load_and_clean_data(Config.TEST_DATA_PATH, clamp=True)

    print(f"Caching test data to {cache_path}...")
    test_df.to_parquet(cache_path, index=False)

    return test_df
