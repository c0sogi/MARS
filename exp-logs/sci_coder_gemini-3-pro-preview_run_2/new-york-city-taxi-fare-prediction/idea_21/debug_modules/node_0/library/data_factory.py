import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    STRICT_FILTER,
    LOOSE_FILTER,
    LEARNER_SUBSET_SIZE,
    SEED,
)
from library.geo_utils import clamp_coordinates, compute_haversine


def get_wisdom_set(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Generates the Wisdom Set (Background) used for statistical fingerprints.
    Applies Strict Filtering to the full training data to ensure robust priors.

    Strict Filter:
    - min_fare <= fare <= max_fare
    - fare_per_km <= max_fare_per_km

    Args:
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: The processed Wisdom Set.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "cached_wisdom_set.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached Wisdom Set from {cache_file}...")
        return pd.read_parquet(cache_file)

    print("Generating Wisdom Set from scratch...")
    # Load full training data
    try:
        df = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_parquet(TRAIN_PATH)

    # 1. Clamp Coordinates (Sanitization)
    df = clamp_coordinates(df)

    # 2. Compute Distance for Strict Filtering
    # We need Haversine distance to calculate fare_per_km
    dists = compute_haversine(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    # 3. Calculate Fare Per Km
    # Handle division by zero: if dist is 0, result is inf.
    # Inf will be filtered out by the <= max_fare_per_km check.
    with np.errstate(divide="ignore", invalid="ignore"):
        fare_per_km = df["fare_amount"].values / dists

    # 4. Apply Strict Filter
    mask = (
        (df["fare_amount"] >= STRICT_FILTER["min_fare"])
        & (df["fare_amount"] <= STRICT_FILTER["max_fare"])
        & (fare_per_km <= STRICT_FILTER["max_fare_per_km"])
    )

    df_wisdom = df.loc[mask].reset_index(drop=True)

    print(f"Wisdom Set created: {len(df_wisdom)} rows (Original: {len(df)})")

    # Save to cache
    print(f"Saving Wisdom Set to {cache_file}...")
    df_wisdom.to_parquet(cache_file)

    return df_wisdom


def get_learner_set(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Generates the Learner Set (Foreground) used for training the model.
    Applies Loose Filtering to retain valid high-fare outliers (Heavy Tail)
    and subsamples to a stable size.

    Loose Filter:
    - min_fare <= fare

    Args:
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: The processed Learner Set.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "cached_learner_set.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached Learner Set from {cache_file}...")
        return pd.read_parquet(cache_file)

    print("Generating Learner Set from scratch...")
    try:
        df = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_parquet(TRAIN_PATH)

    # 1. Clamp Coordinates
    df = clamp_coordinates(df)

    # 2. Apply Loose Filter
    mask = df["fare_amount"] >= LOOSE_FILTER["min_fare"]
    df_learner = df.loc[mask]

    # 3. Subsample
    if len(df_learner) > LEARNER_SUBSET_SIZE:
        print(
            f"Subsampling Learner Set from {len(df_learner)} to {LEARNER_SUBSET_SIZE}..."
        )
        df_learner = df_learner.sample(n=LEARNER_SUBSET_SIZE, random_state=SEED)

    df_learner = df_learner.reset_index(drop=True)

    # Save to cache
    print(f"Saving Learner Set to {cache_file}...")
    df_learner.to_parquet(cache_file)

    return df_learner


def get_val_set(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Generates the Validation Set.
    Applies Loose Filtering (to match Learner domain) and Clamping.

    Args:
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: The processed Validation Set.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "cached_val_set.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached Validation Set from {cache_file}...")
        return pd.read_parquet(cache_file)

    print("Generating Validation Set from scratch...")
    try:
        df = pd.read_parquet(VAL_PATH, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_parquet(VAL_PATH)

    # 1. Clamp Coordinates
    df = clamp_coordinates(df)

    # 2. Apply Loose Filter
    # We remove negative/impossible fares to ensure fair RMSE evaluation
    mask = df["fare_amount"] >= LOOSE_FILTER["min_fare"]
    df_val = df.loc[mask].reset_index(drop=True)

    print(f"Validation Set created: {len(df_val)} rows")

    # Save to cache
    print(f"Saving Validation Set to {cache_file}...")
    df_val.to_parquet(cache_file)

    return df_val


def get_test_set(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Generates the Test Set.
    Applies Clamping only (no target variable to filter).

    Args:
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        pd.DataFrame: The processed Test Set.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "cached_test_set.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached Test Set from {cache_file}...")
        return pd.read_parquet(cache_file)

    print("Generating Test Set from scratch...")
    try:
        df = pd.read_parquet(TEST_PATH, engine="pyarrow")
    except (ImportError, ValueError):
        df = pd.read_parquet(TEST_PATH)

    # 1. Clamp Coordinates
    df = clamp_coordinates(df)

    # Save to cache
    print(f"Saving Test Set to {cache_file}...")
    df.to_parquet(cache_file)

    return df
