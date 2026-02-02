import os
import pandas as pd
import numpy as np
from library import config
from library import utils


def get_wisdom_mask(df: pd.DataFrame) -> pd.Series:
    """
    Generates a boolean mask for the 'Wisdom' set based on strict criteria.
    Wisdom data is used for generating robust statistical priors.

    Criteria from config:
    - min_fare <= fare <= max_fare
    - fare / distance <= max_fare_per_km
    """
    # 1. Basic Fare Range Check
    mask = (df["fare_amount"] >= config.WISDOM_CRITERIA["min_fare"]) & (
        df["fare_amount"] <= config.WISDOM_CRITERIA["max_fare"]
    )

    # 2. Fare per Km Check
    # We need distance for this. Using Haversine as it's standard for this check.
    dist_km = utils.calculate_haversine(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    # Avoid division by zero: if distance is very small, treat as valid only if fare is small?
    # Or simply use a small epsilon.
    # If distance is 0, fare_per_km is inf.
    # We want to filter out cases where fare is high but distance is zero/low.

    # Calculate fare_per_km safely
    # We can just check: fare <= max_rate * distance
    # But let's stick to the ratio for clarity, handling 0 dist.

    # If distance < 0.001 km (1 meter), and fare > 2.50, it's likely bad data or a cancellation
    # that shouldn't be in Wisdom stats (which represent valid motion).
    # Let's use the explicit ratio check with a safe divisor.
    safe_dist = np.maximum(dist_km, 0.001)
    fare_per_km = df["fare_amount"] / safe_dist

    mask = mask & (fare_per_km <= config.WISDOM_CRITERIA["max_fare_per_km"])

    return mask


def get_learner_mask(df: pd.DataFrame) -> pd.Series:
    """
    Generates a boolean mask for the 'Learner' set based on loose criteria.
    Learner data is used for training the model and should include high-fare outliers.

    Criteria from config:
    - fare >= min_fare
    """
    mask = df["fare_amount"] >= config.LEARNER_CRITERIA["min_fare"]
    return mask


def load_dataset(load_cached_data: bool = True):
    """
    Loads, cleans, filters, and splits the data into Learner, Wisdom, Val, and Test sets.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data: If True, attempts to load pre-processed parquet files from cache.

    Returns:
        learner_df, wisdom_df, val_df, test_df
    """
    # Define Cache Paths
    cache_learner_path = os.path.join(config.CACHE_DIR, "cached_learner_set.parquet")
    cache_wisdom_path = os.path.join(config.CACHE_DIR, "cached_wisdom_set.parquet")
    cache_val_path = os.path.join(config.CACHE_DIR, "cached_val_set.parquet")
    cache_test_path = os.path.join(config.CACHE_DIR, "cached_test_set.parquet")

    # Check if cache exists and is requested
    if load_cached_data:
        if (
            os.path.exists(cache_learner_path)
            and os.path.exists(cache_wisdom_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):

            print("Loading data from cache...")
            learner_df = pd.read_parquet(cache_learner_path)
            wisdom_df = pd.read_parquet(cache_wisdom_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)

            return learner_df, wisdom_df, val_df, test_df
        else:
            print("Cache not found or incomplete. Processing from scratch...")
    else:
        print("Forcing data processing from scratch...")

    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load Raw Metadata
    # ---------------------------------------------------------
    print("Loading raw datasets from metadata...")
    # We use the metadata splits.
    # train.parquet contains the 80% training split.
    # val.parquet contains the 20% validation split.
    train_full = pd.read_parquet(config.TRAIN_PATH)
    val_df = pd.read_parquet(config.VAL_PATH)
    test_df = pd.read_parquet(config.TEST_PATH)

    # ---------------------------------------------------------
    # 2. Input Sanitization (Clamping)
    # ---------------------------------------------------------
    print("Clamping coordinates to NYC bounding box...")
    train_full = utils.clamp_coordinates(train_full)
    val_df = utils.clamp_coordinates(val_df)
    test_df = utils.clamp_coordinates(test_df)

    # ---------------------------------------------------------
    # 3. Create Wisdom Set (Global Priors Source)
    # ---------------------------------------------------------
    print("Creating Wisdom Set (Strict Filtering)...")
    # We use the full training split for wisdom to maximize density.
    # We do NOT use val_df for wisdom to strictly avoid leakage.
    wisdom_mask = get_wisdom_mask(train_full)
    wisdom_df = train_full[wisdom_mask].copy()

    # ---------------------------------------------------------
    # 4. Create Learner Set (Training Data)
    # ---------------------------------------------------------
    print("Creating Learner Set (Loose Filtering & Subsampling)...")
    learner_mask = get_learner_mask(train_full)
    learner_df = train_full[learner_mask].copy()

    # Subsample Learner Set
    # We only subsample the learner set because training on 44M rows is too slow.
    # Wisdom set is kept large for accurate stats.
    if len(learner_df) > config.TRAIN_SUBSAMPLE_SIZE:
        print(
            f"Subsampling Learner Set from {len(learner_df)} to {config.TRAIN_SUBSAMPLE_SIZE}..."
        )
        learner_df = learner_df.sample(
            n=config.TRAIN_SUBSAMPLE_SIZE, random_state=config.SEED
        )

    # ---------------------------------------------------------
    # 5. Save to Cache
    # ---------------------------------------------------------
    print("Saving processed datasets to cache...")
    learner_df.to_parquet(cache_learner_path, index=False)
    wisdom_df.to_parquet(cache_wisdom_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    print(f"Data Processing Complete.")
    print(f"Learner Shape: {learner_df.shape}")
    print(f"Wisdom Shape: {wisdom_df.shape}")
    print(f"Val Shape: {val_df.shape}")
    print(f"Test Shape: {test_df.shape}")

    return learner_df, wisdom_df, val_df, test_df
