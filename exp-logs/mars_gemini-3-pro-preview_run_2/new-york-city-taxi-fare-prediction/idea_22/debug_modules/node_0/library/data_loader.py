import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import clamp_coordinates, haversine_array


def load_dataset(path):
    """
    Helper function to load a dataset from a parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_parquet(path)


def create_dual_hygiene_sets(load_cached_data=True):
    """
    Generates the Wisdom, Learner, Validation, and Test sets based on the
    Dual-Hygiene strategy. Implements caching to avoid re-processing.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed sets
                                 from the working directory.

    Returns:
        tuple: (wisdom_df, learner_df, val_df, test_df)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_wisdom = os.path.join(cache_dir, "cached_wisdom_set.parquet")
    path_learner = os.path.join(cache_dir, "cached_learner_set.parquet")
    path_val = os.path.join(cache_dir, "cached_val_set.parquet")
    path_test = os.path.join(cache_dir, "cached_test_set.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(path_wisdom)
            and os.path.exists(path_learner)
            and os.path.exists(path_val)
            and os.path.exists(path_test)
        ):
            print("Loading cached Dual-Hygiene datasets...")
            wisdom_df = pd.read_parquet(path_wisdom)
            learner_df = pd.read_parquet(path_learner)
            val_df = pd.read_parquet(path_val)
            test_df = pd.read_parquet(path_test)
            return wisdom_df, learner_df, val_df, test_df
        else:
            print("Cache missing or incomplete. Regenerating datasets...")

    # 2. Load Raw Metadata
    print("Loading raw metadata...")
    train_full = load_dataset(os.path.join(Config.METADATA_DIR, "train.parquet"))
    val_full = load_dataset(os.path.join(Config.METADATA_DIR, "val.parquet"))
    test_full = load_dataset(os.path.join(Config.METADATA_DIR, "test.parquet"))

    # 3. Input Sanitization (Safety First)
    # Clamp coordinates to NYC bounding box to prevent extrapolation on garbage data
    print("Sanitizing coordinates (Clamping)...")
    train_full = clamp_coordinates(train_full, "pickup_latitude", "pickup_longitude")
    train_full = clamp_coordinates(train_full, "dropoff_latitude", "dropoff_longitude")

    val_full = clamp_coordinates(val_full, "pickup_latitude", "pickup_longitude")
    val_full = clamp_coordinates(val_full, "dropoff_latitude", "dropoff_longitude")

    test_full = clamp_coordinates(test_full, "pickup_latitude", "pickup_longitude")
    test_full = clamp_coordinates(test_full, "dropoff_latitude", "dropoff_longitude")

    # 4. Create Wisdom Set (Background Knowledge)
    # Strict filtering to generate robust statistical fingerprints
    print("Creating Wisdom Set (Strict Filtering)...")

    # Calculate distance for fare/km check
    dists = haversine_array(
        train_full["pickup_latitude"].values,
        train_full["pickup_longitude"].values,
        train_full["dropoff_latitude"].values,
        train_full["dropoff_longitude"].values,
    )

    # Apply filters
    # 1. Fare range
    mask_fare_range = (train_full["fare_amount"] >= Config.WISDOM_MIN_FARE) & (
        train_full["fare_amount"] <= Config.WISDOM_MAX_FARE
    )

    # 2. Fare per KM (avoid div by zero by ensuring dist > 0.001 km, i.e., 1 meter)
    # If distance is near zero, we exclude it from Wisdom stats to avoid noise
    mask_valid_dist = dists > 0.001

    # 3. Price per KM cap
    # We only check price/km where distance is valid
    mask_price_per_km = np.zeros(len(train_full), dtype=bool)
    mask_price_per_km[mask_valid_dist] = (
        train_full.loc[mask_valid_dist, "fare_amount"] / dists[mask_valid_dist]
    ) <= Config.WISDOM_MAX_FARE_PER_KM

    # Combine masks
    wisdom_mask = mask_fare_range & mask_valid_dist & mask_price_per_km
    wisdom_df = train_full[wisdom_mask].copy()

    # 5. Create Learner Set (Foreground Training)
    # Loose filtering + Subsampling
    print("Creating Learner Set (Loose Filtering + Subsampling)...")

    # Loose filter: just min fare
    learner_mask = train_full["fare_amount"] >= Config.LEARNER_MIN_FARE
    learner_candidates = train_full[learner_mask]

    # Subsampling
    if len(learner_candidates) > Config.LEARNER_SAMPLE_SIZE:
        learner_df = learner_candidates.sample(
            n=Config.LEARNER_SAMPLE_SIZE, random_state=Config.SEED
        ).copy()
    else:
        learner_df = learner_candidates.copy()

    # 6. Process Validation Set
    # Apply basic sanity filter (Loose) to avoid evaluating on negative fares
    print("Processing Validation Set...")
    val_mask = val_full["fare_amount"] >= Config.LEARNER_MIN_FARE
    val_df = val_full[val_mask].copy()

    # 7. Cache Results
    print("Caching datasets to working directory...")
    wisdom_df.to_parquet(path_wisdom, index=False)
    learner_df.to_parquet(path_learner, index=False)
    val_df.to_parquet(path_val, index=False)
    test_full.to_parquet(path_test, index=False)

    print(f"Wisdom Set Size: {len(wisdom_df)}")
    print(f"Learner Set Size: {len(learner_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_full)}")

    return wisdom_df, learner_df, val_df, test_full
