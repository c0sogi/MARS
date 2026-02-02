import os
import numpy as np
import pandas as pd
from library.config import (
    METADATA_DIR,
    WORKING_DIR,
    NYC_BB,
    STRICT_FILTER,
    LOOSE_FILTER,
    GEOHASH_LEVELS,
    SEED,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
)
from library.utils import (
    clamp_coordinates,
    haversine_distance,
    manhattan_distance,
    vectorized_geohash,
)


def load_dataset(path):
    """
    Loads a dataset from a Parquet file.
    """
    return pd.read_parquet(path)


def apply_strict_filtering(df):
    """
    Applies STRICT filtering (The 'Wisdom') for generating global statistics.
    Removes outliers, data outside NYC bounding box, and invalid fare/distance ratios.
    """
    # 1. Spatial Filtering: Strictly keep points within NYC Bounding Box
    mask_spatial = (
        (df["pickup_longitude"] >= NYC_BB["min_lon"])
        & (df["pickup_longitude"] <= NYC_BB["max_lon"])
        & (df["pickup_latitude"] >= NYC_BB["min_lat"])
        & (df["pickup_latitude"] <= NYC_BB["max_lat"])
        & (df["dropoff_longitude"] >= NYC_BB["min_lon"])
        & (df["dropoff_longitude"] <= NYC_BB["max_lon"])
        & (df["dropoff_latitude"] >= NYC_BB["min_lat"])
        & (df["dropoff_latitude"] <= NYC_BB["max_lat"])
    )
    df = df[mask_spatial].copy()

    # Calculate distance temporarily for filtering
    dist = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    if "fare_amount" in df.columns:
        # 2. Fare Range Filtering
        mask_fare = (df["fare_amount"] >= STRICT_FILTER["fare_min"]) & (
            df["fare_amount"] <= STRICT_FILTER["fare_max"]
        )

        # 3. Fare per KM Filtering
        # Use a small epsilon for distance to avoid division by zero
        safe_dist = np.where(dist < 0.001, 0.001, dist)
        fare_per_km = df["fare_amount"] / safe_dist
        mask_rate = fare_per_km <= STRICT_FILTER["fare_per_km_max"]

        df = df[mask_fare & mask_rate].copy()

    return df


def apply_loose_filtering(df):
    """
    Applies LOOSE filtering (The 'Learner') for model training.
    Retains valid high-fare outliers but removes physical impossibilities.
    """
    # Calculate distance temporarily for filtering
    dist = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # 1. Minimum Distance Filtering
    mask_dist = dist >= LOOSE_FILTER["min_dist_km"]
    df = df[mask_dist].copy()

    if "fare_amount" in df.columns:
        # 2. Fare Range Filtering (Wider range than strict)
        mask_fare = (df["fare_amount"] >= LOOSE_FILTER["fare_min"]) & (
            df["fare_amount"] <= LOOSE_FILTER["fare_max"]
        )
        df = df[mask_fare].copy()

    return df


def enrich_geometric_features(df, compute_temporal=True):
    """
    Enriches the dataframe with geometric features:
    - Clamped Coordinates
    - Haversine & Manhattan Distances
    - Coordinate Deltas
    - Multi-Scale Geohashes
    - Temporal Features (Year, Month, Day, Weekday, Hour) - Cite solution_lesson_node_00065
    """
    # 1. Clamp Coordinates to NYC Bounding Box (In-place modification)
    df = clamp_coordinates(df)

    # 2. Calculate Distances
    df["haversine_dist"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["manhattan_dist"] = manhattan_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # 3. Calculate Coordinate Deltas
    df["abs_diff_lon"] = np.abs(df["dropoff_longitude"] - df["pickup_longitude"])
    df["abs_diff_lat"] = np.abs(df["dropoff_latitude"] - df["pickup_latitude"])

    # 4. Generate Geohashes (Multi-Scale)
    for level in GEOHASH_LEVELS:
        df[f"pickup_geohash_{level}"] = vectorized_geohash(
            df["pickup_latitude"], df["pickup_longitude"], level
        )
        df[f"dropoff_geohash_{level}"] = vectorized_geohash(
            df["dropoff_latitude"], df["dropoff_longitude"], level
        )

    # 5. Temporal Features (Cite solution_lesson_node_00065)
    # Essential for predicting deviations from the spatial mean (e.g. inflation, rush hour)
    if compute_temporal and "pickup_datetime" in df.columns:
        # Ensure datetime format
        if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        dt = df["pickup_datetime"].dt
        df["year"] = dt.year
        df["month"] = dt.month
        df["day"] = dt.day
        df["weekday"] = dt.dayofweek
        df["hour"] = dt.hour

    return df


def get_processed_data(
    split_name, mode="loose", subsample_size=None, load_cached_data=True
):
    """
    Orchestrates data loading, filtering, enrichment, and caching.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        mode (str): 'strict' (for stats), 'loose' (for training), 'inference' (for test/no filtering).
        subsample_size (int, optional): Number of rows to sample (for training).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Construct cache filename
    sub_str = f"_{subsample_size}" if subsample_size else ""
    cache_file = f"processed_{split_name}_{mode}{sub_str}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_file)

    # 1. Try Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {split_name} data (Mode: {mode})...")

    # 2. Determine Source Path
    if split_name == "train":
        path = TRAIN_PATH
    elif split_name == "val":
        path = VAL_PATH
    elif split_name == "test":
        path = TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split_name}")

    # 3. Load Raw Data
    df = load_dataset(path)

    # 4. Subsample (if requested)
    if subsample_size and len(df) > subsample_size:
        print(f"Subsampling {split_name} to {subsample_size} rows...")
        df = df.sample(n=subsample_size, random_state=SEED).reset_index(drop=True)

    # 5. Apply Filtering
    if mode == "strict":
        df = apply_strict_filtering(df)
    elif mode == "loose":
        df = apply_loose_filtering(df)
    # mode 'inference' applies no row filtering

    # 6. Enrich Features
    # Skip temporal features for 'strict' mode (used only for spatial stats) to save time
    compute_temporal = mode != "strict"
    df = enrich_geometric_features(df, compute_temporal=compute_temporal)

    # 7. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df
