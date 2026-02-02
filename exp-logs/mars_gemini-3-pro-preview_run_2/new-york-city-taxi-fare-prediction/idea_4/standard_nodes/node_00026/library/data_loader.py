import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import log_transform
from library.feature_engineering import (
    SpatialClusterer,
    clamp_coordinates,
    calculate_haversine,
    calculate_manhattan,
    calculate_bearing,
    extract_time_features,
)


def load_and_process_data(load_cached_data=True, debug=False, sample_size=100000):
    """
    Loads and processes the NYC Taxi dataset.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.
        debug (bool): If True, uses a smaller subset of data for quick debugging.
        sample_size (int): Number of rows to use if debug is True.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    cluster_cache = os.path.join(Config.WORKING_DIR, "cluster_centers.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(cluster_cache)
    )

    # 1. Try Loading from Cache
    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        try:
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)

            if debug:
                print(f"Debug mode: Sampling {sample_size} rows from cached data.")
                train_df = train_df.iloc[:sample_size]
                val_df = val_df.iloc[:sample_size]
                test_df = test_df.iloc[:sample_size]

            return train_df, val_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Proceeding to process from scratch.")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load raw metadata
    print("Loading raw metadata...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # Drop rows with missing coordinates to prevent MiniBatchKMeans failure
    # Cite debug_lesson_7: We filter invalid features from train/val to ensure pipeline stability,
    # distinct from target-based filtering which should be restricted to training only.
    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    train_df = train_df.dropna(subset=coord_cols)
    val_df = val_df.dropna(subset=coord_cols)

    # Apply Debug Sampling (before processing to save time)
    if debug:
        print(f"Debug mode: Sampling {sample_size} rows from raw data.")
        train_df = train_df.iloc[:sample_size].copy()
        val_df = val_df.iloc[:sample_size].copy()
        test_df = test_df.iloc[:sample_size].copy()

    # A. Coordinate Clamping
    print("Clamping coordinates...")
    train_df = clamp_coordinates(train_df)
    val_df = clamp_coordinates(val_df)
    test_df = clamp_coordinates(test_df)

    # B. Spatial Clustering
    print("Fitting and applying spatial clustering...")
    clusterer = SpatialClusterer(
        n_clusters=Config.N_CLUSTERS, random_state=Config.RANDOM_SEED
    )

    # Fit on training data (stack pickup and dropoff to learn map)
    coords_train = np.vstack(
        [
            train_df[["pickup_latitude", "pickup_longitude"]].values,
            train_df[["dropoff_latitude", "dropoff_longitude"]].values,
        ]
    )
    clusterer.fit(coords_train)

    # Save cluster centers (only if not debugging, to keep cache consistent)
    if not debug:
        clusterer.save(cluster_cache)

    # Apply to all splits
    for df in [train_df, val_df, test_df]:
        pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
        dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

        df["pickup_cluster"] = clusterer.predict(pickup_coords)
        df["dropoff_cluster"] = clusterer.predict(dropoff_coords)

    # C. Physics Features
    print("Calculating physics features...")
    for df in [train_df, val_df, test_df]:
        df["haversine_dist"] = calculate_haversine(df)
        df["manhattan_dist"] = calculate_manhattan(df)
        df["bearing"] = calculate_bearing(df)

    # D. Time Features
    print("Extracting time features...")
    train_df = extract_time_features(train_df)
    val_df = extract_time_features(val_df)
    test_df = extract_time_features(test_df)

    # Sanitize Target Variables (Cite debug_lesson_8)
    # Remove rows with negative fares to prevent NaN/Inf in log transform
    if "fare_amount" in train_df.columns:
        train_df = train_df[train_df["fare_amount"] >= 0]
        # Cite solution_lesson_node_00017: Sanitize target variable to prevent outliers from destabilizing L2 loss.
        # We filter extreme outliers from training only.
        train_df = train_df[train_df["fare_amount"] < 500]

    if "fare_amount" in val_df.columns:
        val_df = val_df[val_df["fare_amount"] >= 0]

    # E. Target Transformation
    if Config.USE_LOG_TARGET:
        print("Applying log transform to target...")
        if "fare_amount" in train_df.columns:
            train_df["fare_amount"] = log_transform(train_df["fare_amount"].values)
        if "fare_amount" in val_df.columns:
            val_df["fare_amount"] = log_transform(val_df["fare_amount"].values)

    # 3. Save to Cache (if not debug)
    if not debug:
        print("Saving processed data to cache...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
