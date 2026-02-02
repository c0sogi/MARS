import os
import gc
import numpy as np
import pandas as pd
from library.config import (
    CACHE_DIR,
    SEED,
    NYC_BBOX,
    EARTH_RADIUS_KM,
    GEOHASH_LEVELS,
)
from library.geo_utils import (
    compute_haversine,
    compute_manhattan,
    vectorized_geohash,
)
from library.data_factory import (
    get_wisdom_set,
    get_learner_set,
    get_val_set,
    get_test_set,
)
from library.fingerprint_engine import (
    compute_global_stats,
    get_oof_fingerprints,
    get_test_fingerprints,
)

# Airport Coordinates (approximate)
AIRPORTS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "TSQ": (40.7580, -73.9855),  # Times Square
}


def process_datetime(df):
    """Parses pickup_datetime and extracts temporal features."""
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        # Handle " UTC" suffix if present
        # We check the first element to see if string manipulation is needed
        first_val = df["pickup_datetime"].iloc[0] if not df.empty else ""
        if isinstance(first_val, str) and first_val.endswith(" UTC"):
            df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, -4)

        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek

    # Cyclic time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def add_physical_features(df):
    """Computes distance and coordinate features."""
    # 1. Basic Distances
    df["dist_haversine"] = compute_haversine(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    df["dist_manhattan"] = compute_manhattan(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    # 2. Directional Features (Delta)
    df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]

    # 3. Rotated Coordinates (45 degrees)
    # x' = x cos theta - y sin theta
    # y' = x sin theta + y cos theta
    # sin(45) = cos(45) ~= 0.7071
    s = 0.7071
    df["pickup_rot_x"] = df["pickup_longitude"] * s - df["pickup_latitude"] * s
    df["pickup_rot_y"] = df["pickup_longitude"] * s + df["pickup_latitude"] * s
    df["dropoff_rot_x"] = df["dropoff_longitude"] * s - df["dropoff_latitude"] * s
    df["dropoff_rot_y"] = df["dropoff_longitude"] * s + df["dropoff_latitude"] * s

    df["delta_rot_x"] = df["delta_lon"] * s - df["delta_lat"] * s
    df["delta_rot_y"] = df["delta_lon"] * s + df["delta_lat"] * s

    # 4. Airport Distances
    for code, (lat, lon) in AIRPORTS.items():
        # Distance from pickup
        df[f"pickup_dist_{code}"] = compute_haversine(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            np.full(len(df), lat),
            np.full(len(df), lon),
        )
        # Distance from dropoff
        df[f"dropoff_dist_{code}"] = compute_haversine(
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
            np.full(len(df), lat),
            np.full(len(df), lon),
        )

    return df


def compute_spatiotemporal_rate_table(wisdom_df):
    """
    Computes Mean Fare Per Km aggregated by (Geohash5, Hour) on Wisdom Set.
    Returns a DataFrame lookup.
    """
    print("Computing Spatiotemporal Rate Table from Wisdom Set...")

    # Ensure datetime
    wisdom_df = process_datetime(wisdom_df)

    # Compute Distance
    dists = compute_haversine(
        wisdom_df["pickup_latitude"].values,
        wisdom_df["pickup_longitude"].values,
        wisdom_df["dropoff_latitude"].values,
        wisdom_df["dropoff_longitude"].values,
    )

    # Compute Fare Per Km
    # Filter out very short trips to avoid explosion and noise
    valid_mask = dists > 0.1
    temp_df = wisdom_df.loc[valid_mask].copy()
    temp_dists = dists[valid_mask]

    temp_df["fare_per_km"] = temp_df["fare_amount"] / temp_dists

    # Compute Geohash L5
    temp_df["gh5"] = vectorized_geohash(
        temp_df["pickup_latitude"].values, temp_df["pickup_longitude"].values, 5
    )

    # Aggregate
    rate_table = temp_df.groupby(["gh5", "hour"])["fare_per_km"].mean().reset_index()
    rate_table.rename(columns={"fare_per_km": "rate_gh5_hour"}, inplace=True)

    return rate_table


def add_spatiotemporal_rate(df, rate_table):
    """
    Maps the spatiotemporal rate to the dataframe.
    """
    # Ensure features exist
    if "gh5" not in df.columns:
        df["gh5"] = vectorized_geohash(
            df["pickup_latitude"].values, df["pickup_longitude"].values, 5
        )
    if "hour" not in df.columns:
        df = process_datetime(df)

    # Merge
    # Left join to preserve rows
    merged = pd.merge(df, rate_table, on=["gh5", "hour"], how="left")

    # Fill missing with global mean of the rate table
    global_mean_rate = rate_table["rate_gh5_hour"].mean()
    merged["rate_gh5_hour"] = merged["rate_gh5_hour"].fillna(global_mean_rate)

    return merged


def build_features(load_cached_data: bool = True):
    """
    Orchestrates the feature engineering pipeline.
    Returns X_train, y_train, X_val, y_val, X_test, test_keys.
    """
    # Define Cache Paths
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {
        "X_train": os.path.join(CACHE_DIR, "X_train.parquet"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(CACHE_DIR, "X_val.parquet"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.parquet"),
        "test_keys": os.path.join(CACHE_DIR, "test_keys.npy"),
    }

    # Check Cache
    all_exist = all(os.path.exists(p) for p in paths.values())
    if load_cached_data and all_exist:
        print("Loading featurized datasets from cache...")
        X_train = pd.read_parquet(paths["X_train"])
        y_train = np.load(paths["y_train"])
        X_val = pd.read_parquet(paths["X_val"])
        y_val = np.load(paths["y_val"])
        X_test = pd.read_parquet(paths["X_test"])
        test_keys = np.load(paths["test_keys"], allow_pickle=True)
        return X_train, y_train, X_val, y_val, X_test, test_keys

    print("Building features from scratch...")

    # 1. Load Data
    wisdom_df = get_wisdom_set(load_cached_data=load_cached_data)
    learner_df = get_learner_set(load_cached_data=load_cached_data)
    val_df = get_val_set(load_cached_data=load_cached_data)
    test_df = get_test_set(load_cached_data=load_cached_data)

    # 2. Compute Global Wisdom Stats (for Fingerprinting)
    wisdom_stats = compute_global_stats(wisdom_df, load_cached_data=load_cached_data)

    # 3. Compute Spatiotemporal Rate Table (from Wisdom)
    rate_table = compute_spatiotemporal_rate_table(wisdom_df)

    # Free memory
    del wisdom_df
    gc.collect()

    # 4. Process Learner Set (Train)
    print("Processing Learner Set...")
    # A. Distributional Fingerprints (OOF)
    learner_feat = get_oof_fingerprints(
        learner_df, wisdom_stats, n_splits=5, load_cached_data=load_cached_data
    )
    # B. Physical & Temporal
    learner_feat = process_datetime(learner_feat)
    learner_feat = add_physical_features(learner_feat)
    # C. Spatiotemporal Rate
    learner_feat = add_spatiotemporal_rate(learner_feat, rate_table)

    # Prepare X and y
    y_train = learner_feat["fare_amount"].values
    # Drop non-feature columns
    drop_cols = ["key", "fare_amount", "pickup_datetime", "gh5"]
    X_train = learner_feat.drop(
        columns=[c for c in drop_cols if c in learner_feat.columns]
    )

    # Free memory
    del learner_df, learner_feat
    gc.collect()

    # 5. Process Validation Set
    print("Processing Validation Set...")
    # A. Distributional Fingerprints (Direct Map)
    val_feat = get_test_fingerprints(
        val_df, wisdom_stats, load_cached_data=load_cached_data, prefix="val"
    )
    # B. Physical & Temporal
    val_feat = process_datetime(val_feat)
    val_feat = add_physical_features(val_feat)
    # C. Spatiotemporal Rate
    val_feat = add_spatiotemporal_rate(val_feat, rate_table)

    y_val = val_feat["fare_amount"].values
    X_val = val_feat.drop(columns=[c for c in drop_cols if c in val_feat.columns])

    del val_df, val_feat
    gc.collect()

    # 6. Process Test Set
    print("Processing Test Set...")
    # A. Distributional Fingerprints (Direct Map)
    test_feat = get_test_fingerprints(
        test_df, wisdom_stats, load_cached_data=load_cached_data, prefix="test"
    )
    # B. Physical & Temporal
    test_feat = process_datetime(test_feat)
    test_feat = add_physical_features(test_feat)
    # C. Spatiotemporal Rate
    test_feat = add_spatiotemporal_rate(test_feat, rate_table)

    test_keys = test_feat["key"].values
    # Ensure columns match X_train
    X_test = test_feat.drop(columns=[c for c in drop_cols if c in test_feat.columns])

    # Align columns just in case
    X_test = X_test[X_train.columns]
    X_val = X_val[X_train.columns]

    del test_df, test_feat
    gc.collect()

    # 7. Save to Cache
    print("Saving features to cache...")
    X_train.to_parquet(paths["X_train"])
    np.save(paths["y_train"], y_train)
    X_val.to_parquet(paths["X_val"])
    np.save(paths["y_val"], y_val)
    X_test.to_parquet(paths["X_test"])
    np.save(paths["test_keys"], test_keys)

    return X_train, y_train, X_val, y_val, X_test, test_keys
