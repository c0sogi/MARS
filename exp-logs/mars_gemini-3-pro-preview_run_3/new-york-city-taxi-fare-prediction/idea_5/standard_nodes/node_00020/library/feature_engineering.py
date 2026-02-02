import os
import numpy as np
import pandas as pd
import gc
from library.config import (
    WORKING_DIR,
    BB_LAT_MIN,
    BB_LAT_MAX,
    BB_LON_MIN,
    BB_LON_MAX,
    FARE_MIN,
    FARE_MAX,
    ROTATION_ANGLE,
    LANDMARKS,
    R_EARTH_KM,
    RANDOM_SEED,
)


def haversine_array(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_EARTH_KM * c


def rotate_coordinates(lat, lon, angle_degrees):
    """
    Rotate coordinates by a specific angle to align with Manhattan grid.
    Returns rotated latitude (y') and longitude (x').
    """
    angle_rad = np.radians(angle_degrees)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    # Standard 2D rotation
    # x' = x cos(a) - y sin(a)
    # y' = x sin(a) + y cos(a)
    # Here we treat lon as x and lat as y
    lon_rot = lon * cos_a - lat * sin_a
    lat_rot = lon * sin_a + lat * cos_a

    return lat_rot, lon_rot


def add_landmark_distances(df):
    """
    Calculate haversine distance from pickup and dropoff points
    to defined landmarks.
    """
    for name, (l_lat, l_lon) in LANDMARKS.items():
        # Distance from Pickup
        df[f"dist_pickup_{name}"] = haversine_array(
            df["pickup_latitude"], df["pickup_longitude"], l_lat, l_lon
        )
        # Distance from Dropoff
        df[f"dist_dropoff_{name}"] = haversine_array(
            df["dropoff_latitude"], df["dropoff_longitude"], l_lat, l_lon
        )
    return df


def encode_cyclical_time(df, col, max_val):
    """
    Encode a cyclical feature (like hour) into sin and cos components.
    """
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def process_dataset(
    input_path, is_train=True, load_cached_data=True, debug_mode=False, output_name=None
):
    """
    Main function to load, clean, and engineer features for a dataset.
    Handles caching to avoid re-computation.
    """
    # Construct cache filename
    if output_name:
        filename = output_name
    else:
        filename = os.path.basename(input_path).replace(".parquet", "")

    if debug_mode:
        filename += "_debug"
    cache_path = os.path.join(WORKING_DIR, f"{filename}_processed.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached processed data from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Load Raw Data
    print(f"Processing data from {input_path}...")
    df = pd.read_parquet(input_path)

    if debug_mode:
        print("Debug mode: Sampling data...")
        df = df.sample(n=min(len(df), 100000), random_state=RANDOM_SEED).copy()

    initial_len = len(df)

    # 3. Data Cleaning / Filtering
    # Drop rows with NaN in coordinates
    df = df.dropna(
        subset=[
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]
    )

    # Bounding Box Filter (Universal Sanitation)
    # We apply this to both train and test to ensure model stability,
    # though for test we must handle the rows carefully (usually we don't drop test rows
    # in a real pipeline, but for this specific task structure we clean inputs).
    # Note: For Kaggle-style test sets, we usually predict even on outliers,
    # but extreme outliers cause feature explosion.
    # We will clip test data to bounds instead of dropping, or just drop for train.

    if is_train:
        # Strict dropping for training data
        mask = (
            (df["pickup_latitude"].between(BB_LAT_MIN, BB_LAT_MAX))
            & (df["pickup_longitude"].between(BB_LON_MIN, BB_LON_MAX))
            & (df["dropoff_latitude"].between(BB_LAT_MIN, BB_LAT_MAX))
            & (df["dropoff_longitude"].between(BB_LON_MIN, BB_LON_MAX))
        )
        df = df[mask]

        # Filter Target
        df = df[df["fare_amount"].between(FARE_MIN, FARE_MAX)]

        print(f"Data filtered: {initial_len} -> {len(df)} rows.")
    else:
        # For test/val, we don't drop rows to maintain alignment with submission file keys.
        # However, we might want to clip values or just leave them.
        # Given the task description, we should process all rows in test.
        pass

    # 4. Feature Engineering

    # A. Temporal Features
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year

    # Cyclical Encoding
    df = encode_cyclical_time(df, "hour", 24)
    df = encode_cyclical_time(df, "day_of_week", 7)
    df = encode_cyclical_time(df, "month", 12)

    # B. Spatial Features (Basic)
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

    # Haversine Distance
    df["haversine_dist"] = haversine_array(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # C. Physics-Informed Features (Rotation)
    # Rotate Pickup
    df["pickup_lat_rot"], df["pickup_lon_rot"] = rotate_coordinates(
        df["pickup_latitude"], df["pickup_longitude"], ROTATION_ANGLE
    )
    # Rotate Dropoff
    df["dropoff_lat_rot"], df["dropoff_lon_rot"] = rotate_coordinates(
        df["dropoff_latitude"], df["dropoff_longitude"], ROTATION_ANGLE
    )

    # Rotated Manhattan Distance (L1 Norm in rotated space)
    # This approximates driving distance on the grid
    df["rotated_manhattan_dist"] = (
        df["dropoff_lat_rot"] - df["pickup_lat_rot"]
    ).abs() + (df["dropoff_lon_rot"] - df["pickup_lon_rot"]).abs()

    # D. Landmark Features
    df = add_landmark_distances(df)

    # 5. Cleanup
    # Drop raw datetime to save memory/disk space, keep 'key' for submission alignment
    cols_to_drop = ["pickup_datetime"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Optimize types
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    # Be careful with key if it's int, but here key is string/object usually
    for col in int_cols:
        if col != "passenger_count":  # keep passenger count as is or int8
            df[col] = df[col].astype("int32")

    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].astype("uint8")

    # 6. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)

    gc.collect()
    return df
