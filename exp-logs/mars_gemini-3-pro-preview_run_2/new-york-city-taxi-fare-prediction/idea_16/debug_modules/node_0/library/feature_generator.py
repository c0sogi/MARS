import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config
from library.utils import haversine_distance


def add_temporal_features(df):
    """
    Extracts temporal features (hour, year, weekday) from pickup_datetime.
    Handles string-to-datetime conversion if necessary.
    """
    # Check if conversion is needed
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        # Optimization: Remove ' UTC' suffix if present to speed up parsing
        # Check first element to decide strategy, or use vectorized string replace safely
        if df["pickup_datetime"].dtype == "object":
            # Vectorized slice/replace is faster than strptime on large arrays
            # Assuming format "YYYY-MM-DD HH:MM:SS UTC" or similar
            df["pickup_datetime"] = (
                df["pickup_datetime"].astype(str).str.replace(" UTC", "")
            )

        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["weekday"] = df["pickup_datetime"].dt.dayofweek

    return df


def add_geometric_features(df):
    """
    Adds geometric features including Haversine distance, coordinate differences,
    and rotated coordinates to provide inductive bias for spatial splitting.
    """
    # Extract coordinates as numpy arrays for speed
    p_lon = df["pickup_longitude"].values
    p_lat = df["pickup_latitude"].values
    d_lon = df["dropoff_longitude"].values
    d_lat = df["dropoff_latitude"].values

    # 1. Physics Baseline: Haversine Distance
    df["dist_haversine"] = haversine_distance(p_lat, p_lon, d_lat, d_lon)

    # 2. Coordinate Differences (Absolute)
    # Helps model capture grid-like movement costs
    df["abs_lon_diff"] = np.abs(d_lon - p_lon)
    df["abs_lat_diff"] = np.abs(d_lat - p_lat)

    # 3. Manhattan Distance Proxy
    df["dist_manhattan"] = df["abs_lon_diff"] + df["abs_lat_diff"]

    # 4. Rotated Coordinates (45 degrees)
    # Decision trees split orthogonal to axes. Rotating coordinates by 45 degrees
    # (x+y, x-y) allows trees to capture diagonal boundaries efficiently.

    # Pickup Rotations
    df["pickup_rot_sum"] = p_lon + p_lat
    df["pickup_rot_diff"] = p_lon - p_lat

    # Dropoff Rotations
    df["dropoff_rot_sum"] = d_lon + d_lat
    df["dropoff_rot_diff"] = d_lon - d_lat

    return df


def process_features(df, cache_key=None, load_cached_data=True):
    """
    Orchestrates feature engineering with caching.

    Args:
        df (pd.DataFrame): Input dataframe (must contain 'margin' if it's the learner set).
        cache_key (str): Identifier for the cache file (e.g., 'featurized_train', 'featurized_test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    # Determine cache path
    cache_path = None
    if cache_key:
        cache_path = os.path.join(Config.WORKING_DIR, f"{cache_key}.parquet")

    # 1. Try Load
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return pd.read_parquet(cache_path)

    # 2. Process
    print(f"Generating features for {cache_key if cache_key else 'dataframe'}...")
    df = add_temporal_features(df)
    df = add_geometric_features(df)

    # 3. Save
    if cache_path:
        print(f"Saving features to {cache_path}...")
        df.to_parquet(cache_path, index=False)

    return df


def prepare_dmatrix(df, features, target_col=None):
    """
    Converts a DataFrame into an XGBoost DMatrix, explicitly setting the base_margin.

    Args:
        df (pd.DataFrame): Featurized dataframe containing the 'margin' column.
        features (list): List of column names to be used as features.
        target_col (str, optional): Name of the target variable column.

    Returns:
        xgb.DMatrix: The constructed DMatrix ready for training or inference.
    """
    # Extract Features
    X = df[features]

    # Extract Base Margin
    # The margin column is created by margin_logic.py
    if "margin" in df.columns:
        base_margin = df["margin"].values
    else:
        print("Warning: 'margin' column missing in DataFrame. Defaulting to 0.")
        base_margin = np.zeros(len(df))

    # Extract Label (if available)
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].values

    # Construct DMatrix
    # We pass feature_names to ensure consistency in inference
    dmatrix = xgb.DMatrix(
        data=X, label=y, base_margin=base_margin, feature_names=features
    )

    return dmatrix
