import os
import numpy as np
import pandas as pd
from library.config import METADATA_DIR, WORKING_DIR, NYC_BOUNDING_BOX
from library.utils import clamp_coordinates, haversine_array, bearing_array


def add_physical_features(df):
    """
    Generates physical and temporal features from raw coordinates and timestamps.

    Args:
        df (pd.DataFrame): Input dataframe with raw features.

    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original df
    df = df.copy()

    # ---------------------------------------------------------
    # Temporal Features
    # ---------------------------------------------------------
    # Ensure pickup_datetime is datetime type
    # The format in metadata is like '2014-03-30 12:14:00 UTC'
    if "pickup_datetime" in df.columns:
        # Efficient parsing: infer_datetime_format is deprecated in newer pandas,
        # but specifying format or letting pandas guess is fine.
        # We strip ' UTC' if present to speed up parsing or handle it via format.
        # Given the volume, we let pandas handle it, but ensure it's datetime.
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        df["year"] = df["pickup_datetime"].dt.year
        df["month"] = df["pickup_datetime"].dt.month
        df["day"] = df["pickup_datetime"].dt.day
        df["hour"] = df["pickup_datetime"].dt.hour
        df["weekday"] = df["pickup_datetime"].dt.dayofweek

    # ---------------------------------------------------------
    # Spatial Features
    # ---------------------------------------------------------
    # Ensure coordinates are float
    coords = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    for c in coords:
        df[c] = df[c].astype(float)

    # 1. Haversine Distance (km)
    df["dist_haversine"] = haversine_array(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    # 2. Manhattan Distance (approximate km)
    # L1 norm in degrees converted to km.
    # Approx: 1 deg lat ~= 111 km, 1 deg lon ~= 85 km (at 40 deg lat)
    # We use a simplified conversion for feature engineering
    lat_diff = np.abs(df["pickup_latitude"] - df["dropoff_latitude"])
    lon_diff = np.abs(df["pickup_longitude"] - df["dropoff_longitude"])

    # Using a constant approximation for NYC latitude (~40.7)
    # cos(40.7) approx 0.758
    df["dist_manhattan"] = (lat_diff * 111.0) + (lon_diff * 85.0)

    # Also keep raw degree L1 as it's scale-invariant for trees
    df["dist_manhattan_deg"] = lat_diff + lon_diff

    # 3. Bearing
    df["bearing"] = bearing_array(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )

    return df


def load_and_clean_data(split_name, load_cached_data=True):
    """
    Loads data for a specific split, applies cleaning (clamping),
    and manages caching.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from WORKING_DIR.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    cache_path = os.path.join(WORKING_DIR, f"{split_name}_cleaned.parquet")

    # 1. Try Load from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached cleaned data for {split_name} from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Load from Metadata
    filename = f"{split_name}.parquet"
    input_path = os.path.join(METADATA_DIR, filename)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    print(f"Loading raw data for {split_name} from {input_path}...")
    df = pd.read_parquet(input_path)

    # 3. Apply Cleaning (Clamping)
    # This uses the utility function which references the config bounding box
    print(f"Clamping coordinates for {split_name}...")
    df = clamp_coordinates(df)

    # Handle missing values in coordinates
    # Train/Val: Drop rows to prevent errors in feature engineering (IntCastingNaNError)
    # Test: Fill with 0 to ensure submission completeness
    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    # Filter to only columns present in the dataframe
    coord_cols = [c for c in coord_cols if c in df.columns]

    if split_name in ["train", "val"]:
        initial_len = len(df)
        df = df.dropna(subset=coord_cols)
        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            print(
                f"Dropped {dropped_count} rows with missing coordinates in {split_name}."
            )
    else:
        # For test set, fill NaNs to prevent crashes during type casting
        df[coord_cols] = df[coord_cols].fillna(0.0)

    # 4. Save to Cache
    print(f"Saving cleaned data for {split_name} to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def process_data(split_name, load_cached_data=True):
    """
    Orchestrates the loading, cleaning, and feature engineering for a split.
    Result is also cached to avoid re-computing features.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached files.

    Returns:
        pd.DataFrame: The fully processed dataframe ready for model input.
    """
    # Define a separate cache for the fully processed file (with features)
    processed_cache_path = os.path.join(WORKING_DIR, f"{split_name}_processed.parquet")

    # Try to load fully processed data first
    if load_cached_data and os.path.exists(processed_cache_path):
        print(
            f"Loading fully processed data for {split_name} from {processed_cache_path}..."
        )
        try:
            df = pd.read_parquet(processed_cache_path)
            return df
        except Exception as e:
            print(f"Failed to load processed cache: {e}. Re-computing...")

    # If not cached, load cleaned data (which might itself be cached)
    df = load_and_clean_data(split_name, load_cached_data=load_cached_data)

    # Apply Feature Engineering
    print(f"Generating physical features for {split_name}...")
    df = add_physical_features(df)

    # Save fully processed data
    print(f"Saving processed data for {split_name} to {processed_cache_path}...")
    df.to_parquet(processed_cache_path, index=False)

    return df
