import numpy as np
import pandas as pd
from library.config import Config


def clamp_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restricts pickup and dropoff coordinates to the bounding box defined in Config.
    This prevents linear extrapolation artifacts from outliers and ensures
    spatial features remain within the valid domain.

    Args:
        df (pd.DataFrame): Input dataframe containing pickup/dropoff coordinates.

    Returns:
        pd.DataFrame: Dataframe with coordinates clipped to the bounding box.
    """
    # Operate on a copy to ensure safety against SettingWithCopy warnings
    # and to preserve the original dataframe if needed elsewhere.
    df = df.copy()

    # Clamp Pickup Coordinates
    df["pickup_longitude"] = df["pickup_longitude"].clip(
        lower=Config.BB_MIN_LON, upper=Config.BB_MAX_LON
    )
    df["pickup_latitude"] = df["pickup_latitude"].clip(
        lower=Config.BB_MIN_LAT, upper=Config.BB_MAX_LAT
    )

    # Clamp Dropoff Coordinates
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(
        lower=Config.BB_MIN_LON, upper=Config.BB_MAX_LON
    )
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(
        lower=Config.BB_MIN_LAT, upper=Config.BB_MAX_LAT
    )

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized for numpy arrays or pandas series.

    Args:
        lat1, lon1: Start coordinates (float or array-like).
        lat2, lon2: End coordinates (float or array-like).

    Returns:
        float or np.array: Distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Radius of earth in kilometers

    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance (L1 norm) in kilometers,
    approximating the grid-like street structure of NYC.

    Args:
        lat1, lon1: Start coordinates.
        lat2, lon2: End coordinates.

    Returns:
        float or np.array: Distance in kilometers.
    """
    # Approximate conversion factors at NYC latitude (~40.7 deg)
    # 1 deg lat ~= 111 km
    # 1 deg lon ~= 85 km (111 * cos(40.7))
    km_per_lat = 111.0
    km_per_lon = 85.0

    dlat = np.abs(lat1 - lat2) * km_per_lat
    dlon = np.abs(lon1 - lon2) * km_per_lon

    return dlat + dlon


def compute_geohash_bins(lat, lon, level: int):
    """
    Discretizes coordinates into spatial grid indices, simulating Geohashes.
    Used to generate hierarchical priors (Multi-Moment).

    Levels correspond to approximate grid cell edge lengths:
    - Level 5: ~5km  (Step ~0.045 deg)
    - Level 6: ~1km  (Step ~0.009 deg)
    - Level 7: ~150m (Step ~0.00135 deg)

    Args:
        lat, lon: Coordinates to bin (float or array-like).
        level (int): The hierarchy level (5, 6, or 7).

    Returns:
        np.array: Integer hash representing the bin index.
    """
    # Define grid steps (degrees) based on requested level
    # These approximate the resolution of standard Geohash levels
    if level == 5:
        step = 0.045
    elif level == 6:
        step = 0.009
    elif level == 7:
        step = 0.00135
    else:
        raise ValueError(f"Unsupported geohash level: {level}. Use 5, 6, or 7.")

    # Calculate offsets from the bounding box origin
    # Using Config bounds as the origin ensures global alignment across train/test
    lat_offset = lat - Config.BB_MIN_LAT
    lon_offset = lon - Config.BB_MIN_LON

    # Compute grid indices
    # We use floor to bin coordinates into cells
    lat_idx = np.floor(lat_offset / step).astype(np.int32)
    lon_idx = np.floor(lon_offset / step).astype(np.int32)

    # Create a unique integer ID for the bin
    # Multiplier must be larger than the maximum number of longitude bins
    # Max width = 0.5 deg. Min step = 0.00135. Max bins ~ 370.
    # Multiplier 10000 is safe and keeps IDs readable.
    multiplier = 10000

    # Calculate unique Bin ID
    # Format: YYYYXXXX where Y is lat index and X is lon index
    bin_ids = lat_idx * multiplier + lon_idx

    return bin_ids
