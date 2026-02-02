import numpy as np
import pandas as pd
from library.config import NYC_BOUNDING_BOX


def haversine_array(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Start point coordinates (float or numpy array).
        lat2, lon2: End point coordinates (float or numpy array).

    Returns:
        Distance in kilometers (float or numpy array).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Clip 'a' to [0, 1] to avoid floating point errors causing NaNs in arcsin
    a = np.clip(a, 0, 1)

    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers is approx 6371
    km = 6371 * c
    return km


def bearing_array(lat1, lon1, lat2, lon2):
    """
    Calculate the initial bearing (forward azimuth) between two points.

    Args:
        lat1, lon1: Start point coordinates (float or numpy array).
        lat2, lon2: End point coordinates (float or numpy array).

    Returns:
        Bearing in degrees (0-360) (float or numpy array).
    """
    # Convert decimal degrees to radians
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2_rad - lon1_rad

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlon)

    initial_bearing = np.arctan2(y, x)

    # Convert radians to degrees
    initial_bearing = np.degrees(initial_bearing)

    # Normalize to 0-360
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def clamp_coordinates(df):
    """
    Clamps coordinate columns in the dataframe to the bounding box
    defined in the configuration to remove outliers.

    Args:
        df: Input pandas DataFrame containing coordinate columns.

    Returns:
        A new pandas DataFrame with clamped coordinates.
    """
    df_clamped = df.copy()

    # Define columns to map to bounding box limits
    # Keys correspond to standard dataset columns, Values to config keys
    coordinate_map = {
        "pickup_longitude": ("lon_min", "lon_max"),
        "dropoff_longitude": ("lon_min", "lon_max"),
        "pickup_latitude": ("lat_min", "lat_max"),
        "dropoff_latitude": ("lat_min", "lat_max"),
    }

    for col, (min_key, max_key) in coordinate_map.items():
        if col in df_clamped.columns:
            min_val = NYC_BOUNDING_BOX[min_key]
            max_val = NYC_BOUNDING_BOX[max_key]
            df_clamped[col] = df_clamped[col].clip(lower=min_val, upper=max_val)

    return df_clamped
