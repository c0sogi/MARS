import numpy as np
import pandas as pd
from library.config import Config


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Start point coordinates (float or numpy array).
        lat2, lon2: End point coordinates (float or numpy array).

    Returns:
        Distance in kilometers (float or numpy array).
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance (L1 norm) between two points in coordinate degrees.
    This is useful for grid-based road networks like NYC.

    Args:
        lat1, lon1: Start point coordinates.
        lat2, lon2: End point coordinates.

    Returns:
        Manhattan distance in degrees (float or numpy array).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def clamp_coordinates(df):
    """
    Restricts latitude and longitude values to the valid NYC bounding box
    defined in Config. This prevents the model from extrapolating to
    coordinates far outside the training distribution.

    Args:
        df: Pandas DataFrame containing pickup/dropoff coordinate columns.

    Returns:
        Pandas DataFrame with clamped coordinates.
    """
    # Create a copy to avoid SettingWithCopy warnings if a view is passed
    df = df.copy()

    # Define columns to check
    lat_cols = ["pickup_latitude", "dropoff_latitude"]
    lon_cols = ["pickup_longitude", "dropoff_longitude"]

    # Clamp Latitudes
    for col in lat_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=Config.LAT_MIN, upper=Config.LAT_MAX)

    # Clamp Longitudes
    for col in lon_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=Config.LON_MIN, upper=Config.LON_MAX)

    return df
