import numpy as np
import pandas as pd
from library.config import NYC_BOUNDING_BOX, GRID_PRECISION


def clamp_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly clamps pickup and dropoff coordinates to the NYC bounding box.
    This prevents linear extrapolation artifacts from GPS errors.

    Args:
        df: Input DataFrame containing pickup/dropoff coordinates.

    Returns:
        DataFrame with clamped coordinates.
    """
    df = df.copy()

    # Clamp Pickup Coordinates
    df["pickup_latitude"] = df["pickup_latitude"].clip(
        lower=NYC_BOUNDING_BOX["lat_min"], upper=NYC_BOUNDING_BOX["lat_max"]
    )
    df["pickup_longitude"] = df["pickup_longitude"].clip(
        lower=NYC_BOUNDING_BOX["lon_min"], upper=NYC_BOUNDING_BOX["lon_max"]
    )

    # Clamp Dropoff Coordinates
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(
        lower=NYC_BOUNDING_BOX["lat_min"], upper=NYC_BOUNDING_BOX["lat_max"]
    )
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(
        lower=NYC_BOUNDING_BOX["lon_min"], upper=NYC_BOUNDING_BOX["lon_max"]
    )

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth (specified in decimal degrees).
    Vectorized implementation using numpy.

    Returns:
        Distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the L1 norm (Manhattan distance) on the coordinate grid.
    Useful for grid-like cities like NYC.

    Returns:
        L1 distance (sum of absolute differences in degrees).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def add_rotated_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rotated coordinates (45 degrees) to the DataFrame.
    This helps tree-based models capture diagonal features (e.g., Broadway).

    Rotation:
    x' = x * cos(45) - y * sin(45)
    y' = x * sin(45) + y * cos(45)

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with added columns:
        'pickup_rot_lat', 'pickup_rot_lon', 'dropoff_rot_lat', 'dropoff_rot_lon'
    """
    df = df.copy()

    # Rotation angle (45 degrees in radians)
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)

    # Rotate Pickup
    df["pickup_rot_lon"] = df["pickup_longitude"] * c - df["pickup_latitude"] * s
    df["pickup_rot_lat"] = df["pickup_longitude"] * s + df["pickup_latitude"] * c

    # Rotate Dropoff
    df["dropoff_rot_lon"] = df["dropoff_longitude"] * c - df["dropoff_latitude"] * s
    df["dropoff_rot_lat"] = df["dropoff_longitude"] * s + df["dropoff_latitude"] * c

    return df


def get_spatial_grid_id(df: pd.DataFrame, precision: int = GRID_PRECISION) -> pd.Series:
    """
    Generates a unique string identifier for the route (Pickup -> Dropoff) based on
    coordinates rounded to the specified precision. This simulates Geohashing.

    Format: "{p_lat}_{p_lon}_{d_lat}_{d_lon}"

    Args:
        df: Input DataFrame.
        precision: Number of decimal places to round to.

    Returns:
        Pandas Series of string IDs.
    """
    # Round coordinates
    p_lat = df["pickup_latitude"].round(precision)
    p_lon = df["pickup_longitude"].round(precision)
    d_lat = df["dropoff_latitude"].round(precision)
    d_lon = df["dropoff_longitude"].round(precision)

    # Create ID string
    # Using string concatenation which is efficient enough for this purpose
    grid_id = (
        p_lat.astype(str)
        + "_"
        + p_lon.astype(str)
        + "_"
        + d_lat.astype(str)
        + "_"
        + d_lon.astype(str)
    )

    return grid_id
