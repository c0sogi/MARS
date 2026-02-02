import numpy as np
import pandas as pd
from library.config import (
    NYC_MIN_LON,
    NYC_MAX_LON,
    NYC_MIN_LAT,
    NYC_MAX_LAT,
    GRID_PRECISION,
)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1, lat2, lon2: float or array-like
            Coordinates in decimal degrees.

    Returns:
        float or array-like: Distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Radius of earth in kilometers
    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the L1 norm (Manhattan distance) between two points.
    This is a proxy for city block distance in the coordinate space.

    Args:
        lat1, lon1, lat2, lon2: float or array-like
            Coordinates in decimal degrees.

    Returns:
        float or array-like: Distance metric (sum of absolute differences).
    """
    # We calculate the Manhattan distance in degrees.
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def rotate_coordinates(lat, lon, angle_degrees=29):
    """
    Rotates coordinates by a given angle to align with the NYC street grid.

    Args:
        lat, lon: float or array-like
            Coordinates.
        angle_degrees: float
            Rotation angle in degrees. Default 29 aligns roughly with Manhattan.

    Returns:
        tuple: (rotated_lat, rotated_lon)
    """
    theta = np.radians(angle_degrees)

    # Standard 2D rotation matrix:
    # x' = x cos(theta) - y sin(theta)
    # y' = x sin(theta) + y cos(theta)
    # Mapping lon to x and lat to y

    lon_rot = lon * np.cos(theta) - lat * np.sin(theta)
    lat_rot = lon * np.sin(theta) + lat * np.cos(theta)

    return lat_rot, lon_rot


def discretize_coordinates(coords, precision=GRID_PRECISION):
    """
    Rounds coordinates to a specific precision to create grid buckets.

    Args:
        coords: float or array-like
            Input coordinates.
        precision: int
            Number of decimal places.

    Returns:
        float or array-like: Rounded coordinates.
    """
    return np.round(coords, precision)


def clamp_coordinates(
    df,
    min_lon=NYC_MIN_LON,
    max_lon=NYC_MAX_LON,
    min_lat=NYC_MIN_LAT,
    max_lat=NYC_MAX_LAT,
):
    """
    Clamps coordinate columns in the DataFrame to the NYC bounding box.
    This prevents linear extrapolation errors for outliers.

    Args:
        df: pd.DataFrame
            Input dataframe containing pickup/dropoff coordinates.
        min_lon, max_lon, min_lat, max_lat: float
            Bounding box limits.

    Returns:
        pd.DataFrame: Dataframe with clamped coordinates.
    """
    df = df.copy()

    if "pickup_longitude" in df.columns:
        df["pickup_longitude"] = df["pickup_longitude"].clip(min_lon, max_lon)
    if "dropoff_longitude" in df.columns:
        df["dropoff_longitude"] = df["dropoff_longitude"].clip(min_lon, max_lon)
    if "pickup_latitude" in df.columns:
        df["pickup_latitude"] = df["pickup_latitude"].clip(min_lat, max_lat)
    if "dropoff_latitude" in df.columns:
        df["dropoff_latitude"] = df["dropoff_latitude"].clip(min_lat, max_lat)

    return df


def extract_datetime_features(df):
    """
    Extracts temporal features from the pickup_datetime column.

    Args:
        df: pd.DataFrame
            Input dataframe with 'pickup_datetime'.

    Returns:
        pd.DataFrame: Dataframe with added time features.
    """
    df = df.copy()

    # Ensure datetime format
    # The dataset often has ' UTC' suffix which we strip for speed before parsing
    if (
        df["pickup_datetime"].dtype == "object"
        or df["pickup_datetime"].dtype == "string"
    ):
        # Check first element to see if string manipulation is needed
        first_val = df["pickup_datetime"].iloc[0]
        if isinstance(first_val, str) and first_val.endswith(" UTC"):
            df["pickup_datetime"] = pd.to_datetime(
                df["pickup_datetime"].str.slice(0, -4)
            )
        else:
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    dt = df["pickup_datetime"].dt

    df["hour"] = dt.hour
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.day
    df["weekday"] = dt.dayofweek

    return df


def add_distance_features(df):
    """
    Adds Haversine, Manhattan, and Rotated coordinate features.

    Args:
        df: pd.DataFrame
            Input dataframe with coordinate columns.

    Returns:
        pd.DataFrame: Dataframe with added distance features.
    """
    df = df.copy()

    # Base coordinates
    plat = df["pickup_latitude"]
    plon = df["pickup_longitude"]
    dlat = df["dropoff_latitude"]
    dlon = df["dropoff_longitude"]

    # 1. Haversine Distance
    df["distance_haversine"] = haversine_distance(plat, plon, dlat, dlon)

    # 2. Manhattan Distance (L1)
    df["distance_manhattan"] = manhattan_distance(plat, plon, dlat, dlon)

    # 3. Rotated Coordinates (for tree-based splits on grid)
    # Rotation 1: 29 degrees (Manhattan grid alignment)
    plat_rot, plon_rot = rotate_coordinates(plat, plon, angle_degrees=29)
    dlat_rot, dlon_rot = rotate_coordinates(dlat, dlon, angle_degrees=29)

    df["pickup_latitude_rot29"] = plat_rot
    df["pickup_longitude_rot29"] = plon_rot
    df["dropoff_latitude_rot29"] = dlat_rot
    df["dropoff_longitude_rot29"] = dlon_rot

    # 4. Rotated Manhattan Distance
    df["distance_manhattan_rot29"] = np.abs(plat_rot - dlat_rot) + np.abs(
        plon_rot - dlon_rot
    )

    return df
