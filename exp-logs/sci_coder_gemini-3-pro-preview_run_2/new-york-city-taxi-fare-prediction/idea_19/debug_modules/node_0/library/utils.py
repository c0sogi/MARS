import numpy as np
import pandas as pd
from library.config import ProjectConfig


def clamp_coordinates(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Strictly clamps pickup and dropoff coordinates to the NYC bounding box defined in ProjectConfig.
    This prevents linear extrapolation artifacts from GPS errors and ensures all data points
    fall within the valid spatial domain for the model.
    """
    if not inplace:
        df = df.copy()

    # Bounding Box: [Lon Min, Lat Min, Lon Max, Lat Max]
    bb = ProjectConfig.NYC_BOUNDING_BOX
    lon_min, lat_min, lon_max, lat_max = bb[0], bb[1], bb[2], bb[3]

    # Clamp Pickup Coordinates
    df["pickup_longitude"] = df["pickup_longitude"].clip(lower=lon_min, upper=lon_max)
    df["pickup_latitude"] = df["pickup_latitude"].clip(lower=lat_min, upper=lat_max)

    # Clamp Dropoff Coordinates
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(lower=lon_min, upper=lon_max)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(lower=lat_min, upper=lat_max)

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth.
    Inputs are in decimal degrees.
    Returns distance in Kilometers.
    """
    # Convert decimal degrees to radians
    # Handle both scalar and array inputs
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
    Calculates the L1 distance (Manhattan distance) in degrees.
    This serves as a proxy for travel distance in a grid-like city.
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing (direction of travel) between two points.
    Returns degrees [0, 360).
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlon)

    initial_bearing = np.arctan2(y, x)
    # Convert to degrees and normalize to 0-360
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


def rotate_coordinates(lat, lon, angle_deg=-29):
    """
    Rotates coordinates by a given angle (default -29 degrees for NYC)
    to align with the street grid. Returns (lat_rot, lon_rot).
    """
    theta = np.radians(angle_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Standard rotation matrix application
    # x' = x cos - y sin
    # y' = x sin + y cos
    # Here we treat lon as x, lat as y
    lon_rot = lon * cos_theta - lat * sin_theta
    lat_rot = lon * sin_theta + lat * cos_theta

    return lat_rot, lon_rot


def calculate_geohash(lat, lon, precision):
    """
    Vectorized grid binning approximating Geohash precision levels.
    Maps coordinates to a unique integer ID representing a spatial grid cell.

    Precision mappings (approximate grid sizes):
    Level 5: ~5km   (Step 0.045 deg)
    Level 6: ~1km   (Step 0.009 deg)
    Level 7: ~150m  (Step 0.0013 deg)

    Args:
        lat (np.array or pd.Series): Latitude values.
        lon (np.array or pd.Series): Longitude values.
        precision (int): Geohash level (5, 6, or 7).

    Returns:
        np.array: Integer IDs for the grid cells.
    """
    # Define step sizes for each level to approximate Geohash dimensions
    if precision == 5:
        step = 0.045
    elif precision == 6:
        step = 0.009
    elif precision == 7:
        step = 0.0013
    else:
        raise ValueError(f"Unsupported precision level: {precision}. Use 5, 6, or 7.")

    # Define offsets based on NYC Bounding Box to ensure positive indices
    # NYC BB: [-74.5, 40.5, -72.8, 41.95]
    # We use slightly wider bounds for the offset origin
    lat_offset = 40.0
    lon_offset = -75.0

    # Calculate grid indices
    # We use floor division implicitly by casting to int
    lat_idx = ((lat - lat_offset) / step).astype(np.int32)
    lon_idx = ((lon - lon_offset) / step).astype(np.int32)

    # Combine into unique integer ID
    # Multiplier 10000 ensures no overlap (Lon range ~3 deg / 0.0013 ~ 2300 bins)
    # Result fits comfortably in int32
    return lat_idx * 10000 + lon_idx
