import numpy as np
from library.config import Config


def haversine_array(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Vectorized for numpy arrays or pandas series.

    Args:
        lat1: Latitude of the first point.
        lon1: Longitude of the first point.
        lat2: Latitude of the second point.
        lon2: Longitude of the second point.

    Returns:
        Distance in kilometers (float or numpy array).
    """
    # Radius of the earth in kilometers
    R = 6371.0

    # Convert latitude and longitude to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Difference in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    # a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )

    # c = 2 * atan2(sqrt(a), sqrt(1-a))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def rotate_coordinates(lon, lat, angle=None):
    """
    Rotate longitude and latitude coordinates by a given angle to align
    with the NYC street grid system.

    Args:
        lon: Numpy array or Series of longitudes (x coordinates).
        lat: Numpy array or Series of latitudes (y coordinates).
        angle: Rotation angle in degrees. If None, uses Config.ROTATION_ANGLE.

    Returns:
        Tuple of (rotated_lon, rotated_lat).
    """
    if angle is None:
        angle = Config.ROTATION_ANGLE

    # Convert angle to radians
    rads = np.radians(angle)

    # Precompute sine and cosine
    cos_val = np.cos(rads)
    sin_val = np.sin(rads)

    # Apply standard 2D rotation matrix
    # x' = x * cos(theta) - y * sin(theta)
    # y' = x * sin(theta) + y * cos(theta)
    lon_rot = lon * cos_val - lat * sin_val
    lat_rot = lon * sin_val + lat * cos_val

    return lon_rot, lat_rot
