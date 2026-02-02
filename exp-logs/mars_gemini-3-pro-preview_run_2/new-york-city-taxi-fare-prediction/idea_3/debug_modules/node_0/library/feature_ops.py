import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of starting point (float or array-like)
        lat2, lon2: Latitude and Longitude of ending point (float or array-like)

    Returns:
        Distance between points in kilometers (float or ndarray)
    """
    # Radius of earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    d = R * c
    return d


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance (L1 norm) between two points.

    Args:
        lat1, lon1: Latitude and Longitude of starting point
        lat2, lon2: Latitude and Longitude of ending point

    Returns:
        Sum of absolute differences in latitude and longitude (L1 distance)
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def rotate_coordinates(lat, lon, angle_degrees=45):
    """
    Rotates the coordinate system by a specified angle.
    This is useful for tree-based models to approximate diagonal distances
    on a grid system (like NYC streets).

    Args:
        lat: Latitude array (y-axis)
        lon: Longitude array (x-axis)
        angle_degrees: Rotation angle in degrees (default 45)

    Returns:
        Tuple of (rotated_lat, rotated_lon)
    """
    angle_rad = np.radians(angle_degrees)

    # Rotation matrix:
    # x' = x cos(theta) - y sin(theta)
    # y' = x sin(theta) + y cos(theta)
    # Mapping: x -> lon, y -> lat

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    lon_rot = lon * cos_theta - lat * sin_theta
    lat_rot = lon * sin_theta + lat * cos_theta

    return lat_rot, lon_rot
