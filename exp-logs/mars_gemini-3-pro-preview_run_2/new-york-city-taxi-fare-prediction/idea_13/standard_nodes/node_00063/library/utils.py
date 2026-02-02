import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1 (float or np.ndarray): Latitude of the first point(s).
        lon1 (float or np.ndarray): Longitude of the first point(s).
        lat2 (float or np.ndarray): Latitude of the second point(s).
        lon2 (float or np.ndarray): Longitude of the second point(s).

    Returns:
        float or np.ndarray: Distance between the points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the L1 (Manhattan) distance between two points based on coordinate differences.
    Note: This returns the distance in degrees, not kilometers.

    Args:
        lat1 (float or np.ndarray): Latitude of the first point(s).
        lon1 (float or np.ndarray): Longitude of the first point(s).
        lat2 (float or np.ndarray): Latitude of the second point(s).
        lon2 (float or np.ndarray): Longitude of the second point(s).

    Returns:
        float or np.ndarray: L1 distance (|delta_lat| + |delta_lon|).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


def clamp_values(values, min_val, max_val):
    """
    Restricts values in an array to be within a specified range [min_val, max_val].

    Args:
        values (np.ndarray or float): Input values to clamp.
        min_val (float): Minimum allowed value.
        max_val (float): Maximum allowed value.

    Returns:
        np.ndarray or float: Clamped values.
    """
    return np.clip(values, min_val, max_val)


def rotate_coordinates(lat, lon, angle_degrees=0.0):
    """
    Rotates coordinates by a specified angle around the origin (0,0).
    Useful for aligning coordinates with the NYC street grid (approx 29 degrees).

    Args:
        lat (np.ndarray or float): Latitude values (treated as Y).
        lon (np.ndarray or float): Longitude values (treated as X).
        angle_degrees (float): Rotation angle in degrees.

    Returns:
        tuple: (rotated_lat, rotated_lon)
    """
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # Rotation matrix application:
    # x' = x*cos(theta) - y*sin(theta)
    # y' = x*sin(theta) + y*cos(theta)
    # Here we map lon -> x, lat -> y

    lon_rot = lon * cos_theta - lat * sin_theta
    lat_rot = lon * sin_theta + lat * cos_theta

    return lat_rot, lon_rot
