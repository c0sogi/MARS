import numpy as np
import gc


def clean_memory():
    """
    Forces garbage collection to release unreferenced memory.
    Crucial when handling large datasets (e.g., 55M rows) to prevent OOM errors.
    """
    gc.collect()


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth.
    Vectorized implementation using NumPy.

    Args:
        lat1, lon1: Start point latitude and longitude (float or np.array).
        lat2, lon2: End point latitude and longitude (float or np.array).

    Returns:
        Distance in kilometers (float or np.array).
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance (L1 norm) between two points.
    This approximates driving distance in a grid-like city structure better than
    Euclidean distance. Adjusts for longitude scaling based on latitude.

    Args:
        lat1, lon1: Start point latitude and longitude (float or np.array).
        lat2, lon2: End point latitude and longitude (float or np.array).

    Returns:
        Distance in kilometers (float or np.array).
    """
    # Earth radius in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate angular differences
    dlat = np.abs(lat2_rad - lat1_rad)
    dlon = np.abs(lon2_rad - lon1_rad)

    # Calculate distance components
    # Latitude distance is roughly constant (R * radians)
    dist_lat = R * dlat

    # Longitude distance varies with latitude (scale by cos of average lat)
    avg_lat = (lat1_rad + lat2_rad) / 2.0
    dist_lon = R * np.cos(avg_lat) * dlon

    return dist_lat + dist_lon


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing (forward azimuth) from start to end point.
    Useful for capturing directional traffic patterns or airport routes.

    Args:
        lat1, lon1: Start point latitude and longitude (float or np.array).
        lat2, lon2: End point latitude and longitude (float or np.array).

    Returns:
        Bearing in degrees [0, 360) (float or np.array).
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlon = lon2_rad - lon1_rad

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlon)

    bearing_rad = np.arctan2(y, x)
    bearing_deg = np.degrees(bearing_rad)

    # Normalize to 0-360
    return (bearing_deg + 360) % 360


def rotate_coordinates(x, y, angle_degrees):
    """
    Rotates 2D coordinates by a given angle around the origin (0,0).
    Used to align latitude/longitude with the NYC street grid (approx 29 deg).

    Args:
        x: X coordinate or Longitude (float or np.array).
        y: Y coordinate or Latitude (float or np.array).
        angle_degrees: Rotation angle in degrees (counter-clockwise).

    Returns:
        Tuple of (rotated_x, rotated_y).
    """
    angle_rad = np.radians(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta

    return x_rot, y_rot
