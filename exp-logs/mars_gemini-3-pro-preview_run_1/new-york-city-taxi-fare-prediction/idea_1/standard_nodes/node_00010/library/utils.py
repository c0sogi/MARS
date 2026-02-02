import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine (great-circle) distance between two points on the Earth surface.
    This function is vectorized and can handle numpy arrays or scalars.

    Args:
        lat1: Latitude of the first point(s) in decimal degrees.
        lon1: Longitude of the first point(s) in decimal degrees.
        lat2: Latitude of the second point(s) in decimal degrees.
        lon2: Longitude of the second point(s) in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    # Radius of earth in kilometers
    R = 6371.0

    # Convert decimal degrees to radians
    # np.radians handles both scalars and arrays efficiently
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    # Clip values to [-1, 1] to prevent NaNs in arcsin due to floating point errors
    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arcsin(np.sqrt(a))

    return R * c


def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Manhattan distance (L1 norm) between two points based on coordinates.
    This is the sum of the absolute differences of their coordinates.

    Args:
        lat1: Latitude of the first point(s).
        lon1: Longitude of the first point(s).
        lat2: Latitude of the second point(s).
        lon2: Longitude of the second point(s).

    Returns:
        The L1 distance (sum of absolute differences in degrees).
    """
    return np.abs(lat1 - lat2) + np.abs(lon1 - lon2)


# Airport Coordinates (Lat, Lon)
JFK_COORD = (40.6413, -73.7781)
LGA_COORD = (40.7769, -73.8740)
EWR_COORD = (40.6895, -74.1745)
# NYC Center (Approx Times Square)
NYC_CENTER_COORD = (40.7580, -73.9855)

# Additional City Landmarks (Cite solution_lesson_node_00009)
PENN_STATION_COORD = (40.7505, -73.9934)
GRAND_CENTRAL_COORD = (40.7527, -73.9772)
WTC_COORD = (40.7127, -74.0134)


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing between two points.
    Returns degrees in range [-180, 180].
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1

    y = np.sin(d_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    bearing = np.degrees(np.arctan2(y, x))
    return bearing
