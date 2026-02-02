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


# NYC Landmarks Coordinates (Lat, Lon)
NYC_LANDMARKS = {
    "jfk": (40.6413, -73.7781),
    "lga": (40.7769, -73.8740),
    "ewr": (40.6895, -74.1745),
    "tsq": (40.7580, -73.9855),  # Times Square
    "penn": (40.7505, -73.9934),  # Penn Station
    "gct": (40.7527, -73.9772),  # Grand Central Terminal
    "wtc": (40.7127, -74.0134),  # World Trade Center
}
