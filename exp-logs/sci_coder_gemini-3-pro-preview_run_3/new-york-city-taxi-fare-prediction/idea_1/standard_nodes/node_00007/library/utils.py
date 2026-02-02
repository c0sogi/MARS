import numpy as np
from library.config import EARTH_RADIUS_KM


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth's surface
    using the Haversine formula.

    This function is stateless and vectorized, capable of handling both single
    float values and NumPy arrays/Pandas Series.

    Args:
        lat1 (float or array-like): Latitude of the pickup location(s) in degrees.
        lon1 (float or array-like): Longitude of the pickup location(s) in degrees.
        lat2 (float or array-like): Latitude of the dropoff location(s) in degrees.
        lon2 (float or array-like): Longitude of the dropoff location(s) in degrees.

    Returns:
        float or array-like: The distance between the points in kilometers.
    """
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Apply Haversine formula
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )

    # Calculate angular distance in radians
    # np.arctan2 is more numerically stable than np.arcsin for this purpose
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Convert to kilometers
    distance = EARTH_RADIUS_KM * c

    return distance
