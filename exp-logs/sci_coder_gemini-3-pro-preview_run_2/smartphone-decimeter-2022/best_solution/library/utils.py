import numpy as np


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to Latitude, Longitude, Altitude.

    This function uses a closed-form approximation suitable for WGS84.

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (latitude, longitude, altitude) in decimal degrees and meters.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2  # Eccentricity derived from flattening

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    # Calculate Longitude
    lon = np.arctan2(y, x)

    # Calculate Latitude
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate Altitude
    # N is the radius of curvature in the prime vertical
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e**2 * sin_lat**2)

    # Calculate altitude (robust for non-polar regions)
    # For exact polar handling, one might check if cos(lat) is near zero,
    # but for driving datasets, this is sufficient.
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees) using the Haversine formula.

    Args:
        lat1 (np.array or float): Latitude of the first point in degrees.
        lon1 (np.array or float): Longitude of the first point in degrees.
        lat2 (np.array or float): Latitude of the second point in degrees.
        lon2 (np.array or float): Longitude of the second point in degrees.

    Returns:
        np.array or float: Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    # Clip to ensure value is within domain of arcsin/arctan due to floating point errors
    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
