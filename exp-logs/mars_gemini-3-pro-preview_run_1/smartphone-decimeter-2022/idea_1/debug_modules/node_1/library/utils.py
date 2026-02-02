import numpy as np


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered Earth-Fixed (ECEF) coordinates to Geodetic coordinates
    (Latitude, Longitude, Altitude) using the WGS84 ellipsoid.

    This function uses Ferrari's solution (a robust closed-form approximation) to
    convert Cartesian coordinates (x, y, z) to Geodetic coordinates.

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt)
            lat (np.array or float): Latitude in degrees.
            lon (np.array or float): Longitude in degrees.
            alt (np.array or float): Altitude in meters.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis

    e2 = 2 * f - f**2  # first eccentricity squared
    ep2 = (a**2 - b**2) / b**2  # second eccentricity squared

    # Calculations
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    # Calculate Longitude
    lon = np.arctan2(y, x)

    # Calculate Latitude
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    # Calculate Altitude
    # Radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    using the Haversine formula.

    Args:
        lat1 (np.array or float): Latitude of the first point.
        lon1 (np.array or float): Longitude of the first point.
        lat2 (np.array or float): Latitude of the second point.
        lon2 (np.array or float): Longitude of the second point.

    Returns:
        np.array or float: Distance between points in meters.
    """
    # Earth radius in meters (mean radius)
    R = 6371000.0

    # Convert decimal degrees to radians
    phi1, lambda1 = np.radians(lat1), np.radians(lon1)
    phi2, lambda2 = np.radians(lat2), np.radians(lon2)

    # Differences
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    # Haversine formula
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    # Avoid numerical errors for antipodal points (a > 1)
    a = np.clip(a, 0, 1)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance
