import numpy as np
from library.config import WGS84_A, WGS84_F


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).

    Args:
        lat (float or np.ndarray): Latitude in degrees.
        lon (float or np.ndarray): Longitude in degrees.
        alt (float or np.ndarray): Altitude in meters.

    Returns:
        tuple: (x, y, z) coordinates in meters.
    """
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # WGS84 ellipsoid constants
    a = WGS84_A
    f = WGS84_F
    b = a * (1 - f)
    e2 = 2 * f - f**2  # Square of first eccentricity

    # Radius of curvature in the prime vertical
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    # ECEF coordinates
    x = (N + alt) * cos_lat * np.cos(lon_rad)
    y = (N + alt) * cos_lat * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * sin_lat

    return x, y, z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates (X, Y, Z) to geodetic (Latitude, Longitude, Altitude).
    Uses Ferrari's solution for high precision conversion.

    Args:
        x (float or np.ndarray): X coordinate in meters.
        y (float or np.ndarray): Y coordinate in meters.
        z (float or np.ndarray): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # WGS84 ellipsoid constants
    a = WGS84_A
    f = WGS84_F
    b = a * (1 - f)
    e2 = 2 * f - f**2
    ep2 = (a**2 - b**2) / b**2  # Square of second eccentricity

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Latitude
    lat_rad = np.arctan2(z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)

    # Longitude
    lon_rad = np.arctan2(y, x)

    # Altitude
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    # Calculate altitude
    # Using p / cos(lat) - N is generally stable for non-polar regions
    alt = p / cos_lat - N

    # Convert radians to degrees
    lat = np.rad2deg(lat_rad)
    lon = np.rad2deg(lon_rad)

    return lat, lon, alt
