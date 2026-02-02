import os
import random
import numpy as np
import torch

# WGS84 Coordinate System Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).
    Vectorized for numpy arrays.

    Args:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates (X, Y, Z) to geodetic (Latitude, Longitude, Altitude).
    Vectorized for numpy arrays using an iterative method.

    Args:
        x (float or np.array): ECEF X coordinate in meters.
        y (float or np.array): ECEF Y coordinate in meters.
        z (float or np.array): ECEF Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # Longitude is straightforward
    lon_rad = np.arctan2(y, x)

    # Iterative solution for Latitude and Altitude
    p = np.sqrt(x**2 + y**2)

    # Initial guess
    lat_rad = np.arctan2(z, p * (1 - WGS84_E2))

    # Iterate to converge
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
        alt = p / np.cos(lat_rad) - N
        lat_rad = np.arctan2(z, p * (1 - WGS84_E2 * (N / (N + alt))))

    lat = np.rad2deg(lat_rad)
    lon = np.rad2deg(lon_rad)

    # Final altitude calculation
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)
    alt = p / np.cos(lat_rad) - N

    return lat, lon, alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to Local Tangent Plane (East, North, Up) centered at (lat0, lon0, alt0).

    Args:
        x, y, z: Target ECEF coordinates.
        lat0, lon0, alt0: Reference geodetic coordinates (origin).

    Returns:
        tuple: (e, n, u) in meters.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Rotation matrix multiplication
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert Local Tangent Plane (East, North, Up) coordinates to ECEF.

    Args:
        e, n, u: Target ENU coordinates.
        lat0, lon0, alt0: Reference geodetic coordinates (origin).

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse rotation
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    return x0 + dx, y0 + dy, z0 + dz


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert Geodetic coordinates to ENU relative to a reference point.

    Args:
        lat, lon, alt: Target geodetic coordinates.
        lat0, lon0, alt0: Reference geodetic coordinates.

    Returns:
        tuple: (e, n, u) in meters.
    """
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, lat0, lon0, alt0)


def enu_to_geodetic(e, n, u, lat0, lon0, alt0):
    """
    Convert ENU coordinates to Geodetic.

    Args:
        e, n, u: Target ENU coordinates.
        lat0, lon0, alt0: Reference geodetic coordinates.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    x, y, z = enu_to_ecef(e, n, u, lat0, lon0, alt0)
    return ecef_to_geodetic(x, y, z)
