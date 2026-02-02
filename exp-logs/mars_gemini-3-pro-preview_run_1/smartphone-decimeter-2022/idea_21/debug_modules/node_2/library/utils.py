import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# --- WGS84 Ellipsoid Constants ---
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates.

    Args:
        lat (np.array or float): Latitude in degrees.
        lon (np.array or float): Longitude in degrees.
        alt (np.array or float): Altitude in meters.

    Returns:
        tuple: (x, y, z) in meters.
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF Cartesian coordinates to WGS84 geodetic coordinates.
    Uses Ferrari's solution for high precision.

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude and Altitude using Ferrari's solution
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    e_prime_sq = WGS84_E2 / (1 - WGS84_E2)

    lat_num = z + e_prime_sq * WGS84_B * np.sin(theta) ** 3
    lat_den = p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3
    lat = np.arctan2(lat_num, lat_den)

    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n

    return np.rad2deg(lat), np.rad2deg(lon), alt


def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convert WGS84 coordinates to Local Tangent Plane (ENU) coordinates
    relative to a reference point.

    Args:
        lat, lon, alt: Target coordinates (degrees, meters).
        ref_lat, ref_lon, ref_alt: Reference coordinates (degrees, meters).

    Returns:
        tuple: (e, n, u) in meters.
    """
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    xr, yr, zr = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    dx, dy, dz = x - xr, y - yr, z - zr

    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_wgs84(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local Tangent Plane (ENU) coordinates to WGS84 coordinates
    relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters.
        ref_lat, ref_lon, ref_alt: Reference coordinates (degrees, meters).

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    xr, yr, zr = wgs84_to_ecef(ref_lat, ref_lon, ref_alt)

    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Rotation matrix transpose (inverse) applied to (e, n, u) to get dXYZ
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = xr + dx
    y = yr + dy
    z = zr + dz

    return ecef_to_wgs84(x, y, z)
