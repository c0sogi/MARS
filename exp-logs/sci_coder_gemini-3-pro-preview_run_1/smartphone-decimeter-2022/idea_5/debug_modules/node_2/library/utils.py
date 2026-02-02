import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name="GNSS_Logger"):
    """
    Returns a simple logger-like object or setup.
    For this environment, standard print is sufficient, but this provides a hook.
    """
    import logging
    import sys

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# --- Coordinate Transformations ---

# WGS84 Ellipsoid Constants
A = Config.WGS84_A
B = Config.WGS84_B
F = Config.WGS84_F
E2 = 2 * F - F**2  # First eccentricity squared
EP2 = (A**2 - B**2) / (B**2)  # Second eccentricity squared


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).
    Inputs can be scalars or numpy arrays.
    lat, lon in degrees.
    alt in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = A / np.sqrt(1 - E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates (X, Y, Z) to East-North-Up (ENU) relative to a reference point.
    """
    # Reference point in ECEF
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    # Deltas
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix components
    phi = np.radians(lat0)
    lam = np.radians(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotation
    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return e, n, u


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert Geodetic coordinates to ENU relative to a reference point.
    Wrapper for geodetic -> ecef -> enu.
    """
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, lat0, lon0, alt0)


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert ENU coordinates to ECEF relative to a reference point.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)

    phi = np.radians(lat0)
    lam = np.radians(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Inverse Rotation
    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    return x0 + dx, y0 + dy, z0 + dz


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to Geodetic (Lat, Lon, Alt).
    Uses Ferrari's solution.
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * A, p * B)

    lon = np.arctan2(y, x)

    lat = np.arctan2(z + EP2 * B * np.sin(theta) ** 3, p - E2 * A * np.cos(theta) ** 3)

    N = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def enu_to_geodetic(e, n, u, lat0, lon0, alt0):
    """
    Convert ENU coordinates to Geodetic relative to a reference point.
    Wrapper for enu -> ecef -> geodetic.
    """
    x, y, z = enu_to_ecef(e, n, u, lat0, lon0, alt0)
    return ecef_to_geodetic(x, y, z)


# --- Metric Calculation ---


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.
    """
    R = 6371000  # Radius of earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_score(errors):
    """
    Calculates the competition score: Mean of the 50th and 95th percentile errors.
    errors: numpy array of distance errors in meters.
    """
    if len(errors) == 0:
        return 0.0
    p50 = np.percentile(errors, 50)
    p95 = np.percentile(errors, 95)
    return (p50 + p95) / 2.0
