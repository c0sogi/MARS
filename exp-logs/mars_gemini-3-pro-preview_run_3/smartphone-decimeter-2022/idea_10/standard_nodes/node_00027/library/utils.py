import numpy as np
import logging
import sys
import os
from library.config import WORKING_DIR

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def setup_logger(log_file=os.path.join(WORKING_DIR, "log.txt")):
    """
    Sets up a logger that outputs to both console and a file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    # Create directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth.

    Args:
        lat1, lon1: Latitude and Longitude of point 1 (in degrees).
        lat2, lon2: Latitude and Longitude of point 2 (in degrees).

    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Radius of earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    # Protect against floating point errors
    a = np.clip(a, 0.0, 1.0)

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude in meters.

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates to WGS84 Geodetic coordinates using Ferrari's solution.
    Vectorized implementation.

    Args:
        x, y, z: ECEF coordinates in meters.

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m).
    """
    a = WGS84_A
    b = WGS84_B
    e2 = WGS84_E2
    ep2 = (a**2 - b**2) / b**2

    r = np.sqrt(x**2 + y**2)
    E2 = a**2 - b**2
    F = 54 * b**2 * z**2
    G = r**2 + (1 - e2) * z**2 - e2 * E2
    c = (e2**2 * F * r**2) / (G**3)
    s = (1 + c + np.sqrt(c**2 + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G**2)
    Q = np.sqrt(1 + 2 * e2**2 * P)
    ro = -(P * e2 * r) / (1 + Q) + np.sqrt(
        (a**2 / 2) * (1 + 1 / Q)
        - (P * (1 - e2) * z**2) / (Q * (1 + Q))
        - 0.5 * P * r**2
    )
    U = np.sqrt((r - e2 * ro) ** 2 + z**2)
    V = np.sqrt((r - e2 * ro) ** 2 + (1 - e2) * z**2)
    zo = (b**2 * z) / (a * V)

    alt = U * (1 - b**2 / (a * V))
    lat = np.arctan((z + ep2 * zo) / r)
    lon = np.arctan2(y, x)

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
    """
    Convert ECEF coordinates to Local ENU coordinates relative to a reference point.

    Args:
        x, y, z: ECEF coordinates of points to convert.
        ref_lat, ref_lon, ref_alt: Reference point Geodetic coordinates.

    Returns:
        e, n, u: East, North, Up coordinates in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # Rotation matrix elements
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
    """
    Convert Local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u: ENU coordinates in meters.
        ref_lat, ref_lon, ref_alt: Reference point Geodetic coordinates.

    Returns:
        x, y, z: ECEF coordinates in meters.
    """
    # Convert reference point to ECEF
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Inverse rotation (transpose of the matrix used in ecef_to_enu)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz

    return x, y, z
