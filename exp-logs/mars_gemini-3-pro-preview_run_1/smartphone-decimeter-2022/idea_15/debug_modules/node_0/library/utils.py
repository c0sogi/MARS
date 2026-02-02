import os
import random
import numpy as np
import torch
import logging
import sys

# ==========================================
# WGS84 Constants
# ==========================================
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_SEMI_MINOR_AXIS = WGS84_SEMI_MAJOR_AXIS * (1.0 - WGS84_FLATTENING)
WGS84_SQUARED_FIRST_ECCENTRICITY = 2 * WGS84_FLATTENING - WGS84_FLATTENING**2


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_file: str = None):
    """
    Creates and configures a simple logger.
    """
    logger = logging.getLogger("GNSS_Logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def wgs84_radii(lat_deg: float):
    """
    Computes the Meridional Radius of Curvature (M) and the
    Prime Vertical Radius of Curvature (N) at a given latitude.

    Args:
        lat_deg: Latitude in degrees.

    Returns:
        M: Meridional radius of curvature (meters).
        N: Prime vertical radius of curvature (meters).
    """
    lat_rad = np.radians(lat_deg)
    sin_lat = np.sin(lat_rad)
    sin2_lat = sin_lat * sin_lat

    # Prime Vertical Radius of Curvature
    N = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1 - WGS84_SQUARED_FIRST_ECCENTRICITY * sin2_lat)

    # Meridional Radius of Curvature
    M = (WGS84_SEMI_MAJOR_AXIS * (1 - WGS84_SQUARED_FIRST_ECCENTRICITY)) / np.power(
        1 - WGS84_SQUARED_FIRST_ECCENTRICITY * sin2_lat, 1.5
    )

    return M, N


def latlon_to_enu(lat, lon, base_lat, base_lon):
    """
    Converts Latitude/Longitude degrees to East/North meters relative to a baseline.
    Uses WGS84 ellipsoid parameters for conversion.

    Args:
        lat: Target latitude(s) in degrees (float or np.array).
        lon: Target longitude(s) in degrees (float or np.array).
        base_lat: Baseline latitude in degrees.
        base_lon: Baseline longitude in degrees.

    Returns:
        delta_east: Distance east in meters.
        delta_north: Distance north in meters.
    """
    M, N = wgs84_radii(base_lat)

    # Calculate deltas in degrees
    d_lat = lat - base_lat
    d_lon = lon - base_lon

    # Convert to radians for cosine term
    base_lat_rad = np.radians(base_lat)

    # Convert to meters
    delta_north = d_lat * (np.pi / 180.0) * M
    delta_east = d_lon * (np.pi / 180.0) * N * np.cos(base_lat_rad)

    return delta_east, delta_north


def enu_to_latlon(delta_east, delta_north, base_lat, base_lon):
    """
    Converts East/North meters relative to a baseline back to Latitude/Longitude degrees.

    Args:
        delta_east: Distance east in meters (float or np.array).
        delta_north: Distance north in meters (float or np.array).
        base_lat: Baseline latitude in degrees.
        base_lon: Baseline longitude in degrees.

    Returns:
        lat: Resulting latitude in degrees.
        lon: Resulting longitude in degrees.
    """
    M, N = wgs84_radii(base_lat)
    base_lat_rad = np.radians(base_lat)

    d_lat = delta_north / ((np.pi / 180.0) * M)
    d_lon = delta_east / ((np.pi / 180.0) * N * np.cos(base_lat_rad))

    lat = base_lat + d_lat
    lon = base_lon + d_lon

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r
