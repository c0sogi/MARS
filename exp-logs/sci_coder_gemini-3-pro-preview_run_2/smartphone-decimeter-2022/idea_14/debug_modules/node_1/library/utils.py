import os
import random
import numpy as np
import torch
import logging
import sys
from library.config import Config

# WGS84 Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
# Approx meters per degree latitude (2 * pi * R / 360)
METERS_PER_DEGREE_LAT = 111319.9


def set_seed(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def get_logger(name="main"):
    """
    Creates and configures a simple logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of point 1 (float or np.array)
        lat2, lon2: Latitude and Longitude of point 2 (float or np.array)

    Returns:
        Distance in meters (float or np.array)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters
    r = 6371000
    return c * r


def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    """
    Converts latitude/longitude to local metric offsets (North, East)
    relative to a reference point using simple element-wise scaling.
    This preserves local geometric shape better than complex projections
    for small windows.

    Args:
        lat: Target latitude(s)
        lon: Target longitude(s)
        ref_lat: Reference latitude (center of window)
        ref_lon: Reference longitude (center of window)

    Returns:
        delta_north (meters), delta_east (meters)
    """
    # Convert inputs to numpy arrays if they aren't already, to handle both scalar and array inputs
    lat = np.array(lat)
    lon = np.array(lon)
    ref_lat = np.array(ref_lat)
    ref_lon = np.array(ref_lon)

    # Difference in degrees
    d_lat = lat - ref_lat
    d_lon = lon - ref_lon

    # Scaling factors
    # Latitude is constant approx 111,320 m/deg
    scale_lat = METERS_PER_DEGREE_LAT

    # Longitude depends on the cosine of the reference latitude
    # We use the reference latitude for the scaling factor of the whole window
    # to maintain linearity within the window.
    scale_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(ref_lat))

    delta_north = d_lat * scale_lat
    delta_east = d_lon * scale_lon

    return delta_north, delta_east


def meters_to_latlon(delta_north, delta_east, ref_lat, ref_lon):
    """
    Converts local metric offsets (North, East) back to latitude/longitude
    relative to a reference point.

    Args:
        delta_north: Distance north in meters
        delta_east: Distance east in meters
        ref_lat: Reference latitude
        ref_lon: Reference longitude

    Returns:
        lat, lon
    """
    delta_north = np.array(delta_north)
    delta_east = np.array(delta_east)
    ref_lat = np.array(ref_lat)
    ref_lon = np.array(ref_lon)

    scale_lat = METERS_PER_DEGREE_LAT
    scale_lon = METERS_PER_DEGREE_LAT * np.cos(np.radians(ref_lat))

    lat = ref_lat + (delta_north / scale_lat)
    lon = ref_lon + (delta_east / scale_lon)

    return lat, lon
