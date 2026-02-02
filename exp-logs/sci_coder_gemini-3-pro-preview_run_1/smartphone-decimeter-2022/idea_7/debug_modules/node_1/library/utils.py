import os
import random
import numpy as np
import torch

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # Eccentricity squared


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def geodetic_to_enu(lat, lon, lat0, lon0):
    """
    Converts geodetic coordinates (lat, lon) to local East-North (EN) coordinates
    relative to a reference point (lat0, lon0) using WGS84 ellipsoid approximation.

    Args:
        lat (np.array or float): Target latitude(s) in degrees.
        lon (np.array or float): Target longitude(s) in degrees.
        lat0 (np.array or float): Reference latitude(s) in degrees.
        lon0 (np.array or float): Reference longitude(s) in degrees.

    Returns:
        tuple: (east, north) in meters.
    """
    # Convert degrees to radians
    lat_rad = np.deg2rad(lat)
    lat0_rad = np.deg2rad(lat0)
    lon_diff_rad = np.deg2rad(lon - lon0)
    lat_diff_rad = np.deg2rad(lat - lat0)

    # Radius of curvature in the prime vertical
    Rn = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat0_rad) ** 2)

    # Radius of curvature in the meridian
    Rm = Rn * (1 - WGS84_E2) / (1 - WGS84_E2 * np.sin(lat0_rad) ** 2)

    # Calculate East and North offsets
    # East corresponds to longitude change, scaled by cosine of latitude
    east = lon_diff_rad * Rn * np.cos(lat0_rad)

    # North corresponds to latitude change
    north = lat_diff_rad * Rm

    return east, north


def enu_to_geodetic(east, north, lat0, lon0):
    """
    Converts local East-North (EN) coordinates back to geodetic coordinates (lat, lon)
    relative to a reference point (lat0, lon0) using WGS84 ellipsoid approximation.

    Args:
        east (np.array or float): East offset(s) in meters.
        north (np.array or float): North offset(s) in meters.
        lat0 (np.array or float): Reference latitude(s) in degrees.
        lon0 (np.array or float): Reference longitude(s) in degrees.

    Returns:
        tuple: (lat, lon) in degrees.
    """
    # Convert degrees to radians
    lat0_rad = np.deg2rad(lat0)

    # Radius of curvature in the prime vertical
    Rn = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat0_rad) ** 2)

    # Radius of curvature in the meridian
    Rm = Rn * (1 - WGS84_E2) / (1 - WGS84_E2 * np.sin(lat0_rad) ** 2)

    # Calculate Latitude and Longitude deltas in radians
    lat_diff_rad = north / Rm
    lon_diff_rad = east / (Rn * np.cos(lat0_rad))

    # Convert deltas to degrees and add to reference
    lat = lat0 + np.rad2deg(lat_diff_rad)
    lon = lon0 + np.rad2deg(lon_diff_rad)

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth's surface.

    Args:
        lat1 (np.array or float): Latitude of first point(s) in degrees.
        lon1 (np.array or float): Longitude of first point(s) in degrees.
        lat2 (np.array or float): Latitude of second point(s) in degrees.
        lon2 (np.array or float): Longitude of second point(s) in degrees.

    Returns:
        np.array or float: Distance in meters.
    """
    R = 6371000  # Earth radius in meters

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delta_phi = np.deg2rad(lat2 - lat1)
    delta_lambda = np.deg2rad(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance
