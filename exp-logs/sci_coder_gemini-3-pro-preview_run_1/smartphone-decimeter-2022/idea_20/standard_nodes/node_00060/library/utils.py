import os
import random
import numpy as np
import torch

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def wgs84_to_enu(lat, lon, lat_ref, lon_ref):
    """
    Convert WGS84 latitude/longitude to East/North meters relative to a reference point.
    Uses a local tangent plane approximation suitable for small distances (e.g., city scale).

    Args:
        lat: Target latitude(s) in degrees.
        lon: Target longitude(s) in degrees.
        lat_ref: Reference latitude in degrees.
        lon_ref: Reference longitude in degrees.

    Returns:
        east: East offset in meters.
        north: North offset in meters.
    """
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    lat_ref_rad = np.deg2rad(lat_ref)
    lon_rad = np.deg2rad(lon)
    lon_ref_rad = np.deg2rad(lon_ref)

    dlat = lat_rad - lat_ref_rad
    dlon = lon_rad - lon_ref_rad

    # Radius of curvature in the prime vertical (Rn)
    sin_lat_ref = np.sin(lat_ref_rad)
    Rn = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat_ref**2)

    # Radius of curvature in the meridian (Rm)
    Rm = (WGS84_A * (1.0 - WGS84_E2)) / np.power(1.0 - WGS84_E2 * sin_lat_ref**2, 1.5)

    # Calculate East and North offsets
    # East corresponds to longitude change, scaled by Rn and cosine of latitude
    east = dlon * Rn * np.cos(lat_ref_rad)
    # North corresponds to latitude change, scaled by Rm
    north = dlat * Rm

    return east, north


def enu_to_wgs84(east, north, lat_ref, lon_ref):
    """
    Convert East/North meters relative to a reference point back to WGS84 latitude/longitude.
    Inverse of wgs84_to_enu.

    Args:
        east: East offset in meters.
        north: North offset in meters.
        lat_ref: Reference latitude in degrees.
        lon_ref: Reference longitude in degrees.

    Returns:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
    """
    # Convert reference to radians
    lat_ref_rad = np.deg2rad(lat_ref)

    # Radius of curvature in the prime vertical (Rn)
    sin_lat_ref = np.sin(lat_ref_rad)
    Rn = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat_ref**2)

    # Radius of curvature in the meridian (Rm)
    Rm = (WGS84_A * (1.0 - WGS84_E2)) / np.power(1.0 - WGS84_E2 * sin_lat_ref**2, 1.5)

    # Calculate delta radians
    dlat = north / Rm
    dlon = east / (Rn * np.cos(lat_ref_rad))

    # Convert back to degrees and add to reference
    lat = lat_ref + np.rad2deg(dlat)
    lon = lon_ref + np.rad2deg(dlon)

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000.0  # Average radius of earth in meters
    return c * r


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed: Integer seed value.
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
