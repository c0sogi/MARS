import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth surface.

    Args:
        lat1, lon1: Latitude and Longitude of point 1 (in degrees).
        lat2, lon2: Latitude and Longitude of point 2 (in degrees).

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def ecef_to_lla(x, y, z):
    """
    Converts ECEF coordinates to Latitude, Longitude, and Altitude (WGS84).

    Args:
        x, y, z: ECEF coordinates in meters. Can be scalars or NumPy arrays.

    Returns:
        lat, lon, alt: Latitude (degrees), Longitude (degrees), Altitude (meters).
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate altitude (approximate but sufficient for this task)
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def latlon_to_meters(lat_base, lon_base, lat_target, lon_target):
    """
    Converts a target Lat/Lon to metric offsets (East, North) relative to a base Lat/Lon.
    Uses a local flat-earth approximation.

    Args:
        lat_base, lon_base: The reference point (degrees).
        lat_target, lon_target: The point to convert (degrees).

    Returns:
        d_east, d_north: Offsets in meters.
    """
    # Degree to meter conversion factors
    # 1 deg lat ~ 111320 m
    # 1 deg lon ~ 111320 * cos(lat) m

    # Convert latitude difference
    d_lat = lat_target - lat_base
    d_north = d_lat * 111320.0

    # Convert longitude difference (scale by cosine of base latitude)
    d_lon = lon_target - lon_base
    scale_lon = 111320.0 * np.cos(np.radians(lat_base))
    d_east = d_lon * scale_lon

    return d_east, d_north


def meters_to_latlon(lat_base, lon_base, d_east, d_north):
    """
    Converts metric offsets (East, North) back to Lat/Lon relative to a base Lat/Lon.

    Args:
        lat_base, lon_base: The reference point (degrees).
        d_east, d_north: Offsets in meters.

    Returns:
        lat_target, lon_target: The resulting coordinates (degrees).
    """
    # Reverse the conversion
    d_lat = d_north / 111320.0
    lat_target = lat_base + d_lat

    scale_lon = 111320.0 * np.cos(np.radians(lat_base))
    # Avoid division by zero at poles (unlikely in this dataset but good practice)
    # If scale is effectively zero, d_lon is zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        d_lon = np.where(np.abs(scale_lon) > 1e-6, d_east / scale_lon, 0.0)

    lon_target = lon_base + d_lon

    return lat_target, lon_target
