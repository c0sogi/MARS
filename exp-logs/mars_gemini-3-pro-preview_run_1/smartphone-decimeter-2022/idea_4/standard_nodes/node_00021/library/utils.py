import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Ensure hash consistency
    os.environ["PYTHONHASHSEED"] = str(seed)


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF coordinates (x, y, z) to Geodetic coordinates (lat, lon, alt).
    Uses the WGS84 ellipsoid constants from Config.

    Args:
        x, y, z (float or np.array): ECEF coordinates in meters.

    Returns:
        lat (float or np.array): Latitude in degrees.
        lon (float or np.array): Longitude in degrees.
        alt (float or np.array): Altitude in meters.
    """
    a = Config.WGS84_A
    b = Config.WGS84_B
    f = Config.WGS84_F

    # Derived constants
    e2 = 2 * f - f**2  # Square of first eccentricity
    ep2 = (a**2 - b**2) / b**2  # Square of second eccentricity

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    # Calculate radius of curvature in the prime vertical (N)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    alt = p / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def geodetic_to_enu(lat, lon, ref_lat, ref_lon):
    """
    Convert Geodetic coordinates (lat, lon) to ENU (North, East) residuals relative to a reference point.
    This function calculates the distance in meters along the North and East axes.

    Args:
        lat (float or np.array): Target latitude in degrees.
        lon (float or np.array): Target longitude in degrees.
        ref_lat (float or np.array): Reference latitude in degrees (Baseline).
        ref_lon (float or np.array): Reference longitude in degrees (Baseline).

    Returns:
        north (float or np.array): Distance north in meters.
        east (float or np.array): Distance east in meters.
    """
    a = Config.WGS84_A
    f = Config.WGS84_F
    e2 = 2 * f - f**2

    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)

    # Calculate differences in radians
    d_lat = lat_rad - ref_lat_rad
    d_lon = lon_rad - ref_lon_rad

    # Calculate radii of curvature at the reference latitude
    sin_ref_lat = np.sin(ref_lat_rad)

    # Meridional radius of curvature (M) - North-South direction
    M = a * (1 - e2) / np.power(1 - e2 * sin_ref_lat**2, 1.5)

    # Prime vertical radius of curvature (N) - East-West direction
    N = a / np.sqrt(1 - e2 * sin_ref_lat**2)

    # Calculate offsets in meters
    # North offset is along the meridian
    north = M * d_lat
    # East offset is along the prime vertical, scaled by cos(lat) for longitude convergence
    east = N * np.cos(ref_lat_rad) * d_lon

    return north, east


def enu_to_geodetic(north, east, ref_lat, ref_lon):
    """
    Convert ENU residuals (North, East in meters) back to Geodetic coordinates (lat, lon).

    Args:
        north (float or np.array): Distance north in meters (Predicted residual).
        east (float or np.array): Distance east in meters (Predicted residual).
        ref_lat (float or np.array): Reference latitude in degrees (Baseline).
        ref_lon (float or np.array): Reference longitude in degrees (Baseline).

    Returns:
        lat (float or np.array): Resulting latitude in degrees.
        lon (float or np.array): Resulting longitude in degrees.
    """
    a = Config.WGS84_A
    f = Config.WGS84_F
    e2 = 2 * f - f**2

    # Convert reference latitude to radians
    ref_lat_rad = np.radians(ref_lat)

    # Calculate radii of curvature at the reference latitude
    sin_ref_lat = np.sin(ref_lat_rad)

    # Meridional radius of curvature (M)
    M = a * (1 - e2) / np.power(1 - e2 * sin_ref_lat**2, 1.5)

    # Prime vertical radius of curvature (N)
    N = a / np.sqrt(1 - e2 * sin_ref_lat**2)

    # Calculate angular differences in radians
    d_lat_rad = north / M
    d_lon_rad = east / (N * np.cos(ref_lat_rad))

    # Convert differences to degrees and add to reference
    lat = ref_lat + np.degrees(d_lat_rad)
    lon = ref_lon + np.degrees(d_lon_rad)

    return lat, lon
