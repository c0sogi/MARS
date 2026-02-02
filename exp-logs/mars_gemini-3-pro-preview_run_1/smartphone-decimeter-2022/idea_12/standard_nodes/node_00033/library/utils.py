import numpy as np
import torch
import random
import os
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def get_radii_of_curvature(lat_rad):
    """
    Calculates the Meridian (Rm) and Prime Vertical (Rn) radii of curvature
    at a given latitude (in radians) using WGS84 constants.

    Args:
        lat_rad (float or np.array): Latitude in radians.

    Returns:
        Rm (float or np.array): Meridian radius of curvature.
        Rn (float or np.array): Prime Vertical radius of curvature.
    """
    a = Config.WGS84_A
    f = Config.WGS84_F
    e2 = 2 * f - f**2  # Square of first eccentricity

    sin_lat = np.sin(lat_rad)
    sin2_lat = sin_lat**2

    # Radius of curvature in the prime vertical
    Rn = a / np.sqrt(1 - e2 * sin2_lat)

    # Radius of curvature in the meridian
    Rm = (a * (1 - e2)) / np.power(1 - e2 * sin2_lat, 1.5)

    return Rm, Rn


def WGS84_to_Meters(lat_base, lon_base, lat_target, lon_target):
    """
    Converts the difference between a base WGS84 coordinate and a target
    WGS84 coordinate into Cartesian offsets (North, East) in meters.

    Args:
        lat_base (float or np.array): Baseline Latitude in degrees.
        lon_base (float or np.array): Baseline Longitude in degrees.
        lat_target (float or np.array): Target Latitude in degrees.
        lon_target (float or np.array): Target Longitude in degrees.

    Returns:
        delta_north (float or np.array): Offset in meters along the North axis.
        delta_east (float or np.array): Offset in meters along the East axis.
    """
    # Convert degrees to radians
    lat_base_rad = np.deg2rad(lat_base)
    lat_target_rad = np.deg2rad(lat_target)
    lon_base_rad = np.deg2rad(lon_base)
    lon_target_rad = np.deg2rad(lon_target)

    # Calculate differences in radians
    d_lat = lat_target_rad - lat_base_rad
    d_lon = lon_target_rad - lon_base_rad

    # Calculate radii of curvature at the base latitude
    Rm, Rn = get_radii_of_curvature(lat_base_rad)

    # Calculate offsets in meters
    delta_north = d_lat * Rm
    delta_east = d_lon * Rn * np.cos(lat_base_rad)

    return delta_north, delta_east


def Meters_to_WGS84(lat_base, lon_base, delta_north, delta_east):
    """
    Converts Cartesian offsets (North, East) in meters from a base WGS84
    coordinate back into a target WGS84 coordinate.

    Args:
        lat_base (float or np.array): Baseline Latitude in degrees.
        lon_base (float or np.array): Baseline Longitude in degrees.
        delta_north (float or np.array): Offset in meters along the North axis.
        delta_east (float or np.array): Offset in meters along the East axis.

    Returns:
        lat_target (float or np.array): Target Latitude in degrees.
        lon_target (float or np.array): Target Longitude in degrees.
    """
    # Convert base latitude to radians
    lat_base_rad = np.deg2rad(lat_base)

    # Calculate radii of curvature at the base latitude
    Rm, Rn = get_radii_of_curvature(lat_base_rad)

    # Calculate differences in radians
    d_lat = delta_north / Rm
    # Avoid division by zero at poles (cos(pi/2) = 0), though unlikely in dataset
    cos_lat = np.cos(lat_base_rad)
    # Add epsilon for numerical stability if needed, but standard range is safe
    d_lon = delta_east / (Rn * cos_lat)

    # Convert differences to degrees and add to base
    lat_target = lat_base + np.rad2deg(d_lat)
    lon_target = lon_base + np.rad2deg(d_lon)

    return lat_target, lon_target
