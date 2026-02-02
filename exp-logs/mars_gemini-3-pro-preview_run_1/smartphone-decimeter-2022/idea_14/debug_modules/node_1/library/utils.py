import os
import random
import numpy as np
import torch
import math
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


class WGS84Utils:
    """
    Utility class for WGS84 coordinate transformations.
    Converts between Latitude/Longitude (Degrees) and North/East (Meters).
    """

    @staticmethod
    def degrees_to_meters(lat, lon, ref_lat, ref_lon):
        """
        Converts latitude/longitude differences to north/east meters relative to a reference point.

        Args:
            lat: Target latitude(s) in degrees (numpy array or scalar).
            lon: Target longitude(s) in degrees (numpy array or scalar).
            ref_lat: Reference latitude(s) in degrees.
            ref_lon: Reference longitude(s) in degrees.

        Returns:
            tuple: (north_meters, east_meters)
        """
        # Convert to radians
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        ref_lat_rad = np.deg2rad(ref_lat)
        ref_lon_rad = np.deg2rad(ref_lon)

        # Calculate deltas
        delta_lat = lat_rad - ref_lat_rad
        delta_lon = lon_rad - ref_lon_rad

        # WGS84 Constants
        a = Config.WGS84_A
        e2 = Config.WGS84_E2

        # Radii of curvature
        # Prime Vertical Radius of Curvature (N)
        sin_lat = np.sin(ref_lat_rad)
        N = a / np.sqrt(1 - e2 * sin_lat**2)

        # Meridian Radius of Curvature (M)
        M = (a * (1 - e2)) / np.power(1 - e2 * sin_lat**2, 1.5)

        # Calculate North/East
        north = M * delta_lat
        east = N * np.cos(ref_lat_rad) * delta_lon

        return north, east

    @staticmethod
    def meters_to_degrees(north, east, ref_lat, ref_lon):
        """
        Converts north/east meters offsets back to latitude/longitude degrees relative to a reference point.

        Args:
            north: North offset(s) in meters (numpy array or scalar).
            east: East offset(s) in meters (numpy array or scalar).
            ref_lat: Reference latitude(s) in degrees.
            ref_lon: Reference longitude(s) in degrees.

        Returns:
            tuple: (latitude_degrees, longitude_degrees)
        """
        # Convert reference to radians
        ref_lat_rad = np.deg2rad(ref_lat)

        # WGS84 Constants
        a = Config.WGS84_A
        e2 = Config.WGS84_E2

        # Radii of curvature at reference latitude
        sin_lat = np.sin(ref_lat_rad)

        # Prime Vertical Radius of Curvature (N)
        N = a / np.sqrt(1 - e2 * sin_lat**2)

        # Meridian Radius of Curvature (M)
        M = (a * (1 - e2)) / np.power(1 - e2 * sin_lat**2, 1.5)

        # Calculate deltas in radians
        delta_lat_rad = north / M
        delta_lon_rad = east / (N * np.cos(ref_lat_rad))

        # Convert deltas to degrees
        delta_lat_deg = np.rad2deg(delta_lat_rad)
        delta_lon_deg = np.rad2deg(delta_lon_rad)

        # Calculate final coordinates
        lat = ref_lat + delta_lat_deg
        lon = ref_lon + delta_lon_deg

        return lat, lon
