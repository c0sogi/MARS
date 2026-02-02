import os
import random
import numpy as np
import torch
import logging
import math

# --- WGS84 Constants ---
WGS84_SEMI_MAJOR_AXIS = 6378137.0
WGS84_SEMI_MINOR_AXIS = 6356752.314245
WGS84_FLATTENING = 298.257223563
WGS84_E2 = 6.69437999014e-3  # Square of eccentricity


def fix_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "main", log_file: str = None):
    """
    Creates and returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


class WGS84Utils:
    """
    Utility class for WGS84 coordinate transformations.
    Converts between (Latitude, Longitude) and local (East, North) offsets in meters.
    """

    @staticmethod
    def _get_radii_of_curvature(lat_deg):
        """
        Calculates the Meridional Radius (M) and Prime Vertical Radius (N)
        at a given latitude.
        """
        lat_rad = np.radians(lat_deg)
        sin_lat = np.sin(lat_rad)
        sin_sq = sin_lat**2

        # Prime Vertical Radius
        N = WGS84_SEMI_MAJOR_AXIS / np.sqrt(1 - WGS84_E2 * sin_sq)

        # Meridional Radius
        M = (WGS84_SEMI_MAJOR_AXIS * (1 - WGS84_E2)) / np.power(
            1 - WGS84_E2 * sin_sq, 1.5
        )

        return M, N

    @staticmethod
    def latlon_to_meters_diff(lat_base, lon_base, lat_target, lon_target):
        """
        Calculates the difference in meters (dEast, dNorth) between a base point
        and a target point using local WGS84 approximation.

        Args:
            lat_base, lon_base: Reference point (Degrees)
            lat_target, lon_target: Target point (Degrees)

        Returns:
            dEast, dNorth (Meters)
        """
        M, N = WGS84Utils._get_radii_of_curvature(lat_base)

        d_lat_deg = lat_target - lat_base
        d_lon_deg = lon_target - lon_base

        d_lat_rad = np.radians(d_lat_deg)
        d_lon_rad = np.radians(d_lon_deg)

        # Calculate offsets
        dNorth = M * d_lat_rad
        dEast = N * np.cos(np.radians(lat_base)) * d_lon_rad

        return dEast, dNorth

    @staticmethod
    def meters_to_latlon(lat_base, lon_base, dEast, dNorth):
        """
        Converts local offsets in meters (dEast, dNorth) back to Latitude/Longitude
        relative to a base point.

        Args:
            lat_base, lon_base: Reference point (Degrees)
            dEast, dNorth: Offsets in meters

        Returns:
            new_lat, new_lon (Degrees)
        """
        M, N = WGS84Utils._get_radii_of_curvature(lat_base)

        # Calculate angular differences in radians
        d_lat_rad = dNorth / M
        d_lon_rad = dEast / (N * np.cos(np.radians(lat_base)))

        # Convert to degrees
        new_lat = lat_base + np.degrees(d_lat_rad)
        new_lon = lon_base + np.degrees(d_lon_rad)

        return new_lat, new_lon

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculates the great-circle distance between two points on a sphere.
        Useful for metric calculation validation.
        """
        R = 6371000.0  # Earth radius in meters

        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c
