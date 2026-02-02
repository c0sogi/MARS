import os
import sys
import random
import logging
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str = "main"):
    """
    Creates and configures a standard logger that outputs to stdout.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Prevent propagation to root logger to avoid double logging if root is configured
    logger.propagate = False

    return logger


class WGS84:
    """
    Utility class for WGS84 coordinate transformations and flat-earth approximations.
    """

    # WGS84 Ellipsoid Constants
    A = 6378137.0  # Semi-major axis (meters)
    F = 1 / 298.257223563  # Flattening
    E2 = F * (2 - F)  # First eccentricity squared

    @staticmethod
    def lat_lon_to_meters_flat(lat_diff, lon_diff, lat_ref):
        """
        Converts latitude and longitude differences (in degrees) to meters
        using a flat-earth approximation centered at lat_ref.

        This implements the "Simple Element-wise Scaling" strategy.

        Args:
            lat_diff: Latitude difference in degrees (target - reference).
            lon_diff: Longitude difference in degrees (target - reference).
            lat_ref: Reference latitude in degrees for longitude scaling.

        Returns:
            d_north (meters), d_east (meters)
        """
        # Latitude scaling is approximately constant (defined in Config)
        d_north = lat_diff * Config.LAT_SCALE_FACTOR

        # Longitude scaling depends on the cosine of the latitude
        lat_rad = np.radians(lat_ref)
        scale_lon = Config.LAT_SCALE_FACTOR * np.cos(lat_rad)
        d_east = lon_diff * scale_lon

        return d_north, d_east

    @staticmethod
    def meters_to_lat_lon_flat(d_north, d_east, lat_ref):
        """
        Converts distances in meters (North, East) back to latitude and longitude
        differences (in degrees) using a flat-earth approximation.

        Args:
            d_north: Distance north in meters.
            d_east: Distance east in meters.
            lat_ref: Reference latitude in degrees.

        Returns:
            lat_diff (degrees), lon_diff (degrees)
        """
        lat_diff = d_north / Config.LAT_SCALE_FACTOR

        lat_rad = np.radians(lat_ref)
        scale_lon = Config.LAT_SCALE_FACTOR * np.cos(lat_rad)

        # Handle potential division by zero at poles (though unlikely in this dataset)
        # We use a small epsilon or just assume valid range [-89, 89]
        if isinstance(scale_lon, np.ndarray):
            scale_lon[np.abs(scale_lon) < 1e-9] = 1e-9
        elif abs(scale_lon) < 1e-9:
            scale_lon = 1e-9

        lon_diff = d_east / scale_lon

        return lat_diff, lon_diff

    @staticmethod
    def lla_to_ecef(lat, lon, alt):
        """
        Convert Latitude, Longitude, Altitude to ECEF (Earth-Centered, Earth-Fixed) coordinates.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            alt: Altitude in meters.

        Returns:
            x, y, z in meters.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        N = WGS84.A / np.sqrt(1 - WGS84.E2 * sin_lat**2)

        x = (N + alt) * cos_lat * cos_lon
        y = (N + alt) * cos_lat * sin_lon
        z = (N * (1 - WGS84.E2) + alt) * sin_lat

        return x, y, z

    @staticmethod
    def ecef_to_lla(x, y, z):
        """
        Convert ECEF coordinates to Latitude, Longitude, Altitude.

        Args:
            x, y, z: ECEF coordinates in meters.

        Returns:
            lat (degrees), lon (degrees), alt (meters).
        """
        a = WGS84.A
        e = np.sqrt(WGS84.E2)

        b = np.sqrt(a**2 * (1 - e**2))
        ep = np.sqrt((a**2 - b**2) / b**2)

        p = np.sqrt(x**2 + y**2)
        th = np.arctan2(a * z, b * p)

        lon = np.arctan2(y, x)
        lat = np.arctan2(
            (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
        )

        N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N

        lat = np.degrees(lat)
        lon = np.degrees(lon)

        return lat, lon, alt
