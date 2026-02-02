import os
import sys
import logging
import numpy as np


class WGS84Utils:
    """
    Utility class for WGS84 coordinate transformations.
    Converts between Geodetic coordinates (Latitude, Longitude) and
    Local Cartesian offsets (North, East) in meters using local approximation.
    """

    # WGS84 Ellipsoid Constants
    A = 6378137.0  # Semi-major axis (meters)
    F = 1 / 298.257223563  # Flattening
    B = A * (1 - F)  # Semi-minor axis
    E2 = F * (2 - F)  # First eccentricity squared

    @staticmethod
    def degrees_to_meters(lat, lon, lat_ref, lon_ref):
        """
        Converts latitude/longitude to North/East offsets in meters relative to a reference point.

        Args:
            lat: Target latitude (degrees) or array of latitudes.
            lon: Target longitude (degrees) or array of longitudes.
            lat_ref: Reference latitude (degrees).
            lon_ref: Reference longitude (degrees).

        Returns:
            (d_north, d_east): Tuple of offsets in meters.
        """
        # Convert reference latitude to radians for curvature calculations
        lat_rad = np.radians(lat_ref)
        sin_lat = np.sin(lat_rad)

        # Calculate radii of curvature
        # Meridional radius of curvature (M)
        m = (
            WGS84Utils.A
            * (1 - WGS84Utils.E2)
            / np.power(1 - WGS84Utils.E2 * sin_lat**2, 1.5)
        )
        # Prime vertical radius of curvature (N)
        n = WGS84Utils.A / np.sqrt(1 - WGS84Utils.E2 * sin_lat**2)

        # Calculate differences in degrees
        d_lat = lat - lat_ref
        d_lon = lon - lon_ref

        # Convert to meters
        # d_north = d_lat_rad * M
        d_north = d_lat * (np.pi / 180.0) * m
        # d_east = d_lon_rad * N * cos(lat)
        d_east = d_lon * (np.pi / 180.0) * n * np.cos(lat_rad)

        return d_north, d_east

    @staticmethod
    def meters_to_degrees(d_north, d_east, lat_ref):
        """
        Converts North/East offsets in meters to latitude/longitude deltas relative to a reference point.

        Args:
            d_north: Offset in North direction (meters) or array.
            d_east: Offset in East direction (meters) or array.
            lat_ref: Reference latitude (degrees).

        Returns:
            (d_lat, d_lon): Deltas in degrees (add these to ref to get absolute coordinates).
        """
        # Convert reference latitude to radians
        lat_rad = np.radians(lat_ref)
        sin_lat = np.sin(lat_rad)

        # Calculate radii of curvature
        m = (
            WGS84Utils.A
            * (1 - WGS84Utils.E2)
            / np.power(1 - WGS84Utils.E2 * sin_lat**2, 1.5)
        )
        n = WGS84Utils.A / np.sqrt(1 - WGS84Utils.E2 * sin_lat**2)

        # Calculate differences in degrees
        # d_lat = d_north / M * (180/pi)
        d_lat = d_north / ((np.pi / 180.0) * m)
        # d_lon = d_east / (N * cos(lat)) * (180/pi)
        d_lon = d_east / ((np.pi / 180.0) * n * np.cos(lat_rad))

        return d_lat, d_lon


def setup_logger(log_file_path):
    """
    Sets up a logger that writes to both console and a file.

    Args:
        log_file_path: Path to the log file.

    Returns:
        logger: Configured logger instance.
    """
    # Create directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("GNSS_Logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates if setup is called multiple times
    if logger.handlers:
        logger.handlers = []

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
