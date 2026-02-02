import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for random number generators in python, numpy, and torch.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


class WGS84Utils:
    """
    Utilities for WGS84 coordinate transformations.
    Supports conversion between Geodetic (Lat, Lon, Alt), ECEF (X, Y, Z),
    and Local Tangent Plane (ENU - East, North, Up).
    """

    # WGS84 ellipsoid constants
    A = 6378137.0  # Semi-major axis
    F = 1 / 298.257223563  # Flattening
    B = A * (1 - F)  # Semi-minor axis
    E2 = 2 * F - F**2  # First eccentricity squared
    EP2 = (A**2 - B**2) / B**2  # Second eccentricity squared

    @staticmethod
    def geodetic_to_ecef(lat, lon, h):
        """
        Convert Geodetic coordinates to ECEF.
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            h: Altitude in meters
        Returns:
            x, y, z: ECEF coordinates in meters
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        n = WGS84Utils.A / np.sqrt(1 - WGS84Utils.E2 * np.sin(lat_rad) ** 2)

        x = (n + h) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (n + h) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (n * (1 - WGS84Utils.E2) + h) * np.sin(lat_rad)

        return x, y, z

    @staticmethod
    def ecef_to_geodetic(x, y, z):
        """
        Convert ECEF coordinates to Geodetic.
        Uses Ferrari's solution.
        Args:
            x, y, z: ECEF coordinates in meters
        Returns:
            lat, lon: Degrees
            h: Meters
        """
        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * WGS84Utils.A, p * WGS84Utils.B)

        lon_rad = np.arctan2(y, x)

        lat_num = z + WGS84Utils.EP2 * WGS84Utils.B * np.sin(theta) ** 3
        lat_den = p - WGS84Utils.E2 * WGS84Utils.A * np.cos(theta) ** 3
        lat_rad = np.arctan2(lat_num, lat_den)

        n = WGS84Utils.A / np.sqrt(1 - WGS84Utils.E2 * np.sin(lat_rad) ** 2)
        h = p / np.cos(lat_rad) - n

        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)

        return lat, lon, h

    @staticmethod
    def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
        """
        Convert ECEF coordinates to Local Tangent Plane (ENU) relative to a reference point.
        Args:
            x, y, z: Target ECEF coordinates
            ref_lat, ref_lon, ref_alt: Reference Geodetic coordinates
        Returns:
            e, n, u: East, North, Up coordinates in meters
        """
        # Convert reference point to ECEF
        ref_x, ref_y, ref_z = WGS84Utils.geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

        dx = x - ref_x
        dy = y - ref_y
        dz = z - ref_z

        # Rotation matrix parameters
        lat_rad = np.radians(ref_lat)
        lon_rad = np.radians(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # ECEF to ENU rotation
        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u

    @staticmethod
    def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
        """
        Convert Local Tangent Plane (ENU) coordinates back to ECEF.
        Args:
            e, n, u: ENU coordinates in meters
            ref_lat, ref_lon, ref_alt: Reference Geodetic coordinates (Origin of ENU)
        Returns:
            x, y, z: ECEF coordinates in meters
        """
        # Convert reference point to ECEF
        ref_x, ref_y, ref_z = WGS84Utils.geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

        lat_rad = np.radians(ref_lat)
        lon_rad = np.radians(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Inverse rotation (Transpose of the rotation matrix)
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        x = ref_x + dx
        y = ref_y + dy
        z = ref_z + dz

        return x, y, z
