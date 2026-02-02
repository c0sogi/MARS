import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of latitude/longitude coordinates.
    Inputs can be scalars or numpy arrays.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    d = R * c
    return d


class WGS84Utils:
    """
    Utilities for converting between WGS84 Geodetic coordinates (Lat, Lon, Alt)
    and ECEF (Earth-Centered, Earth-Fixed) or ENU (East, North, Up) coordinates.
    """

    # WGS84 Ellipsoid constants
    A = 6378137.0  # Semi-major axis
    F = 1 / 298.257223563  # Flattening
    B = A * (1 - F)  # Semi-minor axis
    E2 = 1 - (B**2 / A**2)  # Eccentricity squared

    @staticmethod
    def geodetic_to_ecef(lat, lon, h):
        """
        Convert geodetic coordinates to ECEF.
        lat, lon in degrees, h in meters.
        """
        # Convert to radians
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
        Convert ECEF coordinates to geodetic.
        Returns lat, lon in degrees, h in meters.
        """
        # Ferrari's solution
        a = WGS84Utils.A
        b = WGS84Utils.B

        p = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z * a, p * b)

        e_prime_sq = (a**2 - b**2) / b**2

        lon = np.arctan2(y, x)
        lat = np.arctan2(
            z + e_prime_sq * b * np.sin(theta) ** 3,
            p - WGS84Utils.E2 * a * np.cos(theta) ** 3,
        )

        n = a / np.sqrt(1 - WGS84Utils.E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n

        return np.degrees(lat), np.degrees(lon), h

    @staticmethod
    def ecef_to_enu(x, y, z, lat0, lon0, h0):
        """
        Convert ECEF coordinates to Local Tangent Plane (ENU) centered at (lat0, lon0, h0).
        """
        # Convert reference point to ECEF
        x0, y0, z0 = WGS84Utils.geodetic_to_ecef(lat0, lon0, h0)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)

        sin_lat = np.sin(lat0_rad)
        cos_lat = np.cos(lat0_rad)
        sin_lon = np.sin(lon0_rad)
        cos_lon = np.cos(lon0_rad)

        # Rotation matrix
        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u

    @staticmethod
    def enu_to_ecef(e, n, u, lat0, lon0, h0):
        """
        Convert ENU coordinates centered at (lat0, lon0, h0) to ECEF.
        """
        # Convert reference point to ECEF
        x0, y0, z0 = WGS84Utils.geodetic_to_ecef(lat0, lon0, h0)

        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)

        sin_lat = np.sin(lat0_rad)
        cos_lat = np.cos(lat0_rad)
        sin_lon = np.sin(lon0_rad)
        cos_lon = np.cos(lon0_rad)

        # Inverse rotation
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        x = x0 + dx
        y = y0 + dy
        z = z0 + dz

        return x, y, z

    @classmethod
    def convert_wgs84_to_local_cartesian(cls, lat, lon, alt, ref_lat, ref_lon, ref_alt):
        """
        Wrapper to convert WGS84 (Lat, Lon, Alt) to Local Cartesian (East, North, Up)
        relative to a reference point.
        """
        x, y, z = cls.geodetic_to_ecef(lat, lon, alt)
        e, n, u = cls.ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
        return e, n, u

    @classmethod
    def convert_local_cartesian_to_wgs84(cls, e, n, u, ref_lat, ref_lon, ref_alt):
        """
        Wrapper to convert Local Cartesian (East, North, Up) back to WGS84 (Lat, Lon, Alt)
        relative to a reference point.
        """
        x, y, z = cls.enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
        lat, lon, alt = cls.ecef_to_geodetic(x, y, z)
        return lat, lon, alt
