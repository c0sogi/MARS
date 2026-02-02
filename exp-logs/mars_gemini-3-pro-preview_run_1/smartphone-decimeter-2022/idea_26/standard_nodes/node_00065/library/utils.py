import os
import random
import numpy as np
import torch


class WGS84:
    """
    WGS84 Coordinate System Constants and Utility Functions.
    """

    # Semi-major axis (meters)
    a = 6378137.0
    # Semi-minor axis (meters)
    b = 6356752.314245
    # Flattening
    f = 1 / 298.257223563
    # First eccentricity squared
    e2 = 1 - (b**2 / a**2)

    @staticmethod
    def geodetic_to_ecef(lat, lon, alt):
        """
        Convert Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            alt: Altitude in meters.

        Returns:
            x, y, z: ECEF coordinates in meters.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        N = WGS84.a / np.sqrt(1 - WGS84.e2 * np.sin(lat_rad) ** 2)

        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - WGS84.e2) + alt) * np.sin(lat_rad)

        return x, y, z

    @staticmethod
    def ecef_to_geodetic(x, y, z):
        """
        Convert ECEF coordinates (X, Y, Z) to Geodetic (Latitude, Longitude, Altitude).
        Uses Ferrari's solution for high precision.

        Args:
            x, y, z: ECEF coordinates in meters.

        Returns:
            lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m).
        """
        # Distance from Z-axis
        p = np.sqrt(x**2 + y**2)

        # Longitude
        lon = np.degrees(np.arctan2(y, x))

        # Latitude and Altitude calculation variables
        theta = np.arctan2(z * WGS84.a, p * WGS84.b)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Second eccentricity squared
        ep2 = (WGS84.a**2 - WGS84.b**2) / WGS84.b**2

        lat_rad = np.arctan2(
            z + ep2 * WGS84.b * sin_theta**3, p - WGS84.e2 * WGS84.a * cos_theta**3
        )

        lat = np.degrees(lat_rad)

        N = WGS84.a / np.sqrt(1 - WGS84.e2 * np.sin(lat_rad) ** 2)
        alt = p / np.cos(lat_rad) - N

        return lat, lon, alt

    @staticmethod
    def latlon_to_meters(lat_diff, lon_diff, lat_ref):
        """
        Convert angular offsets (Latitude/Longitude degrees) to meters (North/East).
        Uses radii of curvature at the reference latitude.

        Args:
            lat_diff: Difference in latitude (degrees).
            lon_diff: Difference in longitude (degrees).
            lat_ref: Reference latitude (degrees) for curvature calculation.

        Returns:
            dn: Delta North in meters.
            de: Delta East in meters.
        """
        lat_rad = np.radians(lat_ref)

        # Radius of curvature in the prime vertical
        Rn = WGS84.a / np.sqrt(1 - WGS84.e2 * np.sin(lat_rad) ** 2)

        # Radius of curvature in the meridian
        Rm = (WGS84.a * (1 - WGS84.e2)) / (1 - WGS84.e2 * np.sin(lat_rad) ** 2) ** 1.5

        dn = np.radians(lat_diff) * Rm
        de = np.radians(lon_diff) * Rn * np.cos(lat_rad)

        return dn, de

    @staticmethod
    def meters_to_latlon(dn, de, lat_ref):
        """
        Convert metric offsets (North/East meters) to angular offsets (Latitude/Longitude degrees).
        Uses radii of curvature at the reference latitude.

        Args:
            dn: Delta North in meters.
            de: Delta East in meters.
            lat_ref: Reference latitude (degrees).

        Returns:
            dlat: Difference in latitude (degrees).
            dlon: Difference in longitude (degrees).
        """
        lat_rad = np.radians(lat_ref)

        # Radius of curvature in the prime vertical
        Rn = WGS84.a / np.sqrt(1 - WGS84.e2 * np.sin(lat_rad) ** 2)

        # Radius of curvature in the meridian
        Rm = (WGS84.a * (1 - WGS84.e2)) / (1 - WGS84.e2 * np.sin(lat_rad) ** 2) ** 1.5

        dlat = np.degrees(dn / Rm)
        dlon = np.degrees(de / (Rn * np.cos(lat_rad)))

        return dlat, dlon


def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
