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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the earth (specified in decimal degrees).
    Vectorized version using numpy.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in meters (mean radius)
    r = 6371000
    return c * r


class WGS84Converter:
    """
    Handles coordinate transformations between Geodetic (Lat/Lon) and
    Local Cartesian (East/North) systems using WGS84 ellipsoid constants.
    """

    def __init__(self):
        self.a = Config.WGS84_A
        self.f = Config.WGS84_F
        self.e2 = 2 * self.f - self.f**2

    def deg_to_meters(self, lat, lon, ref_lat, ref_lon):
        """
        Converts target latitude/longitude to offset meters (East, North)
        relative to a reference position (e.g., WLS baseline).

        Args:
            lat, lon: Target positions (degrees).
            ref_lat, ref_lon: Reference/Baseline positions (degrees).

        Returns:
            d_east, d_north: Offsets in meters.
        """
        # Convert to radians
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        ref_lat_rad = np.deg2rad(ref_lat)
        ref_lon_rad = np.deg2rad(ref_lon)

        d_lat = lat_rad - ref_lat_rad
        d_lon = lon_rad - ref_lon_rad

        # Calculate radii of curvature at reference latitude
        sin_lat = np.sin(ref_lat_rad)
        w = np.sqrt(1.0 - self.e2 * sin_lat**2)

        # Meridian radius of curvature
        m = self.a * (1.0 - self.e2) / (w**3)

        # Prime vertical radius of curvature
        n = self.a / w

        d_north = d_lat * m
        d_east = d_lon * n * np.cos(ref_lat_rad)

        return d_east, d_north

    def meters_to_deg(self, d_east, d_north, ref_lat, ref_lon):
        """
        Converts predicted offsets (East, North) in meters back to
        latitude/longitude degrees relative to a reference position.

        Args:
            d_east, d_north: Predicted offsets in meters.
            ref_lat, ref_lon: Reference/Baseline positions (degrees).

        Returns:
            lat, lon: Predicted positions in degrees.
        """
        ref_lat_rad = np.deg2rad(ref_lat)

        sin_lat = np.sin(ref_lat_rad)
        w = np.sqrt(1.0 - self.e2 * sin_lat**2)

        # Meridian radius of curvature
        m = self.a * (1.0 - self.e2) / (w**3)

        # Prime vertical radius of curvature
        n = self.a / w

        d_lat_rad = d_north / m
        d_lon_rad = d_east / (n * np.cos(ref_lat_rad))

        lat = ref_lat + np.rad2deg(d_lat_rad)
        lon = ref_lon + np.rad2deg(d_lon_rad)

        return lat, lon
