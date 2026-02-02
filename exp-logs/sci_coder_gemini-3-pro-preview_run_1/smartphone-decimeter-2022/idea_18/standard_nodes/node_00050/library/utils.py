import os
import math
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
    Calculates the Haversine distance between two points on the Earth.
    Inputs can be numpy arrays or scalars.
    """
    R = 6371000  # Radius of Earth in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


class WGS84:
    """
    Utilities for WGS84 coordinate transformations.
    Uses constants from Config.
    """

    def __init__(self):
        self.a = Config.WGS84_A
        self.b = Config.WGS84_B
        self.f = (self.a - self.b) / self.a
        self.e2 = (self.a**2 - self.b**2) / self.a**2

    def geodetic_to_ecef(self, lat, lon, alt):
        """
        Convert Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).
        Lat/Lon in degrees, Alt in meters.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat_rad) ** 2)

        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.e2) + alt) * np.sin(lat_rad)

        return x, y, z

    def ecef_to_geodetic(self, x, y, z):
        """
        Convert ECEF coordinates (X, Y, Z) to Geodetic (Latitude, Longitude, Altitude).
        Returns Lat/Lon in degrees, Alt in meters.
        """
        ep2 = (self.a**2 - self.b**2) / self.b**2
        p = np.sqrt(x**2 + y**2)
        th = np.arctan2(self.a * z, self.b * p)

        lon = np.arctan2(y, x)
        lat = np.arctan2(
            z + ep2 * self.b * np.sin(th) ** 3, p - self.e2 * self.a * np.cos(th) ** 3
        )

        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N

        # Handle poles (cos(lat) close to 0)
        # For this dataset, we are unlikely to be at poles, but good for robustness
        # If p is small, use z to calculate alt
        pole_mask = p < 1e-10
        if np.any(pole_mask):
            if np.isscalar(pole_mask):
                alt = np.abs(z) - self.b
                lat = np.sign(z) * np.pi / 2
            else:
                alt[pole_mask] = np.abs(z[pole_mask]) - self.b
                lat[pole_mask] = np.sign(z[pole_mask]) * np.pi / 2

        return np.degrees(lat), np.degrees(lon), alt

    def ecef_to_enu(self, x, y, z, ref_lat, ref_lon, ref_alt):
        """
        Convert ECEF coordinates to Local Tangent Plane (ENU) centered at a reference point.
        """
        ref_x, ref_y, ref_z = self.geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

        dx = x - ref_x
        dy = y - ref_y
        dz = z - ref_z

        ref_lat_rad = np.radians(ref_lat)
        ref_lon_rad = np.radians(ref_lon)

        sin_lat = np.sin(ref_lat_rad)
        cos_lat = np.cos(ref_lat_rad)
        sin_lon = np.sin(ref_lon_rad)
        cos_lon = np.cos(ref_lon_rad)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u

    def enu_to_ecef(self, e, n, u, ref_lat, ref_lon, ref_alt):
        """
        Convert Local Tangent Plane (ENU) coordinates to ECEF.
        """
        ref_x, ref_y, ref_z = self.geodetic_to_ecef(ref_lat, ref_lon, ref_alt)

        ref_lat_rad = np.radians(ref_lat)
        ref_lon_rad = np.radians(ref_lon)

        sin_lat = np.sin(ref_lat_rad)
        cos_lat = np.cos(ref_lat_rad)
        sin_lon = np.sin(ref_lon_rad)
        cos_lon = np.cos(ref_lon_rad)

        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        x = ref_x + dx
        y = ref_y + dy
        z = ref_z + dz

        return x, y, z


def calculate_competition_metric(df_pred, df_gt):
    """
    Calculates the competition metric: mean of the 50th and 95th percentile distance errors.

    Args:
        df_pred: DataFrame containing 'phone_name', 'LatitudeDegrees', 'LongitudeDegrees'
        df_gt: DataFrame containing 'phone_name', 'LatitudeDegrees', 'LongitudeDegrees'

    Returns:
        float: The calculated score
    """
    # Ensure alignment
    # Assuming df_pred and df_gt are aligned by index or merged previously
    # For this function, we assume they are aligned arrays or a merged DF passed as args
    # If passed as separate DFs, we compute distance row-wise.

    distances = haversine_distance(
        df_pred["LatitudeDegrees"].values,
        df_pred["LongitudeDegrees"].values,
        df_gt["LatitudeDegrees"].values,
        df_gt["LongitudeDegrees"].values,
    )

    df_scores = df_pred[["phone_name"]].copy()
    df_scores["distance"] = distances

    # Calculate percentiles per phone
    scores = []
    for phone in df_scores["phone_name"].unique():
        phone_dists = df_scores.loc[df_scores["phone_name"] == phone, "distance"]
        p50 = np.percentile(phone_dists, 50)
        p95 = np.percentile(phone_dists, 95)
        scores.append((p50 + p95) / 2)

    return np.mean(scores)
