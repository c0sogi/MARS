import numpy as np
import pandas as pd
import os


class CoordinateTransformer:
    # WGS84 ellipsoid constants
    A = 6378137.0
    B = 6356752.314245
    F = 1 / 298.257223563
    E2 = F * (2 - F)  # Square of eccentricity

    @staticmethod
    def wgs84_to_ecef(lat, lon, alt):
        """
        Convert WGS84 geodetic coordinates to ECEF.
        lat, lon in degrees. alt in meters.
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        n = CoordinateTransformer.A / np.sqrt(
            1 - CoordinateTransformer.E2 * np.sin(lat_rad) ** 2
        )

        x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (n * (1 - CoordinateTransformer.E2) + alt) * np.sin(lat_rad)

        return x, y, z

    @staticmethod
    def ecef_to_wgs84(x, y, z):
        """
        Convert ECEF coordinates to WGS84.
        Uses Ferrari's method (closed form) for numerical stability.
        """
        # Distance from Z-axis
        r = np.sqrt(x**2 + y**2)

        # Check for poles to avoid division by zero
        # If r is very small, we are at the poles
        # Using a small epsilon for safety, though exact 0 check is usually fine with numpy
        if np.all(r < 1e-9):
            lat = np.sign(z) * 90.0
            lon = np.zeros_like(z)
            alt = np.abs(z) - CoordinateTransformer.B
            return lat, lon, alt

        # Constants
        a = CoordinateTransformer.A
        b = CoordinateTransformer.B
        e2 = CoordinateTransformer.E2
        ep2 = (a**2 - b**2) / b**2

        f = 54 * b**2 * z**2
        g = r**2 + (1 - e2) * z**2 - e2 * (a**2 - b**2)
        c = (e2**2 * f * r**2) / (g**3)
        s = (1 + c + np.sqrt(c**2 + 2 * c)) ** (1 / 3)
        p = f / (3 * (s + 1 / s + 1) ** 2 * g**2)
        q = np.sqrt(1 + 2 * e2**2 * p)
        r0 = -(p * e2 * r) / (1 + q) + np.sqrt(
            0.5 * a**2 * (1 + 1 / q)
            - (p * (1 - e2) * z**2) / (q * (1 + q))
            - 0.5 * p * r**2
        )

        u = np.sqrt((r - e2 * r0) ** 2 + z**2)
        v = np.sqrt((r - e2 * r0) ** 2 + (1 - e2) * z**2)
        z0 = (b**2 * z) / (a * v)

        alt = u * (1 - b**2 / (a * v))
        lat = np.degrees(np.arctan((z + ep2 * z0) / r))
        lon = np.degrees(np.arctan2(y, x))

        return lat, lon, alt

    @staticmethod
    def ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt):
        """
        Convert ECEF coordinates to local ENU coordinates relative to a reference point.
        """
        # Convert reference point to ECEF
        ref_x, ref_y, ref_z = CoordinateTransformer.wgs84_to_ecef(
            ref_lat, ref_lon, ref_alt
        )

        dx = x - ref_x
        dy = y - ref_y
        dz = z - ref_z

        lat_rad = np.radians(ref_lat)
        lon_rad = np.radians(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u

    @staticmethod
    def enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt):
        """
        Convert local ENU coordinates to ECEF relative to a reference point.
        """
        lat_rad = np.radians(ref_lat)
        lon_rad = np.radians(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # Rotation matrix transpose (inverse of ECEF->ENU rotation)
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        ref_x, ref_y, ref_z = CoordinateTransformer.wgs84_to_ecef(
            ref_lat, ref_lon, ref_alt
        )

        x = ref_x + dx
        y = ref_y + dy
        z = ref_z + dz

        return x, y, z


class MetricCalculator:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Radius of earth in meters
        return c * r

    @staticmethod
    def calc_score(df_pred, df_gt):
        """
        Calculate the competition metric: mean of the 50th and 95th percentile distance errors.
        Expects dataframes with 'tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'.
        """
        # Ensure types match for merging
        df_pred = df_pred.copy()
        df_gt = df_gt.copy()

        df_pred["UnixTimeMillis"] = df_pred["UnixTimeMillis"].astype(int)
        df_gt["UnixTimeMillis"] = df_gt["UnixTimeMillis"].astype(int)

        # Merge on tripId and timestamp
        merged = pd.merge(
            df_pred, df_gt, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
        )

        if len(merged) == 0:
            return np.nan

        # Calculate distances
        dists = MetricCalculator.haversine_distance(
            merged["LatitudeDegrees_pred"],
            merged["LongitudeDegrees_pred"],
            merged["LatitudeDegrees_gt"],
            merged["LongitudeDegrees_gt"],
        )

        merged["dist"] = dists

        # Group by tripId to calculate per-phone metrics
        scores = []
        for trip_id, group in merged.groupby("tripId"):
            d = group["dist"].values
            p50 = np.percentile(d, 50)
            p95 = np.percentile(d, 95)
            scores.append((p50 + p95) / 2)

        return np.mean(scores)


class IOHelper:
    CACHE_DIR = "./working/idea_20/"

    @staticmethod
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_parquet(df, filename):
        """
        Saves a dataframe to the cache directory as a parquet file.
        """
        IOHelper.ensure_dir(IOHelper.CACHE_DIR)
        path = os.path.join(IOHelper.CACHE_DIR, filename)
        df.to_parquet(path, index=False)
        print(f"Saved {filename} to cache.")

    @staticmethod
    def load_parquet(filename):
        """
        Loads a parquet file from the cache directory if it exists.
        Returns None if file does not exist.
        """
        path = os.path.join(IOHelper.CACHE_DIR, filename)
        if os.path.exists(path):
            print(f"Loading {filename} from cache.")
            return pd.read_parquet(path)
        return None
