import numpy as np
import pandas as pd
import os

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Square of eccentricity


class GeodeticToEcef:
    """
    Converts Geodetic coordinates (Latitude, Longitude, Altitude) to ECEF (X, Y, Z).
    """

    @staticmethod
    def transform(lat, lon, alt):
        """
        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            alt: Altitude in meters.
        Returns:
            x, y, z: ECEF coordinates in meters.
        """
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

        return x, y, z


class EcefToGeodetic:
    """
    Converts ECEF coordinates (X, Y, Z) to Geodetic (Latitude, Longitude, Altitude).
    Uses an iterative approach for high precision.
    """

    @staticmethod
    def transform(x, y, z):
        """
        Args:
            x, y, z: ECEF coordinates in meters.
        Returns:
            lat, lon, alt: Geodetic coordinates (degrees, degrees, meters).
        """
        r = np.sqrt(x**2 + y**2)

        # Initial guess
        lon = np.arctan2(y, x)
        lat = np.arctan2(z, r * (1 - WGS84_E2))
        h = 0.0

        # Iterative refinement
        for _ in range(5):
            sin_lat = np.sin(lat)
            N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat**2)
            h = r / np.cos(lat) - N
            lat = np.arctan2(z, r * (1 - WGS84_E2 * N / (N + h)))

        return np.rad2deg(lat), np.rad2deg(lon), h


class EcefToEnu:
    """
    Converts ECEF coordinates to Local Tangent Plane (ENU: East, North, Up).
    Requires a reference geodetic point (anchor).
    """

    @staticmethod
    def transform(x, y, z, ref_lat, ref_lon, ref_alt):
        """
        Args:
            x, y, z: Target ECEF coordinates.
            ref_lat, ref_lon, ref_alt: Reference geodetic coordinates (Anchor).
        Returns:
            e, n, u: ENU coordinates in meters relative to the anchor.
        """
        ref_x, ref_y, ref_z = GeodeticToEcef.transform(ref_lat, ref_lon, ref_alt)

        dx = x - ref_x
        dy = y - ref_y
        dz = z - ref_z

        lat_rad = np.deg2rad(ref_lat)
        lon_rad = np.deg2rad(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        e = -sin_lon * dx + cos_lon * dy
        n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

        return e, n, u


class EnuToEcef:
    """
    Converts Local Tangent Plane (ENU) coordinates to ECEF.
    Requires a reference geodetic point (anchor).
    """

    @staticmethod
    def transform(e, n, u, ref_lat, ref_lon, ref_alt):
        """
        Args:
            e, n, u: ENU coordinates in meters.
            ref_lat, ref_lon, ref_alt: Reference geodetic coordinates (Anchor).
        Returns:
            x, y, z: ECEF coordinates in meters.
        """
        ref_x, ref_y, ref_z = GeodeticToEcef.transform(ref_lat, ref_lon, ref_alt)

        lat_rad = np.deg2rad(ref_lat)
        lon_rad = np.deg2rad(ref_lon)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
        dz = cos_lat * n + sin_lat * u

        return ref_x + dx, ref_y + dy, ref_z + dz


class EnuToGeodetic:
    """
    Converts Local Tangent Plane (ENU) coordinates directly to Geodetic.
    Combines EnuToEcef and EcefToGeodetic.
    """

    @staticmethod
    def transform(e, n, u, ref_lat, ref_lon, ref_alt):
        """
        Args:
            e, n, u: ENU coordinates in meters.
            ref_lat, ref_lon, ref_alt: Reference geodetic coordinates (Anchor).
        Returns:
            lat, lon, alt: Geodetic coordinates.
        """
        x, y, z = EnuToEcef.transform(e, n, u, ref_lat, ref_lon, ref_alt)
        return EcefToGeodetic.transform(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great circle distance between two points on the earth.

    Args:
        lat1, lon1: First point coordinates in degrees.
        lat2, lon2: Second point coordinates in degrees.

    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Radius of Earth in meters

    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame containing the metadata.
    """
    path = f"./metadata/{split}_metadata.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    return pd.read_csv(path)


def calculate_competition_metric(df_pred, df_gt):
    """
    Calculates the competition metric: Mean of the 50th and 95th percentile distance errors,
    averaged for each phone.

    Args:
        df_pred: DataFrame containing 'tripId', 'LatitudeDegrees', 'LongitudeDegrees'.
        df_gt: DataFrame containing 'tripId', 'LatitudeDegrees', 'LongitudeDegrees'.

    Returns:
        float: The calculated score.
    """
    # Merge predictions with ground truth on tripId and timestamp if available,
    # or assume they are aligned if passed as such.
    # For robustness, we assume they are aligned by index or passed as a single DF with suffix.
    # Here we assume df_pred and df_gt are aligned row-by-row or we calculate based on input arrays.

    # Calculate Haversine distance
    dists = haversine_distance(
        df_pred["LatitudeDegrees"],
        df_pred["LongitudeDegrees"],
        df_gt["LatitudeDegrees"],
        df_gt["LongitudeDegrees"],
    )

    # Create a temporary dataframe for grouping
    df_temp = pd.DataFrame({"dist": dists, "tripId": df_gt["tripId"]})

    # Extract phone name from tripId (Format: drive_id-phone_name)
    # The last part after the last hyphen is the phone name
    df_temp["phone"] = df_temp["tripId"].apply(lambda x: x.split("-")[-1])

    phone_scores = []
    for phone, group in df_temp.groupby("phone"):
        p50 = np.percentile(group["dist"], 50)
        p95 = np.percentile(group["dist"], 95)
        phone_scores.append((p50 + p95) / 2)

    return np.mean(phone_scores)
