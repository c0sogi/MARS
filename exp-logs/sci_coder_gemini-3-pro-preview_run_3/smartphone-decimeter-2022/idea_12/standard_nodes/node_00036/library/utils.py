import numpy as np
import pandas as pd

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared


def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 geodetic coordinates to ECEF Cartesian coordinates.

    Args:
        lat: Latitude in degrees (float or numpy array)
        lon: Longitude in degrees (float or numpy array)
        alt: Altitude in meters (float or numpy array)

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_wgs84(x, y, z):
    """
    Convert ECEF Cartesian coordinates to WGS84 geodetic coordinates.
    Uses an iterative method for high precision.

    Args:
        x, y, z: ECEF coordinates in meters

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    p = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)

    # Initial approximation
    lat = np.arctan2(z, p * (1 - WGS84_E2))

    # Iterative solution for latitude
    # Usually converges in 3-4 iterations
    for _ in range(5):
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E2 * N / (N + alt)))

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to local East-North-Up (ENU) coordinates
    relative to a reference point (lat0, lon0, alt0).

    Args:
        x, y, z: Target ECEF coordinates
        lat0, lon0, alt0: Reference WGS84 coordinates

    Returns:
        e, n, u: ENU coordinates in meters
    """
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Rotation matrix multiplication
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to ECEF coordinates
    relative to a reference point (lat0, lon0, alt0).

    Args:
        e, n, u: ENU coordinates in meters
        lat0, lon0, alt0: Reference WGS84 coordinates

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    # Inverse rotation (Transpose of the rotation matrix)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    return x0 + dx, y0 + dy, z0 + dz


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in meters
    """
    R = 6371000.0  # Earth radius in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_competition_metric(df_pred, df_gt):
    """
    Calculate the competition metric: mean of the 50th and 95th percentile
    distance errors, averaged across phones.

    Args:
        df_pred: DataFrame containing ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        df_gt: DataFrame containing ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        float: The competition score
    """
    # Ensure columns match expected names
    pred_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    gt_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]

    # Merge predictions with ground truth on tripId and timestamp
    merged = pd.merge(
        df_pred[pred_cols],
        df_gt[gt_cols],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_pred", "_gt"),
    )

    if len(merged) == 0:
        print(
            "Warning: No overlapping timestamps found between prediction and ground truth."
        )
        return np.nan

    # Calculate distance errors
    merged["dist_error"] = haversine_distance(
        merged["LatitudeDegrees_pred"],
        merged["LongitudeDegrees_pred"],
        merged["LatitudeDegrees_gt"],
        merged["LongitudeDegrees_gt"],
    )

    # Calculate metric per phone (tripId)
    trip_scores = []
    for trip_id, group in merged.groupby("tripId"):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        trip_score = (p50 + p95) / 2
        trip_scores.append(trip_score)

    # Final score is the mean across all phones
    final_score = np.mean(trip_scores)

    return final_score
