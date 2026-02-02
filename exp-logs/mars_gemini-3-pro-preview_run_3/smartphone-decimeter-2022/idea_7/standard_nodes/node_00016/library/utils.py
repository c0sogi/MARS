import numpy as np
import pandas as pd
from library.config import SEED

# =============================================================================
# CONSTANTS (WGS84 Ellipsoid)
# =============================================================================
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)  # Second eccentricity squared


# =============================================================================
# DISTANCE METRICS
# =============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized for numpy arrays.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in meters.
    """
    R = 6371000.0  # Radius of earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# =============================================================================
# COORDINATE TRANSFORMATIONS
# =============================================================================
def wgs84_to_ecef(lat, lon, alt):
    """
    Convert WGS84 Geodetic coordinates to ECEF (Earth-Centered, Earth-Fixed).

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in meters

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
    Convert ECEF coordinates to WGS84 Geodetic coordinates.

    Args:
        x, y, z: ECEF coordinates in meters

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to Local Tangent Plane (ENU) centered at lat0, lon0, alt0.

    Args:
        x, y, z: Target ECEF coordinates
        lat0, lon0, alt0: Anchor point WGS84 coordinates

    Returns:
        e, n, u: East, North, Up coordinates in meters relative to anchor
    """
    # Anchor point in ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Relative vector
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix parameters
    phi = np.radians(lat0)
    lam = np.radians(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # ECEF to ENU rotation
    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return e, n, u


def enu_to_ecef(e, n, u, lat0, lon0, alt0):
    """
    Convert ENU coordinates to ECEF centered at lat0, lon0, alt0.

    Args:
        e, n, u: ENU coordinates in meters
        lat0, lon0, alt0: Anchor point WGS84 coordinates

    Returns:
        x, y, z: ECEF coordinates in meters
    """
    # Anchor point in ECEF
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, alt0)

    # Rotation matrix parameters
    phi = np.radians(lat0)
    lam = np.radians(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Inverse rotation (Transpose of R)
    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    return x0 + dx, y0 + dy, z0 + dz


def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert WGS84 (Lat, Lon, Alt) to Local Tangent Plane (East, North, Up).
    Wrapper around wgs84_to_ecef and ecef_to_enu.
    """
    x, y, z = wgs84_to_ecef(lat, lon, alt)
    return ecef_to_enu(x, y, z, lat0, lon0, alt0)


def enu_to_wgs84(e, n, u, lat0, lon0, alt0):
    """
    Convert Local Tangent Plane (East, North, Up) to WGS84 (Lat, Lon, Alt).
    Wrapper around enu_to_ecef and ecef_to_wgs84.
    """
    x, y, z = enu_to_ecef(e, n, u, lat0, lon0, alt0)
    return ecef_to_wgs84(x, y, z)


# =============================================================================
# SCORING
# =============================================================================
def calculate_score(df_pred, df_gt):
    """
    Calculate the competition metric: Mean of the 50th and 95th percentile distance errors.
    The score is calculated per phone (tripId) and then averaged.

    Args:
        df_pred: DataFrame containing ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        df_gt: DataFrame containing ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        float: The calculated score.
    """
    # Ensure columns exist
    req_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if not all(c in df_pred.columns for c in req_cols):
        raise ValueError(f"df_pred missing required columns: {req_cols}")
    if not all(c in df_gt.columns for c in req_cols):
        raise ValueError(f"df_gt missing required columns: {req_cols}")

    # Merge predictions with ground truth
    merged = pd.merge(
        df_pred, df_gt, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
    )

    if len(merged) == 0:
        return np.nan

    # Calculate Haversine Distance
    dist = haversine_distance(
        merged["LatitudeDegrees_pred"],
        merged["LongitudeDegrees_pred"],
        merged["LatitudeDegrees_gt"],
        merged["LongitudeDegrees_gt"],
    )

    merged["dist"] = dist

    # Calculate score per phone (tripId)
    def agg_score(group):
        p50 = np.percentile(group["dist"], 50)
        p95 = np.percentile(group["dist"], 95)
        return (p50 + p95) / 2

    scores = merged.groupby("tripId").apply(agg_score)

    # Mean across all phones
    final_score = scores.mean()

    return final_score
