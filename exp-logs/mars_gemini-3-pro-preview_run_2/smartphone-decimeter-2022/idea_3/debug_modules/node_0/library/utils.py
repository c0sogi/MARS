import numpy as np
import pandas as pd

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Square of eccentricity


def WGS84_to_ECEF(lat, lon, alt):
    """
    Convert WGS84 Latitude, Longitude, Altitude to ECEF X, Y, Z.

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


def ECEF_to_WGS84(x, y, z):
    """
    Convert ECEF X, Y, Z to WGS84 Latitude, Longitude, Altitude.
    Uses Heikkinen's exact solution.

    Args:
        x, y, z: ECEF coordinates in meters (float or numpy array)

    Returns:
        lat, lon, alt: Latitude (deg), Longitude (deg), Altitude (m)
    """
    # Heikkinen's method
    ep2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        z + ep2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def ECEF_to_ENU(x, y, z, lat0, lon0, alt0):
    """
    Convert ECEF coordinates to local ENU coordinates relative to a reference point.

    Args:
        x, y, z: Target ECEF coordinates
        lat0, lon0, alt0: Reference point WGS84 coordinates

    Returns:
        e, n, u: East, North, Up coordinates in meters
    """
    # Reference point in ECEF
    x0, y0, z0 = WGS84_to_ECEF(lat0, lon0, alt0)

    # Deltas
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # Rotation matrix components
    phi = np.radians(lat0)
    lam = np.radians(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Rotation
    e = -sin_lam * dx + cos_lam * dy
    n = -sin_phi * cos_lam * dx - sin_phi * sin_lam * dy + cos_phi * dz
    u = cos_phi * cos_lam * dx + cos_phi * sin_lam * dy + sin_phi * dz

    return e, n, u


def ENU_to_ECEF(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to ECEF coordinates relative to a reference point.

    Args:
        e, n, u: Local ENU coordinates in meters
        lat0, lon0, alt0: Reference point WGS84 coordinates

    Returns:
        x, y, z: ECEF coordinates
    """
    # Reference point in ECEF
    x0, y0, z0 = WGS84_to_ECEF(lat0, lon0, alt0)

    # Rotation matrix components
    phi = np.radians(lat0)
    lam = np.radians(lon0)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Inverse Rotation
    dx = -sin_lam * e - sin_phi * cos_lam * n + cos_phi * cos_lam * u
    dy = cos_lam * e - sin_phi * sin_lam * n + cos_phi * sin_lam * u
    dz = cos_phi * n + sin_phi * u

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def ENU_to_WGS84(e, n, u, lat0, lon0, alt0):
    """
    Convert local ENU coordinates to WGS84 Latitude, Longitude, Altitude.

    Args:
        e, n, u: Local ENU coordinates in meters
        lat0, lon0, alt0: Reference point WGS84 coordinates

    Returns:
        lat, lon, alt: WGS84 coordinates
    """
    x, y, z = ENU_to_ECEF(e, n, u, lat0, lon0, alt0)
    return ECEF_to_WGS84(x, y, z)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        distance: Distance in meters
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


def CompetitionMetric(df_pred, df_gt):
    """
    Calculate the competition metric: mean of the 50th and 95th percentile distance errors.

    Args:
        df_pred: DataFrame with ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        df_gt: DataFrame with ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        score: The competition metric score
    """
    # Ensure required columns exist
    req_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    if not all(col in df_pred.columns for col in req_cols):
        raise ValueError(f"df_pred missing columns. Required: {req_cols}")
    if not all(col in df_gt.columns for col in req_cols):
        raise ValueError(f"df_gt missing columns. Required: {req_cols}")

    # Merge predictions and ground truth
    df = pd.merge(
        df_pred, df_gt, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
    )

    if df.empty:
        print("Warning: No matching timestamps between prediction and ground truth.")
        return np.nan

    # Calculate distance error
    df["dist"] = haversine_distance(
        df["LatitudeDegrees_pred"],
        df["LongitudeDegrees_pred"],
        df["LatitudeDegrees_gt"],
        df["LongitudeDegrees_gt"],
    )

    # Calculate percentiles per trip
    def get_percentiles(group):
        p50 = np.percentile(group["dist"], 50)
        p95 = np.percentile(group["dist"], 95)
        return (p50 + p95) / 2

    trip_scores = df.groupby("tripId").apply(get_percentiles)

    # Mean of trip scores
    return trip_scores.mean()
