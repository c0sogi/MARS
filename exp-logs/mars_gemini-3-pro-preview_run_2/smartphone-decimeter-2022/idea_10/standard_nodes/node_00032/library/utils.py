import numpy as np
from library.config import LAT_DEG_TO_METERS

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)  # Second eccentricity squared


def llh_to_ecef(lat, lon, alt):
    """
    Convert Latitude, Longitude, Altitude (LLH) to ECEF coordinates.

    Args:
        lat (np.array or float): Latitude in degrees.
        lon (np.array or float): Longitude in degrees.
        alt (np.array or float): Altitude in meters.

    Returns:
        tuple: (x, y, z) in ECEF meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat_rad) ** 2)

    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E2) + alt) * np.sin(lat_rad)

    return x, y, z


def ecef_to_llh(x, y, z):
    """
    Convert ECEF coordinates to Latitude, Longitude, Altitude (LLH).
    Uses Heikkinen's exact solution or Ferrari's method.

    Args:
        x (np.array or float): X coordinate in meters.
        y (np.array or float): Y coordinate in meters.
        z (np.array or float): Z coordinate in meters.

    Returns:
        tuple: (lat, lon, alt) in degrees and meters.
    """
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    lon = np.arctan2(y, x)

    lat_num = z + WGS84_EP2 * WGS84_B * np.sin(theta) ** 3
    lat_den = p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3
    lat = np.arctan2(lat_num, lat_den)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)

    # Safe division for altitude calculation
    cos_lat = np.cos(lat)
    # Clip cos_lat to avoid division by zero (e.g. at poles)
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)

    alt = p / cos_lat - N

    return np.degrees(lat), np.degrees(lon), alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).

    Returns:
        np.array or float: Distance in meters.
    """
    R = 6371000.0  # Radius of Earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def degrees_to_meters_diff(d_lat_deg, d_lon_deg, ref_lat_deg):
    """
    Convert differences in degrees to meters using local linear approximation.

    Args:
        d_lat_deg: Difference in latitude (degrees).
        d_lon_deg: Difference in longitude (degrees).
        ref_lat_deg: Reference latitude for longitude scaling (degrees).

    Returns:
        tuple: (d_lat_m, d_lon_m)
    """
    d_lat_m = d_lat_deg * LAT_DEG_TO_METERS
    d_lon_m = d_lon_deg * LAT_DEG_TO_METERS * np.cos(np.radians(ref_lat_deg))
    return d_lat_m, d_lon_m


def meters_to_degrees_diff(d_lat_m, d_lon_m, ref_lat_deg):
    """
    Convert differences in meters back to degrees using local linear approximation.

    Args:
        d_lat_m: Difference in latitude (meters).
        d_lon_m: Difference in longitude (meters).
        ref_lat_deg: Reference latitude for longitude scaling (degrees).

    Returns:
        tuple: (d_lat_deg, d_lon_deg)
    """
    d_lat_deg = d_lat_m / LAT_DEG_TO_METERS

    # Avoid division by zero at poles, though unlikely in this dataset
    cos_lat = np.cos(np.radians(ref_lat_deg))
    # Clip cos_lat to avoid extreme values
    cos_lat = np.where(np.abs(cos_lat) < 1e-6, 1e-6, cos_lat)

    d_lon_deg = d_lon_m / (LAT_DEG_TO_METERS * cos_lat)
    return d_lat_deg, d_lon_deg


def calculate_metrics(pred_lat, pred_lon, gt_lat, gt_lon):
    """
    Calculate the mean of the 50th and 95th percentile distance errors.

    Args:
        pred_lat, pred_lon: Predicted coordinates.
        gt_lat, gt_lon: Ground truth coordinates.

    Returns:
        dict: Dictionary containing 'mean_50_95', 'p50', 'p95'.
    """
    errors = haversine_distance(pred_lat, pred_lon, gt_lat, gt_lon)
    p50 = np.percentile(errors, 50)
    p95 = np.percentile(errors, 95)
    mean_50_95 = (p50 + p95) / 2

    return {
        "mean_50_95": mean_50_95,
        "p50": p50,
        "p95": p95,
        "mean_error": np.mean(errors),
    }
