import numpy as np
import pandas as pd
from library.config import Config


def ecef_to_lla(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) coordinates to
    Latitude, Longitude, and Altitude (LLA) using WGS84 constants.

    Args:
        x, y, z: Arrays or scalars of ECEF coordinates in meters.

    Returns:
        lat, lon, alt: Arrays or scalars of Latitude (degrees),
                       Longitude (degrees), and Altitude (meters).
    """
    # WGS84 ellipsoid constants
    a = 6378137.0
    e = 8.1819190842622e-2

    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)

    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2(
        (z + ep**2 * b * np.sin(th) ** 3), (p - e**2 * a * np.cos(th) ** 3)
    )

    # Calculate altitude (approximate)
    N = a / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return lat, lon, alt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).

    Args:
        lat1, lon1: Latitude and Longitude of point 1.
        lat2, lon2: Latitude and Longitude of point 2.

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    # Convert degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def deg_to_meters(lat_deg, lon_deg, ref_lat):
    """
    Convert differences in degrees to meters using a local flat-earth approximation.

    Args:
        lat_deg: Difference in latitude (degrees).
        lon_deg: Difference in longitude (degrees).
        ref_lat: Reference latitude (degrees) for longitude scaling.

    Returns:
        lat_m: Difference in meters (North).
        lon_m: Difference in meters (East).
    """
    # Constant conversion for latitude
    lat_m = lat_deg * Config.METERS_PER_DEG_LAT

    # Longitude conversion depends on latitude
    scale_lon = Config.METERS_PER_DEG_LAT * np.cos(np.radians(ref_lat))
    lon_m = lon_deg * scale_lon

    return lat_m, lon_m


def meters_to_deg(lat_m, lon_m, ref_lat):
    """
    Convert differences in meters to degrees using a local flat-earth approximation.
    Inverse of deg_to_meters.

    Args:
        lat_m: Difference in meters (North).
        lon_m: Difference in meters (East).
        ref_lat: Reference latitude (degrees) for longitude scaling.

    Returns:
        lat_deg: Difference in latitude (degrees).
        lon_deg: Difference in longitude (degrees).
    """
    lat_deg = lat_m / Config.METERS_PER_DEG_LAT

    scale_lon = Config.METERS_PER_DEG_LAT * np.cos(np.radians(ref_lat))
    # Avoid division by zero at poles, though unlikely in this dataset
    scale_lon = np.where(np.abs(scale_lon) < 1e-6, 1e-6, scale_lon)

    lon_deg = lon_m / scale_lon

    return lat_deg, lon_deg


def calculate_competition_metric(df_pred, df_gt):
    """
    Calculate the competition metric: mean of the 50th and 95th percentile distance errors.

    Args:
        df_pred: DataFrame containing ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        df_gt: DataFrame containing ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        score: The calculated metric.
    """
    # Merge predictions with ground truth on tripId and timestamp
    # Ensure column names match what is expected
    pred_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    gt_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]

    # Rename columns for clarity after merge
    df_merged = pd.merge(
        df_pred[pred_cols],
        df_gt[gt_cols],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_pred", "_gt"),
        how="inner",
    )

    if df_merged.empty:
        print("Warning: No matching timestamps between prediction and ground truth.")
        return np.nan

    # Calculate distance error for each point
    df_merged["error_m"] = haversine_distance(
        df_merged["LatitudeDegrees_pred"],
        df_merged["LongitudeDegrees_pred"],
        df_merged["LatitudeDegrees_gt"],
        df_merged["LongitudeDegrees_gt"],
    )

    # Group by tripId (phone)
    trip_scores = []
    for trip_id, group in df_merged.groupby("tripId"):
        errors = group["error_m"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        trip_score = (p50 + p95) / 2
        trip_scores.append(trip_score)

    # Final score is the mean across all phones
    if not trip_scores:
        return np.nan

    final_score = np.mean(trip_scores)
    return final_score
