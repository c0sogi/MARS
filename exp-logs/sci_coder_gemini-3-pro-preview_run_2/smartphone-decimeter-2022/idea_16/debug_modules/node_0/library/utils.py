import numpy as np
import pandas as pd
from library.config import LAT_METERS_PER_DEGREE


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two sets of coordinates.

    Args:
        lat1, lon1: First set of coordinates (scalar or numpy array).
        lat2, lon2: Second set of coordinates (scalar or numpy array).

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def wgs84_to_meters_relative(lat_base, lon_base, lat_target, lon_target):
    """
    Converts WGS84 coordinates to local metric offsets relative to a base point.
    Uses a simple flat-earth approximation suitable for small distances.

    Args:
        lat_base, lon_base: Base coordinates (center of window).
        lat_target, lon_target: Target coordinates.

    Returns:
        Tuple (delta_x, delta_y) in meters.
        delta_x: Easting (meters)
        delta_y: Northing (meters)
    """
    # Latitude difference in meters
    delta_lat_deg = lat_target - lat_base
    delta_y = delta_lat_deg * LAT_METERS_PER_DEGREE

    # Longitude difference in meters (scaled by cosine of latitude)
    # Use the base latitude for the scaling factor
    delta_lon_deg = lon_target - lon_base
    lon_scale = np.cos(np.radians(lat_base))
    delta_x = delta_lon_deg * LAT_METERS_PER_DEGREE * lon_scale

    return delta_x, delta_y


def meters_to_wgs84_relative(lat_base, lon_base, delta_x, delta_y):
    """
    Converts local metric offsets back to WGS84 coordinates relative to a base point.

    Args:
        lat_base, lon_base: Base coordinates.
        delta_x: Offset in Easting (meters).
        delta_y: Offset in Northing (meters).

    Returns:
        Tuple (lat_target, lon_target).
    """
    # Convert Northing to latitude degrees
    delta_lat_deg = delta_y / LAT_METERS_PER_DEGREE
    lat_target = lat_base + delta_lat_deg

    # Convert Easting to longitude degrees
    # Use the base latitude for the scaling factor
    lon_scale = np.cos(np.radians(lat_base))
    # Avoid division by zero close to poles (unlikely in this dataset but good practice)
    lon_scale = np.where(np.abs(lon_scale) < 1e-9, 1e-9, lon_scale)

    delta_lon_deg = delta_x / (LAT_METERS_PER_DEGREE * lon_scale)
    lon_target = lon_base + delta_lon_deg

    return lat_target, lon_target


def calculate_score(predictions_df, ground_truth_df):
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors.

    Args:
        predictions_df: DataFrame with columns ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        ground_truth_df: DataFrame with columns ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        float: The calculated score.
    """
    # Ensure columns exist
    req_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in req_cols:
        if col not in predictions_df.columns:
            raise ValueError(f"predictions_df missing column: {col}")
        if col not in ground_truth_df.columns:
            raise ValueError(f"ground_truth_df missing column: {col}")

    # Merge predictions with ground truth
    # We use inner join to evaluate only on timestamps present in both
    merged = pd.merge(
        ground_truth_df,
        predictions_df,
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_gt", "_pred"),
    )

    if len(merged) == 0:
        print(
            "Warning: No matching timestamps found between predictions and ground truth."
        )
        return np.nan

    # Calculate Haversine distance
    dist = haversine_distance(
        merged["LatitudeDegrees_gt"],
        merged["LongitudeDegrees_gt"],
        merged["LatitudeDegrees_pred"],
        merged["LongitudeDegrees_pred"],
    )

    merged["error_m"] = dist

    # Calculate metric per trip (phone)
    scores = []
    for trip_id, group in merged.groupby("tripId"):
        errors = group["error_m"].values
        if len(errors) == 0:
            continue
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score = (p50 + p95) / 2
        scores.append(score)

    if not scores:
        return np.nan

    # Mean across all phones (trips)
    final_score = np.mean(scores)

    return final_score
