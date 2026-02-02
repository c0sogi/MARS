import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points on the Earth's surface.

    Args:
        lat1, lon1: Latitude and Longitude of the first point(s) in degrees.
        lat2, lon2: Latitude and Longitude of the second point(s) in degrees.

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


def compute_competition_metric(df_pred, df_gt):
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors.

    The metric is calculated as follows:
    1. Compute horizontal distance error for each point.
    2. For each phone (tripId), calculate the 50th and 95th percentiles of errors.
    3. Average the 50th and 95th percentiles for each phone.
    4. Calculate the mean of these averaged values across all phones.

    Args:
        df_pred (pd.DataFrame): Predictions DataFrame. Must contain 'tripId', 'UnixTimeMillis',
                                and latitude/longitude columns.
        df_gt (pd.DataFrame): Ground Truth DataFrame. Must contain 'tripId', 'UnixTimeMillis',
                              and latitude/longitude columns.

    Returns:
        float: The computed metric score.
    """
    # Identify latitude and longitude columns dynamically to handle variations
    pred_cols = df_pred.columns
    gt_cols = df_gt.columns

    # Determine column names for predictions
    if "LatitudeDegrees" in pred_cols:
        pred_lat = "LatitudeDegrees"
        pred_lon = "LongitudeDegrees"
    else:
        pred_lat = "lat"
        pred_lon = "lon"

    # Determine column names for ground truth
    if "LatitudeDegrees" in gt_cols:
        gt_lat = "LatitudeDegrees"
        gt_lon = "LongitudeDegrees"
    else:
        gt_lat = "lat"
        gt_lon = "lon"

    # Check if required columns exist
    required_pred = {"tripId", "UnixTimeMillis", pred_lat, pred_lon}
    required_gt = {"tripId", "UnixTimeMillis", gt_lat, gt_lon}

    if not required_pred.issubset(pred_cols):
        raise ValueError(
            f"df_pred missing columns. Required: {required_pred}, Found: {pred_cols}"
        )
    if not required_gt.issubset(gt_cols):
        raise ValueError(
            f"df_gt missing columns. Required: {required_gt}, Found: {gt_cols}"
        )

    # Merge predictions and ground truth
    # We use inner join to score only on timestamps present in both
    merged = pd.merge(
        df_pred, df_gt, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
    )

    if merged.empty:
        print(
            "Warning: No overlapping timestamps found between predictions and ground truth."
        )
        return np.nan

    # Extract coordinates
    lat_pred = merged[f"{pred_lat}_pred"]
    lon_pred = merged[f"{pred_lon}_pred"]
    lat_gt = merged[f"{gt_lat}_gt"]
    lon_gt = merged[f"{gt_lon}_gt"]

    # Calculate distances
    dist_errors = haversine_distance(lat_pred, lon_pred, lat_gt, lon_gt)
    merged["dist_error"] = dist_errors

    # Function to calculate score per trip
    def get_trip_score(group):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        return (p50 + p95) / 2

    # Group by tripId and calculate scores
    trip_scores = merged.groupby("tripId").apply(get_trip_score)

    # Final metric is mean of trip scores
    final_score = trip_scores.mean()

    return final_score
