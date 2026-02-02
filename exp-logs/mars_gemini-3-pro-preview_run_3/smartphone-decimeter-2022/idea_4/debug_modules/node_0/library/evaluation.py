import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance between two points on the earth.
    Vectorized implementation using numpy.

    Args:
        lat1, lon1: Latitude and Longitude of point 1 (decimal degrees).
        lat2, lon2: Latitude and Longitude of point 2 (decimal degrees).

    Returns:
        Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2

    # Arc tangent of two numbers
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_metric(df_pred, df_gt):
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors.

    The metric is calculated as follows:
    1. For each phone (tripId), calculate the horizontal distance error for every timestamp.
    2. Compute the 50th (median) and 95th percentiles of these errors.
    3. Average the 50th and 95th percentiles for that phone.
    4. The final score is the mean of these averaged values across all phones.

    Args:
        df_pred (pd.DataFrame): DataFrame containing predicted locations.
                                Must contain ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'].
        df_gt (pd.DataFrame): DataFrame containing ground truth locations.
                              Must contain ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees'].

    Returns:
        float: The calculated metric score.
    """
    # Define required columns
    req_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]

    # Validate columns
    for col in req_cols:
        if col not in df_pred.columns:
            raise KeyError(f"Prediction DataFrame missing required column: {col}")
        if col not in df_gt.columns:
            raise KeyError(f"Ground Truth DataFrame missing required column: {col}")

    # Rename columns to avoid suffixes during merge and for clarity
    pred_subset = df_pred[req_cols].rename(
        columns={"LatitudeDegrees": "lat_pred", "LongitudeDegrees": "lon_pred"}
    )

    gt_subset = df_gt[req_cols].rename(
        columns={"LatitudeDegrees": "lat_gt", "LongitudeDegrees": "lon_gt"}
    )

    # Merge predictions and ground truth on tripId and UnixTimeMillis
    # Use inner join to ensure we only evaluate on timestamps present in both
    merged = pd.merge(
        gt_subset, pred_subset, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    if merged.empty:
        print(
            "Warning: No overlapping timestamps found between predictions and ground truth."
        )
        return np.nan

    # Calculate Haversine distance for each row
    merged["dist_error"] = haversine_distance(
        merged["lat_pred"], merged["lon_pred"], merged["lat_gt"], merged["lon_gt"]
    )

    # Function to calculate the specific metric for a single trip
    def trip_metric(group):
        errors = group["dist_error"].values
        if len(errors) == 0:
            return np.nan
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        return (p50 + p95) / 2.0

    # Apply metric calculation per trip
    trip_scores = merged.groupby("tripId").apply(trip_metric)

    # Final metric is the mean of trip scores
    score = trip_scores.mean()

    return score
