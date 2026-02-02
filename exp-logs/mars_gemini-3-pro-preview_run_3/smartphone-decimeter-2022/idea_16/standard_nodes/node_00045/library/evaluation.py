import numpy as np
import pandas as pd
from library.utils import haversine_distance


def calculate_distance_errors(df, lat_col_gt, lon_col_gt, lat_col_pred, lon_col_pred):
    """
    Calculate the haversine distance between ground truth and predicted coordinates.

    Args:
        df (pd.DataFrame): Dataframe containing the coordinates.
        lat_col_gt (str): Column name for ground truth latitude.
        lon_col_gt (str): Column name for ground truth longitude.
        lat_col_pred (str): Column name for predicted latitude.
        lon_col_pred (str): Column name for predicted longitude.

    Returns:
        np.array: Array of distances in meters.
    """
    return haversine_distance(
        df[lat_col_gt], df[lon_col_gt], df[lat_col_pred], df[lon_col_pred]
    )


def compute_percentile_errors(errors, percentiles=[50, 95]):
    """
    Compute specific percentiles of errors.

    Args:
        errors (np.array): Array of error values.
        percentiles (list): List of percentiles to compute (0-100).

    Returns:
        np.array: Calculated percentiles.
    """
    return np.percentile(errors, percentiles)


def score_submission(pred_df, gt_df):
    """
    Score a submission dataframe against a ground truth dataframe using the competition metric.
    Metric: Mean of the (Mean of 50th and 95th percentile errors for each phone).

    Args:
        pred_df (pd.DataFrame): Predictions containing [tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees].
        gt_df (pd.DataFrame): Ground truth containing [tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees].

    Returns:
        float: The competition metric score.
    """
    # Ensure necessary columns exist
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        if col not in pred_df.columns:
            raise ValueError(f"Prediction dataframe missing column: {col}")
        if col not in gt_df.columns:
            raise ValueError(f"Ground truth dataframe missing column: {col}")

    # Prepare subsets for merging
    pred_subset = pred_df[required_cols].rename(
        columns={"LatitudeDegrees": "lat_pred", "LongitudeDegrees": "lon_pred"}
    )

    gt_subset = gt_df[required_cols].rename(
        columns={"LatitudeDegrees": "lat_gt", "LongitudeDegrees": "lon_gt"}
    )

    # Merge on tripId and UnixTimeMillis to align predictions with ground truth
    merged = pd.merge(
        gt_subset, pred_subset, on=["tripId", "UnixTimeMillis"], how="inner"
    )

    if merged.empty:
        print(
            "Warning: No overlapping timestamps found between predictions and ground truth."
        )
        return np.nan

    # Calculate distance errors in meters
    merged["error"] = calculate_distance_errors(
        merged, "lat_gt", "lon_gt", "lat_pred", "lon_pred"
    )

    # Calculate score per phone (tripId)
    phone_scores = []

    # Group by tripId as the metric is averaged per phone
    for trip_id, group in merged.groupby("tripId"):
        errors = group["error"].values
        if len(errors) == 0:
            continue

        p50, p95 = compute_percentile_errors(errors, [50, 95])
        phone_score = (p50 + p95) / 2.0
        phone_scores.append(phone_score)

    if not phone_scores:
        return np.nan

    # Final score is the mean of the averaged percentile errors across all phones
    final_score = np.mean(phone_scores)

    return final_score
