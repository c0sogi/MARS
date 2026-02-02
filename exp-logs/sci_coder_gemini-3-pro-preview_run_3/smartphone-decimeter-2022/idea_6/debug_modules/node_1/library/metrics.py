import numpy as np
import pandas as pd
from library.utils import haversine_distance


def calculate_distance_errors(
    df: pd.DataFrame,
    pred_lat_col: str = "LatitudeDegrees_pred",
    pred_lon_col: str = "LongitudeDegrees_pred",
    gt_lat_col: str = "LatitudeDegrees",
    gt_lon_col: str = "LongitudeDegrees",
) -> pd.Series:
    """
    Calculate the Haversine distance error (in meters) for each row in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing predicted and ground truth coordinates.
        pred_lat_col (str): Column name for predicted latitude.
        pred_lon_col (str): Column name for predicted longitude.
        gt_lat_col (str): Column name for ground truth latitude.
        gt_lon_col (str): Column name for ground truth longitude.

    Returns:
        pd.Series: Series containing the distance errors in meters.
    """
    return haversine_distance(
        df[pred_lat_col], df[pred_lon_col], df[gt_lat_col], df[gt_lon_col]
    )


def calculate_percentile_errors(
    df: pd.DataFrame,
    pred_lat_col: str = "LatitudeDegrees_pred",
    pred_lon_col: str = "LongitudeDegrees_pred",
    gt_lat_col: str = "LatitudeDegrees",
    gt_lon_col: str = "LongitudeDegrees",
) -> pd.DataFrame:
    """
    Calculate the 50th and 95th percentile distance errors for each trip (phone).

    The competition metric defines the score for a phone as:
    score = (error_50th_percentile + error_95th_percentile) / 2

    Args:
        df (pd.DataFrame): DataFrame containing predicted and ground truth coordinates and 'tripId'.
        pred_lat_col (str): Column name for predicted latitude.
        pred_lon_col (str): Column name for predicted longitude.
        gt_lat_col (str): Column name for ground truth latitude.
        gt_lon_col (str): Column name for ground truth longitude.

    Returns:
        pd.DataFrame: A DataFrame indexed by 'tripId' with columns ['p50', 'p95', 'score'].
    """
    # Ensure we work on a copy to avoid side effects
    local_df = df.copy()

    # Calculate errors if not provided
    if "error_m" not in local_df.columns:
        local_df["error_m"] = calculate_distance_errors(
            local_df, pred_lat_col, pred_lon_col, gt_lat_col, gt_lon_col
        )

    # Define aggregation function
    def get_metrics(g):
        # Linear interpolation is standard for percentiles in numpy
        p50 = np.percentile(g["error_m"], 50)
        p95 = np.percentile(g["error_m"], 95)
        return pd.Series({"p50": p50, "p95": p95, "score": (p50 + p95) / 2})

    # Group by tripId and apply metrics
    if "tripId" not in local_df.columns:
        # Fallback if tripId is missing but drive_id and phone_name exist
        if "drive_id" in local_df.columns and "phone_name" in local_df.columns:
            local_df["tripId"] = (
                local_df["drive_id"].astype(str)
                + "-"
                + local_df["phone_name"].astype(str)
            )
        else:
            raise ValueError("DataFrame must contain 'tripId' column for grouping.")

    scores = local_df.groupby("tripId").apply(get_metrics)

    return scores


def competition_score(
    df: pd.DataFrame,
    pred_lat_col: str = "LatitudeDegrees_pred",
    pred_lon_col: str = "LongitudeDegrees_pred",
    gt_lat_col: str = "LatitudeDegrees",
    gt_lon_col: str = "LongitudeDegrees",
) -> float:
    """
    Calculate the final competition score.

    The final score is the mean of the per-phone scores across the entire dataset.

    Args:
        df (pd.DataFrame): DataFrame containing predicted and ground truth coordinates and 'tripId'.
        pred_lat_col (str): Column name for predicted latitude.
        pred_lon_col (str): Column name for predicted longitude.
        gt_lat_col (str): Column name for ground truth latitude.
        gt_lon_col (str): Column name for ground truth longitude.

    Returns:
        float: The mean competition score.
    """
    scores_df = calculate_percentile_errors(
        df, pred_lat_col, pred_lon_col, gt_lat_col, gt_lon_col
    )

    final_score = scores_df["score"].mean()

    return final_score
