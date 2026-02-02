import numpy as np
import pandas as pd
import logging
import sys


def get_logger(name: str = "GNSS_Logger") -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Calculates the Haversine distance between two sets of coordinates.

    Args:
        lat1 (np.ndarray): Latitude of first point(s) in degrees.
        lon1 (np.ndarray): Longitude of first point(s) in degrees.
        lat2 (np.ndarray): Latitude of second point(s) in degrees.
        lon2 (np.ndarray): Longitude of second point(s) in degrees.

    Returns:
        np.ndarray: Distance in meters.
    """
    R = 6371000  # Radius of Earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calc_score(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """
    Computes the competition metric: mean of the 50th and 95th percentile distance errors.

    The metric is calculated for each phone (tripId) and then averaged across all phones.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predictions.
                                Must have columns: ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        gt_df (pd.DataFrame): DataFrame containing ground truth.
                              Must have columns: ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']

    Returns:
        float: The calculated score (lower is better).
    """
    # Ensure necessary columns exist
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        if col not in pred_df.columns:
            raise ValueError(f"Prediction DataFrame missing column: {col}")
        if col not in gt_df.columns:
            raise ValueError(f"Ground Truth DataFrame missing column: {col}")

    # Merge predictions with ground truth on tripId and timestamp
    # Suffixes _pred and _gt are used to distinguish coordinates
    merged_df = pd.merge(
        pred_df, gt_df, on=["tripId", "UnixTimeMillis"], suffixes=("_pred", "_gt")
    )

    if merged_df.empty:
        raise ValueError(
            "No overlapping data found between predictions and ground truth."
        )

    # Calculate Haversine distance for each point
    merged_df["distance_error"] = haversine_distance(
        merged_df["LatitudeDegrees_pred"].values,
        merged_df["LongitudeDegrees_pred"].values,
        merged_df["LatitudeDegrees_gt"].values,
        merged_df["LongitudeDegrees_gt"].values,
    )

    # Calculate score per tripId (phone)
    def evaluate_trip(group):
        errors = group["distance_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        return (p50 + p95) / 2

    trip_scores = merged_df.groupby("tripId").apply(evaluate_trip)

    # Final score is the mean of trip scores
    final_score = trip_scores.mean()

    return final_score
