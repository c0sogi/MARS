import pandas as pd
import numpy as np
import os
import sys

# Import from library
from library.config import SUBMISSION_OUTPUT_PATH, SEED
from library.feature_engineering import extract_features
from library.odometry import extract_odometry
from library.model import ResidualRegressor
from library.optimizer import process_optimization
from library.utils import ecef_to_wgs84, wgs84_to_ecef

# Set random seed
np.random.seed(SEED)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
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


def calculate_metric(df):
    """
    Calculates the competition metric:
    Mean of the (50th percentile + 95th percentile) / 2 calculated for each phone (trip).
    """
    # Calculate distance error for each point
    df["dist_err"] = haversine_distance(
        df["LatitudeDegrees_gt"],
        df["LongitudeDegrees_gt"],
        df["LatitudeDegrees"],
        df["LongitudeDegrees"],
    )

    # Group by tripId
    trip_scores = []
    for trip_id, group in df.groupby("tripId"):
        p50 = np.percentile(group["dist_err"], 50)
        p95 = np.percentile(group["dist_err"], 95)
        score = (p50 + p95) / 2
        trip_scores.append(score)

    return np.mean(trip_scores)


def perform_failure_analysis(df_val, df_preds):
    """
    Correlate error magnitude with features.
    """
    print("\n--- Failure Analysis ---")

    # Merge predictions with features and GT
    # df_val contains features and GT
    # df_preds contains predictions

    # Ensure we are working with the same rows
    merged = pd.merge(
        df_val, df_preds, on=["tripId", "UnixTimeMillis"], suffixes=("", "_pred")
    )

    # Calculate error
    # df_preds has optimized predictions as LatitudeDegrees, LongitudeDegrees (from optimizer)
    # df_val has GT LatitudeDegrees.

    lat_gt = merged["LatitudeDegrees"]
    lon_gt = merged["LongitudeDegrees"]
    lat_pred = merged["LatitudeDegrees_pred"]
    lon_pred = merged["LongitudeDegrees_pred"]

    errors = haversine_distance(lat_gt, lon_gt, lat_pred, lon_pred)
    merged["error_meters"] = errors

    # Select numerical features for correlation
    # Exclude IDs and non-feature columns
    feature_cols = [
        c
        for c in merged.columns
        if c
        not in [
            "tripId",
            "UnixTimeMillis",
            "drive_id",
            "phone_name",
            "gt_path",
            "gnss_path",
            "imu_path",
            "LatitudeDegrees",
            "LongitudeDegrees",
            "AltitudeMeters",
            "LatitudeDegrees_pred",
            "LongitudeDegrees_pred",
            "error_meters",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
            "target_E",
            "target_N",
        ]
    ]

    # Filter only numeric
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(merged[c])]

    # Compute correlation
    correlations = (
        merged[feature_cols]
        .corrwith(merged["error_meters"])
        .sort_values(ascending=False)
    )

    print(
        "Top 10 Features positively correlated with Error (High value -> High Error):"
    )
    print(correlations.head(10))
    print(
        "\nTop 10 Features negatively correlated with Error (High value -> Low Error):"
    )
    print(correlations.tail(10))


def main():
    print("Starting End-to-End Pipeline...")

    # 1. Load Data & Extract Features
    print("\n[Step 1] Loading Data and Extracting Features...")
    # Load cached if available, otherwise compute
    df_train = extract_features("train", load_cached_data=True)
    df_val = extract_features("val", load_cached_data=True)
    df_test = extract_features("test", load_cached_data=True)

    # 2. Extract Odometry
    print("\n[Step 2] Extracting Odometry...")
    df_val_odom = extract_odometry("val", load_cached_data=True)
    df_test_odom = extract_odometry("test", load_cached_data=True)

    # 3. Train Model
    print("\n[Step 3] Training Residual Regressor...")
    model = ResidualRegressor()
    model.train(df_train)

    # 4. Validation Inference & Optimization
    print("\n[Step 4] Validation Inference...")
    # Predict residuals -> Absolute positions (Anchors)
    val_anchors = model.predict(df_val)

    # Optimize Trajectory
    print("Optimizing Validation Trajectory...")
    # We do NOT load cached data for optimization results here to ensure we test the current model's output
    val_opt = process_optimization(
        val_anchors, df_val_odom, "val", load_cached_data=False
    )

    # 5. Metric Calculation
    print("\n[Step 5] Calculating Validation Metric...")
    # Prepare dataframe for metric calc
    # df_val has GT: LatitudeDegrees, LongitudeDegrees
    # val_opt has Pred: LatitudeDegrees, LongitudeDegrees

    # Merge to align
    val_eval = pd.merge(
        df_val[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        val_opt,
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_gt", ""),
    )

    score = calculate_metric(val_eval)
    print(f"Final Validation Metric: {score}")

    # 6. Failure Analysis
    perform_failure_analysis(df_val, val_opt)

    # 7. Submission
    THRESHOLD = 4.160290813847215
    if score < THRESHOLD:
        print(f"\n[Step 6] Score {score} < {THRESHOLD}. Generating Submission...")

        # Predict Test Anchors
        test_anchors = model.predict(df_test)

        # Optimize Test Trajectory
        print("Optimizing Test Trajectory...")
        test_opt = process_optimization(
            test_anchors, df_test_odom, "test", load_cached_data=False
        )

        # Format Submission
        # Required columns: tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
        submission = test_opt[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        # Save
        submission.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_OUTPUT_PATH}")
    else:
        print(f"\n[Step 6] Score {score} >= {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    main()
