import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean

# Import from library
from library.config import Config
from library.feature_engineering import FeatureGenerator
from library.model import LGBMResidualModel
from library.kalman_smoothing import KinematicKalmanSmoother, generate_submission
from library.utils import GeoUtils


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
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


def evaluate_validation(val_df, model):
    """
    Runs prediction and smoothing on the validation set, computes the metric,
    and performs failure analysis.
    """
    print("\n" + "=" * 60)
    print(" VALIDATION & FAILURE ANALYSIS")
    print("=" * 60)

    # 1. Predict Residuals
    print("Predicting residuals on validation set...")
    val_preds = model.predict(val_df)

    # Merge necessary columns for smoothing and evaluation
    # We need WLS positions (to add residuals) and Doppler velocities (for smoothing control)
    # And Ground Truth (for metric)
    cols_to_merge = [
        "tripId",
        "UnixTimeMillis",
        "WlsLat",
        "WlsLon",
        "WlsAlt",
        "v_east",
        "v_north",
        "LatitudeDegrees_gt",
        "LongitudeDegrees_gt",  # Targets from metadata
    ]

    # Note: FeatureGenerator adds _gt suffix to targets in _compute_targets,
    # but let's double check column names in val_df.
    # In FeatureGenerator._compute_targets:
    # df_merged = pd.merge(..., suffixes=('', '_gt'))
    # So original 'LatitudeDegrees' from metadata becomes 'LatitudeDegrees_gt' if 'LatitudeDegrees' existed?
    # Actually, FeatureGenerator merges GNSS agg (no lat/lon) with Metadata (has lat/lon).
    # Wait, the metadata has 'LatitudeDegrees'. The merge in _compute_targets merges with metadata subset.
    # If val_df has 'LatitudeDegrees', it's from metadata.
    # Let's check FeatureGenerator again.
    # It merges signal_df (no lat) with wls_df (has WlsLat).
    # Then it merges with gt_df (has LatitudeDegrees).
    # So 'LatitudeDegrees' in val_df is the Ground Truth.

    # Let's verify columns in val_df
    cols_available = val_df.columns.tolist()

    # Prepare dataframe for smoothing
    eval_df = val_preds.copy()

    # Join features back
    eval_df = pd.merge(
        eval_df,
        val_df,
        on=["tripId", "UnixTimeMillis"],
        how="left",
        suffixes=("", "_feat"),
    )

    # 2. Apply Smoothing
    print("Applying Kinematic Kalman Smoothing on Validation...")
    smoother = KinematicKalmanSmoother()
    unique_trips = eval_df["tripId"].unique()

    smoothed_results = []

    for trip in tqdm(unique_trips, desc="Smoothing Val Trips"):
        trip_data = eval_df[eval_df["tripId"] == trip].copy()

        # Sort
        trip_data = trip_data.sort_values("UnixTimeMillis").reset_index(drop=True)

        # A. Convert WLS to ENU
        anchor_lat = trip_data.iloc[0]["WlsLat"]
        anchor_lon = trip_data.iloc[0]["WlsLon"]
        anchor_alt = trip_data.iloc[0]["WlsAlt"]

        x, y, z = GeoUtils.lla_to_ecef(
            trip_data["WlsLat"].values,
            trip_data["WlsLon"].values,
            trip_data["WlsAlt"].values,
        )
        e, n, u = GeoUtils.ecef_to_enu(x, y, z, anchor_lat, anchor_lon, anchor_alt)

        # B. Add Predictions
        trip_data["meas_east"] = e + trip_data["pred_east"]
        trip_data["meas_north"] = n + trip_data["pred_north"]

        # C. Smooth
        s_e, s_n = smoother.smooth_trip(trip_data)

        # D. Convert back to LLA
        sx, sy, sz = GeoUtils.enu_to_ecef(
            s_e, s_n, u, anchor_lat, anchor_lon, anchor_alt
        )
        slat, slon, salt = GeoUtils.ecef_to_lla(sx, sy, sz)

        trip_data["pred_lat"] = slat
        trip_data["pred_lon"] = slon

        smoothed_results.append(trip_data)

    eval_df = pd.concat(smoothed_results, ignore_index=True)

    # 3. Compute Metric
    # Metric: Mean of (50th + 95th percentile errors) per phone

    # Calculate distance error
    # Ground truth columns: 'LatitudeDegrees', 'LongitudeDegrees' (from the merge with val_df which had metadata)
    # Note: In FeatureGenerator, 'LatitudeDegrees' comes from the GT merge.

    eval_df["err_dist"] = calculate_haversine_distance(
        eval_df["pred_lat"],
        eval_df["pred_lon"],
        eval_df["LatitudeDegrees"],
        eval_df["LongitudeDegrees"],
    )

    # Aggregation
    trip_metrics = []
    for trip, group in eval_df.groupby("tripId"):
        p50 = np.percentile(group["err_dist"], 50)
        p95 = np.percentile(group["err_dist"], 95)
        trip_metrics.append((p50 + p95) / 2)

    final_metric = np.mean(trip_metrics)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Correlate error distance with features
    # Select numerical features
    feature_cols = [
        c
        for c in val_df.columns
        if c.startswith("signal_")
        or c.startswith("imu_")
        or c in ["speed", "wls_speed"]
    ]

    # Merge error back to features if not already aligned (they should be)
    analysis_df = eval_df[["err_dist"] + feature_cols].dropna()

    correlations = analysis_df.corr()["err_dist"].sort_values(ascending=False)
    print("Top Correlations with Error Magnitude:")
    print(correlations.head(5))
    print("\nTop Negative Correlations with Error Magnitude:")
    print(correlations.tail(5))

    return final_metric


def main():
    # 1. Feature Generation
    print("Initializing Feature Generator...")
    fg = FeatureGenerator()

    # Generate/Load Train features
    # Using limit=None to use full dataset as per instructions for best performance within time limit
    # The EDA showed 200k rows, which is small.
    print("Generating Training Data...")
    train_df = fg.generate_features("train", load_cached_data=True)

    print("Generating Validation Data...")
    val_df = fg.generate_features("val", load_cached_data=True)

    # 2. Model Training
    print("\n" + "=" * 60)
    print(" MODEL TRAINING")
    print("=" * 60)
    model = LGBMResidualModel()
    model.train(train_df, val_df)

    # 3. Validation & Analysis
    metric = evaluate_validation(val_df, model)

    # 4. Submission
    THRESHOLD = 4.32379283550646

    if metric < THRESHOLD:
        print(
            f"\nMetric {metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Generate Test Features
        print("Generating Test Data...")
        # Ensure test features are generated/cached
        fg.generate_features("test", load_cached_data=True)

        # Generate Submission using library function
        generate_submission(load_cached_features=True)

    else:
        print(
            f"\nMetric {metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
