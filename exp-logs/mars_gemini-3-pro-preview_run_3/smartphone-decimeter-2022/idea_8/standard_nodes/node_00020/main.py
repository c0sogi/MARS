import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Import library modules
from library.config import Config
from library.data_loader import get_processed_dataset
from library.model_lgbm import train_residual_model, predict_residuals
from library.kalman_filter import RobustKalmanSmoother
from library.evaluation import compute_competition_metric, haversine_distance
from library.coord_utils import enu_to_geodetic

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_feature_columns(df):
    """
    Identify feature columns dynamically, excluding metadata and targets.
    """
    exclude_cols = [
        "tripId",
        "UnixTimeMillis",
        "drive_id",
        "phone_name",
        "gnss_path",
        "imu_path",
        "gt_path",
        "LatitudeDegrees",
        "LongitudeDegrees",
        "wls_lat",
        "wls_lon",
        "wls_alt",
        "target_E",
        "target_N",
    ]
    return [c for c in df.columns if c not in exclude_cols]


def reconstruct_path(df, pred_e, pred_n):
    """
    Reconstruct Geodetic coordinates from WLS baseline and predicted ENU residuals.
    """
    lats, lons = [], []

    # Vectorized reconstruction
    # We use the WLS position as the reference for the local ENU frame at each point
    # This is an approximation (local tangent plane at each step), but valid for small residuals

    wls_lats = df["wls_lat"].values
    wls_lons = df["wls_lon"].values
    wls_alts = df["wls_alt"].values

    # Iterate is slow, but enu_to_geodetic is scalar in the provided utils?
    # Let's check coord_utils.py provided in prompt.
    # It uses numpy functions, so it should support vectorization if inputs are arrays.

    # ecef_to_wgs84 supports arrays.
    # ecef_to_enu supports arrays.
    # enu_to_ecef supports arrays.
    # enu_to_geodetic calls enu_to_ecef then ecef_to_wgs84.
    # So it is vectorized.

    # Note: enu_to_geodetic(e, n, u, lat0, lon0, alt0)
    # We assume u=0 for the residual (we don't predict vertical error)

    pred_u = np.zeros_like(pred_e)

    rec_lat, rec_lon, _ = enu_to_geodetic(
        pred_e, pred_n, pred_u, wls_lats, wls_lons, wls_alts
    )

    return rec_lat, rec_lon


def apply_smoothing(df):
    """
    Apply Robust Kalman Smoothing to each trip in the dataframe.
    """
    smoother = RobustKalmanSmoother()

    trips = df["tripId"].unique()
    smoothed_dfs = []

    # Process each trip
    # Using a simple loop as overhead is low compared to KF operations
    for trip in trips:
        trip_df = df[df["tripId"] == trip].copy()
        trip_df = trip_df.sort_values("UnixTimeMillis")

        # Rename for smoother expectation
        trip_df = trip_df.rename(columns={"pred_lat": "lat", "pred_lon": "lon"})

        # Apply smoother
        try:
            trip_df = smoother.apply(trip_df)
        except Exception as e:
            print(f"Error smoothing trip {trip}: {e}")
            # Fallback to raw predictions if smoothing fails
            pass

        smoothed_dfs.append(trip_df)

    result = pd.concat(smoothed_dfs, ignore_index=True)
    # Rename back
    result = result.rename(
        columns={"lat": "LatitudeDegrees", "lon": "LongitudeDegrees"}
    )
    return result


def main():
    set_seed(Config.SEED)
    print("Starting Velocity-Initialized Physics-Gated Residual Boosting Pipeline...")

    # Cite debug_lesson_3: Invalidate Data Caches When Modifying Data Processing Logic
    # Remove cached files to ensure new data cleaning logic is applied
    for path in [Config.TRAIN_FEATURES_PATH, Config.VAL_FEATURES_PATH]:
        if os.path.exists(path):
            print(f"Removing stale cache: {path}")
            os.remove(path)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading Training Data...")
    train_df = get_processed_dataset("train", load_cached_data=True)

    # Identify features
    feature_cols = get_feature_columns(train_df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # -------------------------------------------------------------------------
    # 2. Train Models
    # -------------------------------------------------------------------------
    print("\n[2/5] Training LightGBM Models...")

    # Train East Model
    models_e, _, score_e = train_residual_model(
        train_df, feature_cols, "target_E", group_col="drive_id"
    )

    # Train North Model
    models_n, _, score_n = train_residual_model(
        train_df, feature_cols, "target_N", group_col="drive_id"
    )

    # Clean up train data to save memory
    del train_df
    import gc

    gc.collect()

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[3/5] Validating...")
    val_df = get_processed_dataset("val", load_cached_data=True)

    # Predict Residuals
    print("Predicting Validation Residuals...")
    val_pred_e = predict_residuals(val_df, feature_cols, models_e)
    val_pred_n = predict_residuals(val_df, feature_cols, models_n)

    # Reconstruct Coordinates (Raw ML)
    print("Reconstructing Geodetic Coordinates...")
    rec_lat, rec_lon = reconstruct_path(val_df, val_pred_e, val_pred_n)

    # Prepare DF for smoothing
    val_res_df = val_df[["tripId", "UnixTimeMillis"]].copy()
    val_res_df["pred_lat"] = rec_lat
    val_res_df["pred_lon"] = rec_lon

    # Apply Smoothing
    print("Applying Robust Kalman Smoothing...")
    val_smoothed = apply_smoothing(val_res_df)

    # Prepare Ground Truth for Evaluation
    # We need to ensure indices match or merge on keys
    val_gt = val_df[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ].copy()

    # Compute Metric
    print("Computing Metric...")
    score = compute_competition_metric(val_smoothed, val_gt)
    print(f"Final Validation Metric: {score}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Merge errors back with features
    analysis_df = val_df.merge(
        val_smoothed[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_gt", "_pred"),
    )

    # Calculate distance error
    analysis_df["error"] = haversine_distance(
        analysis_df["LatitudeDegrees_pred"],
        analysis_df["LongitudeDegrees_pred"],
        analysis_df["LatitudeDegrees_gt"],
        analysis_df["LongitudeDegrees_gt"],
    )

    # Correlate error with features
    correlations = (
        analysis_df[feature_cols + ["error"]]
        .corr()["error"]
        .sort_values(ascending=False)
    )
    print("Top 5 Features correlated with Error:")
    print(correlations.head(6))  # Include error itself

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 4.32379283550646

    if score < THRESHOLD:
        print(
            f"\n[4/5] Validation Score ({score}) < Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_df = get_processed_dataset("test", load_cached_data=True)

        # Predict
        test_pred_e = predict_residuals(test_df, feature_cols, models_e)
        test_pred_n = predict_residuals(test_df, feature_cols, models_n)

        # Reconstruct
        t_rec_lat, t_rec_lon = reconstruct_path(test_df, test_pred_e, test_pred_n)

        test_res_df = test_df[["tripId", "UnixTimeMillis"]].copy()
        test_res_df["pred_lat"] = t_rec_lat
        test_res_df["pred_lon"] = t_rec_lon

        # Smooth
        test_smoothed = apply_smoothing(test_res_df)

        # Save
        submission_path = Config.SUBMISSION_PATH
        test_smoothed.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\n[4/5] Validation Score ({score}) >= Threshold ({THRESHOLD}). Skipping Submission."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
