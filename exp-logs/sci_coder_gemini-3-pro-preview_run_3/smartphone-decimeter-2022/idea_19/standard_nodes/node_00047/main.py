import os
import numpy as np
import pandas as pd
import torch
import random
import warnings

from library.config import SEED, SUBMISSION_FILE_PATH, SAMPLE_SUBMISSION_PATH
from library.data_manager import DataManager
from library.model import SplitBandLGBM
from library.optimizer import optimize_dataframe
from library.utils import wgs84_to_ecef

# Suppress warnings
warnings.filterwarnings("ignore")

# Set Random Seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    R = 6371000  # Radius of earth in meters

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_metric(df):
    """
    Compute the mean of the 50th and 95th percentile distance errors.
    """
    # Calculate distance error
    df["dist_error"] = haversine_distance(
        df["LatitudeDegrees"], df["LongitudeDegrees"], df["gt_lat"], df["gt_lon"]
    )

    # Group by phone (tripId)
    # The metric description says "For every phone... averaged for each phone"
    # In this dataset, tripId is unique per phone-drive.
    scores = []
    for trip_id, group in df.groupby("tripId"):
        p50 = np.percentile(group["dist_error"], 50)
        p95 = np.percentile(group["dist_error"], 95)
        scores.append((p50 + p95) / 2)

    return np.mean(scores)


def predict_and_optimize(model, df, dm, load_cached_opt=False):
    """
    Predict residuals using LGBM and refine using Graph Optimization.
    """
    # 1. Extract Features
    X = dm.get_X_y(df)
    # Ensure feature alignment
    for f in model.feature_names:
        if f not in X.columns:
            X[f] = 0
    X = X[model.feature_names]

    # 2. Predict Residuals (Ensemble Average)
    print("  Predicting East residuals...")
    pred_e = np.zeros(len(X))
    for m in model.models_e:
        pred_e += m.predict(X)
    pred_e /= len(model.models_e)

    print("  Predicting North residuals...")
    pred_n = np.zeros(len(X))
    for m in model.models_n:
        pred_n += m.predict(X)
    pred_n /= len(model.models_n)

    # 3. Add predictions to dataframe for optimizer
    df["pred_E"] = pred_e
    df["pred_N"] = pred_n

    # 4. Run Graph Optimization (PyTorch)
    # We set load_cached_data=False to avoid reading stale cache from previous runs
    print("  Running Graph Trajectory Optimization...")
    optimized_df = optimize_dataframe(df, load_cached_data=load_cached_opt)

    return optimized_df


def run_pipeline():
    print("--- Starting Pipeline ---")

    # 1. Data Loading
    dm = DataManager()
    print("Loading Metadata...")
    train_meta, val_meta = dm.load_train_val_metadata()

    print("Preparing Train Dataset...")
    train_df = dm.prepare_dataset(train_meta, "train", load_cached_data=True)

    print("Preparing Validation Dataset...")
    val_df = dm.prepare_dataset(val_meta, "val", load_cached_data=True)

    # 2. Training
    print("\n--- Training Split-Band LGBM ---")
    model = SplitBandLGBM()
    model.train(train_df)

    # 3. Validation
    print("\n--- Validating ---")
    # We need to preserve GT for metric calculation
    # val_df has 'LatitudeDegrees' as GT (from metadata merge)
    # We rename them to avoid overwriting during optimization
    val_df_eval = val_df.copy()
    val_df_eval.rename(
        columns={"LatitudeDegrees": "gt_lat", "LongitudeDegrees": "gt_lon"},
        inplace=True,
    )

    # Predict and Optimize
    val_preds = predict_and_optimize(model, val_df_eval, dm, load_cached_opt=False)

    # Merge predictions back with GT
    # optimize_dataframe returns [tripId, UnixTimeMillis, Lat, Lon]
    # We merge on keys
    val_result = val_df_eval.merge(
        val_preds[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_gt", ""),  # Original had _gt, new has nothing
    )
    # Note: merge might result in gt_lat, gt_lon from left and LatitudeDegrees from right

    metric = calculate_metric(val_result)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    val_result["error"] = haversine_distance(
        val_result["LatitudeDegrees"],
        val_result["LongitudeDegrees"],
        val_result["gt_lat"],
        val_result["gt_lon"],
    )

    # Correlate error with features
    # We need features from val_df_eval
    analysis_df = val_result[["tripId", "UnixTimeMillis", "error"]].merge(
        val_df_eval.drop(columns=["gt_lat", "gt_lon"]), on=["tripId", "UnixTimeMillis"]
    )

    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    correlations = (
        analysis_df[numeric_cols]
        .corrwith(analysis_df["error"])
        .sort_values(ascending=False)
    )

    print("Top 5 Features correlated with Error:")
    print(correlations.head(5))
    print("\nTop 5 Negatively correlated (Good signal indicators):")
    print(correlations.tail(5))

    # 5. Submission
    THRESHOLD = 4.160290813847215
    if metric < THRESHOLD:
        print(f"\nMetric {metric} < {THRESHOLD}. Generating Submission...")

        print("Loading Test Metadata...")
        test_meta = dm.load_test_metadata()

        print("Preparing Test Dataset...")
        test_df = dm.prepare_dataset(test_meta, "test", load_cached_data=True)

        print("Predicting Test Set...")
        test_preds = predict_and_optimize(model, test_df, dm, load_cached_opt=False)

        # Format for submission
        # Sample submission structure
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        # Merge to ensure correct order and rows
        submission = sample[["tripId", "UnixTimeMillis"]].merge(
            test_preds, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Fill missing if any (shouldn't be, but safety first)
        if submission["LatitudeDegrees"].isnull().any():
            print("Warning: NaNs in submission. Interpolating.")
            submission = (
                submission.interpolate().fillna(method="bfill").fillna(method="ffill")
            )

        submission.to_csv(SUBMISSION_FILE_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_FILE_PATH}")
    else:
        print(f"\nMetric {metric} >= {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    run_pipeline()
