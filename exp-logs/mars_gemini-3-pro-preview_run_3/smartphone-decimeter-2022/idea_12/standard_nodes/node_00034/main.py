import os
import sys
import pandas as pd
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.training import run_group_kfold
from library.inference import generate_submission
from library.feature_engineering import process_data
from library.data_loader import load_dataset
from library.model import apply_corrections
from library.utils import calculate_competition_metric, haversine_distance

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("========================================================")
    print("   Sector-Aware Physics-Boosted Residual Ensemble")
    print("========================================================")

    # -------------------------------------------------------------------------
    # 1. Train Model
    # -------------------------------------------------------------------------
    # We use debug=False to train on the full dataset for maximum performance.
    # The LightGBM ensemble is efficient enough to run within the time limit.
    print("\n[Step 1/4] Training Model Ensemble...")
    model = run_group_kfold(n_folds=5, load_cached_data=True, debug=False, seed=42)

    # -------------------------------------------------------------------------
    # 2. Validation on Hold-out Set
    # -------------------------------------------------------------------------
    print("\n[Step 2/4] Evaluating on Hold-out Validation Set...")

    # Load features for the validation split (cached if available)
    val_feats, _ = process_data("val", load_cached_data=True)

    # Load raw data to get WLS baselines and Ground Truth for the validation split
    val_gnss, _, val_gt = load_dataset("val", load_cached_data=True)

    # Prepare Feature Matrix X_val
    drop_cols = ["tripId", "UnixTimeMillis"]
    X_val = val_feats.drop(columns=drop_cols)

    # Generate Predictions (ENU Residuals)
    print(f"Predicting on {len(X_val)} validation samples...")
    pred_e, pred_n = model.predict(X_val)

    # Reconstruct Absolute Positions (WLS + Residuals)
    # We need to align predictions with WLS positions from the raw GNSS data
    wls_cols = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Group by timestamp to get unique WLS fix per epoch
    wls_ref = (
        val_gnss.groupby(["tripId", "UnixTimeMillis"])[wls_cols].first().reset_index()
    )

    # Merge WLS reference with prediction index
    val_pred_df = val_feats[["tripId", "UnixTimeMillis"]].copy()
    val_pred_df = pd.merge(
        val_pred_df, wls_ref, on=["tripId", "UnixTimeMillis"], how="left"
    )

    # Apply corrections to WLS
    pred_lat, pred_lon = apply_corrections(val_pred_df, pred_e, pred_n)

    val_pred_df["LatitudeDegrees"] = pred_lat
    val_pred_df["LongitudeDegrees"] = pred_lon

    # Calculate Competition Metric
    # The metric is the mean of the 50th and 95th percentile distance errors
    score = calculate_competition_metric(val_pred_df, val_gt)
    print(f"Final Validation Metric: {score}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Performing Failure Analysis...")

    # Merge predictions with Ground Truth to calculate errors per row
    analysis_df = pd.merge(
        val_pred_df,
        val_gt[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_pred", "_gt"),
    )

    # Calculate Haversine distance error for each point
    analysis_df["error_meters"] = haversine_distance(
        analysis_df["LatitudeDegrees_pred"],
        analysis_df["LongitudeDegrees_pred"],
        analysis_df["LatitudeDegrees_gt"],
        analysis_df["LongitudeDegrees_gt"],
    )

    # Join with input features to find correlations
    analysis_df = pd.merge(analysis_df, val_feats, on=["tripId", "UnixTimeMillis"])

    # Calculate correlations between Error Magnitude and Features
    feature_cols = [
        c for c in val_feats.columns if c not in ["tripId", "UnixTimeMillis"]
    ]

    correlations = {}
    for col in feature_cols:
        if col in analysis_df.columns:
            # Handle potential NaNs in features
            valid_data = analysis_df[[col, "error_meters"]].dropna()
            # Ensure variance exists
            if len(valid_data) > 100 and valid_data[col].std() > 1e-9:
                corr = np.corrcoef(valid_data[col], valid_data["error_meters"])[0, 1]
                correlations[col] = corr

    # Sort and print top correlations
    # Positive correlation: Higher feature value -> Higher error (Risk factor)
    # Negative correlation: Higher feature value -> Lower error (Quality signal)
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Positioning Error:")
    for name, corr in sorted_corr[:10]:
        print(f"  {name:<50} : {corr:+.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Checking Submission Criteria...")
    THRESHOLD = 4.32379283550646

    if score < THRESHOLD:
        print(f"Validation score {score} is below threshold {THRESHOLD}.")
        print("Generating submission file...")
        generate_submission(load_cached_data=True)
    else:
        print(f"Validation score {score} is NOT below threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
