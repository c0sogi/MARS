import pandas as pd
import numpy as np
import os
import sys
import random

# Import from provided libraries
from library.config import Config
from library.data_loader import load_metadata
from library.feature_engineering import generate_dataset
from library.model import ResidualRegressor
from library.kalman_filter import apply_kalman_smoothing
from library.evaluation import calculate_metric, haversine_distance


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting runfile.py...")

    # 2. Load Metadata
    print("Loading metadata...")
    train_meta = load_metadata("train")
    val_meta = load_metadata("val")
    test_meta = load_metadata("test")

    # 3. Generate Datasets (Features + Targets)
    # The generate_dataset function handles caching automatically
    print("Generating/Loading Training Data...")
    X_train, y_train = generate_dataset(train_meta, mode="train", load_cached_data=True)

    print("Generating/Loading Validation Data...")
    X_val, y_val = generate_dataset(val_meta, mode="val", load_cached_data=True)

    # 4. Train Model
    print("Initializing and Training Model...")
    model = ResidualRegressor()
    model.fit(X_train, y_train, X_val, y_val)

    # 5. Validation Inference
    print("Predicting on Validation Set...")
    val_res_pred = model.predict(X_val)

    # Reconstruct Absolute Positions from Residuals
    # P_pred = P_wls + Delta_pred
    val_pred_df = X_val[["tripId", "UnixTimeMillis"]].copy()
    val_pred_df["LatitudeDegrees"] = X_val["wls_lat"] + val_res_pred["pred_lat_res"]
    val_pred_df["LongitudeDegrees"] = X_val["wls_lon"] + val_res_pred["pred_lon_res"]

    # 6. Post-Processing (Kalman Smoothing)
    print("Applying Kalman Smoothing to Validation Predictions...")
    val_pred_smoothed = apply_kalman_smoothing(val_pred_df)

    # 7. Evaluation
    print("Calculating Validation Metric...")
    # We need ground truth for validation.
    # We can reconstruct it from X_val wls + y_val target, or merge with original metadata.
    # Reconstructing is safer/easier given the aligned dataframes.
    val_gt_df = X_val[["tripId", "UnixTimeMillis"]].copy()
    val_gt_df["LatitudeDegrees"] = X_val["wls_lat"] + y_val["target_lat"]
    val_gt_df["LongitudeDegrees"] = X_val["wls_lon"] + y_val["target_lon"]

    score = calculate_metric(val_pred_smoothed, val_gt_df)
    print(f"Final Validation Metric: {score}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per row
    # We use the smoothed predictions for analysis to see what remains
    # Merge smoothed preds with GT
    analysis_df = pd.merge(
        val_pred_smoothed,
        val_gt_df,
        on=["tripId", "UnixTimeMillis"],
        suffixes=("_pred", "_gt"),
    )

    analysis_df["error_meters"] = haversine_distance(
        analysis_df["LatitudeDegrees_pred"],
        analysis_df["LongitudeDegrees_pred"],
        analysis_df["LatitudeDegrees_gt"],
        analysis_df["LongitudeDegrees_gt"],
    )

    # Join with features to find correlations
    # Ensure indices align (reset index in generate_dataset ensures simple concat works if order preserved)
    # Safest is merge on keys
    features_only = X_val.drop(columns=["wls_lat", "wls_lon"])
    analysis_full = pd.merge(
        analysis_df, features_only, on=["tripId", "UnixTimeMillis"]
    )

    # Compute correlations
    numeric_cols = analysis_full.select_dtypes(include=[np.number]).columns
    # Exclude non-feature columns
    exclude_cols = [
        "UnixTimeMillis",
        "LatitudeDegrees_pred",
        "LongitudeDegrees_pred",
        "LatitudeDegrees_gt",
        "LongitudeDegrees_gt",
        "error_meters",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    correlations = (
        analysis_full[feature_cols]
        .corrwith(analysis_full["error_meters"])
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

    # 9. Submission Generation
    THRESHOLD = 4.32379283550646
    if score < THRESHOLD:
        print(
            f"\nValidation score {score} passed threshold {THRESHOLD}. Generating submission..."
        )

        print("Generating/Loading Test Data...")
        X_test, _ = generate_dataset(test_meta, mode="test", load_cached_data=True)

        print("Predicting on Test Set...")
        test_res_pred = model.predict(X_test)

        # Reconstruct
        test_pred_df = X_test[["tripId", "UnixTimeMillis"]].copy()
        test_pred_df["LatitudeDegrees"] = (
            X_test["wls_lat"] + test_res_pred["pred_lat_res"]
        )
        test_pred_df["LongitudeDegrees"] = (
            X_test["wls_lon"] + test_res_pred["pred_lon_res"]
        )

        print("Applying Kalman Smoothing to Test Predictions...")
        test_pred_smoothed = apply_kalman_smoothing(test_pred_df)

        # Format for submission
        # Ensure columns are correct
        submission = test_pred_smoothed[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        print(f"Saving submission to {Config.SUBMISSION_FILE}...")
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print("Submission saved successfully.")

    else:
        print(
            f"\nValidation score {score} did not meet threshold {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
