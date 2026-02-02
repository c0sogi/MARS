import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import (
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    MODEL_EAST_PATH,
    MODEL_NORTH_PATH,
    FINAL_SUBMISSION_PATH,
    FEATURE_NAMES,
    LGBM_PARAMS,
    QUANTILES,
    SEED,
)
from library.data_factory import prepare_training_data, prepare_test_data
from library.quantile_regressor import QuantileLGBMWrapper
from library.adaptive_smoother import AdaptiveKalmanFilter
from library.metrics import competition_score
from library.utils import wgs84_to_enu, enu_to_wgs84


def run_pipeline():
    print("--- Starting Pipeline ---")

    # 1. Data Preparation
    print("\n[1] Preparing Data...")
    train_df, val_df = prepare_training_data(load_cached_data=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")

    # 2. Model Training
    print("\n[2] Training Models...")

    # Define features and targets
    X_train = train_df[FEATURE_NAMES]
    y_east_train = train_df["target_east"]
    y_north_train = train_df["target_north"]

    X_val = val_df[FEATURE_NAMES]
    y_east_val = val_df["target_east"]
    y_north_val = val_df["target_north"]

    # Train East Model
    print("Training East Model...")
    model_east = QuantileLGBMWrapper(
        MODEL_EAST_PATH.replace(".txt", ""), params=LGBM_PARAMS, quantiles=QUANTILES
    )
    model_east.fit(
        X_train, y_east_train, X_val, y_east_val, feature_names=FEATURE_NAMES
    )

    # Train North Model
    print("Training North Model...")
    model_north = QuantileLGBMWrapper(
        MODEL_NORTH_PATH.replace(".txt", ""), params=LGBM_PARAMS, quantiles=QUANTILES
    )
    model_north.fit(
        X_train, y_north_train, X_val, y_north_val, feature_names=FEATURE_NAMES
    )

    # 3. Validation & Smoothing
    print("\n[3] Validating and Smoothing...")

    def process_split(df, mode="val"):
        # Predict Residuals and Uncertainty
        print(f"Predicting for {mode}...")
        pred_east_med, pred_east_unc = model_east.predict_with_uncertainty(
            df[FEATURE_NAMES]
        )
        pred_north_med, pred_north_unc = model_north.predict_with_uncertainty(
            df[FEATURE_NAMES]
        )

        # Store raw predictions for analysis
        df["pred_east_res"] = pred_east_med
        df["pred_north_res"] = pred_north_med
        df["unc_east"] = pred_east_unc
        df["unc_north"] = pred_north_unc

        # Apply Smoothing per Trip
        smoothed_lats = []
        smoothed_lons = []

        # We need to preserve the order, so we iterate by the dataframe index or group carefully
        # Grouping by tripId
        trip_ids = df["tripId"].unique()

        # Create a dictionary to map indices to smoothed results to insert back into DF correctly
        results_map = {}  # index -> (lat, lon)

        print(f"Smoothing {len(trip_ids)} trips...")
        for trip in trip_ids:
            trip_mask = df["tripId"] == trip
            trip_df = df[trip_mask].copy()
            indices = trip_df.index.values

            # Sort by time just in case (though data factory should be sorted)
            trip_df = trip_df.sort_values("UnixTimeMillis")
            sorted_indices = trip_df.index.values

            # 1. Convert WLS trajectory to local ENU frame (relative to first point)
            ref_lat = trip_df["WlsLat"].iloc[0]
            ref_lon = trip_df["WlsLon"].iloc[0]
            ref_alt = trip_df["WlsAlt"].iloc[0]

            wls_e, wls_n, _ = wgs84_to_enu(
                trip_df["WlsLat"].values,
                trip_df["WlsLon"].values,
                trip_df["WlsAlt"].values,
                ref_lat,
                ref_lon,
                ref_alt,
            )

            # 2. Add predicted residuals to get "Corrected Observations" in local ENU
            # Note: The model predicts residual relative to instantaneous WLS.
            # Since wls_e/n are relative to ref, and pred is relative to wls, we just add them.
            obs_e = wls_e + trip_df["pred_east_res"].values
            obs_n = wls_n + trip_df["pred_north_res"].values

            observations = np.column_stack((obs_e, obs_n))

            # 3. Get uncertainties
            unc_e = trip_df["unc_east"].values
            unc_n = trip_df["unc_north"].values
            uncertainties = np.column_stack((unc_e, unc_n))

            timestamps = trip_df["UnixTimeMillis"].values

            # 4. Run Kalman Smoother
            kf = AdaptiveKalmanFilter()
            smoothed_enu = kf.smooth(observations, uncertainties, timestamps)

            # 5. Convert back to WGS84
            # We assume Altitude doesn't change much or use WLS alt
            # For 2D smoothing, we just use the WLS altitude for conversion back
            s_lat, s_lon, _ = enu_to_wgs84(
                smoothed_enu[:, 0],
                smoothed_enu[:, 1],
                trip_df["WlsAlt"].values,
                ref_lat,
                ref_lon,
                ref_alt,
            )

            for idx, lat, lon in zip(sorted_indices, s_lat, s_lon):
                results_map[idx] = (lat, lon)

        # Map back to dataframe order
        pred_lats = [results_map[i][0] for i in df.index]
        pred_lons = [results_map[i][1] for i in df.index]

        df["LatitudeDegrees_pred"] = pred_lats
        df["LongitudeDegrees_pred"] = pred_lons

        return df

    val_df = process_split(val_df, mode="val")

    # Calculate Metric
    score = competition_score(val_df)
    print(f"Final Validation Metric: {score}")

    # 4. Failure Analysis
    print("\n[4] Failure Analysis...")
    # Calculate error magnitude
    from library.metrics import calculate_distance_errors

    val_df["error"] = calculate_distance_errors(val_df)

    # Correlation with features
    print("Correlation between Error and Features:")
    corrs = (
        val_df[FEATURE_NAMES + ["error"]].corr()["error"].sort_values(ascending=False)
    )
    print(corrs)

    # Correlation with Uncertainty
    print("\nCorrelation between Error and Predicted Uncertainty:")
    unc_corr_e = val_df["error"].corr(val_df["unc_east"])
    unc_corr_n = val_df["error"].corr(val_df["unc_north"])
    print(f"East Uncertainty Corr: {unc_corr_e:.4f}")
    print(f"North Uncertainty Corr: {unc_corr_n:.4f}")

    # 5. Submission
    THRESHOLD = 4.32379283550646
    if score < THRESHOLD:
        print(f"\n[5] Score {score} < {THRESHOLD}. Generating Submission...")

        test_df = prepare_test_data(load_cached_data=True)
        print(f"Test shape: {test_df.shape}")

        test_df = process_split(test_df, mode="test")

        # Create submission file
        submission = test_df[
            [
                "tripId",
                "UnixTimeMillis",
                "LatitudeDegrees_pred",
                "LongitudeDegrees_pred",
            ]
        ].copy()
        submission.rename(
            columns={
                "LatitudeDegrees_pred": "LatitudeDegrees",
                "LongitudeDegrees_pred": "LongitudeDegrees",
            },
            inplace=True,
        )

        submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {FINAL_SUBMISSION_PATH}")

    else:
        print(f"\n[5] Score {score} >= {THRESHOLD}. Skipping Submission.")


if __name__ == "__main__":
    run_pipeline()
