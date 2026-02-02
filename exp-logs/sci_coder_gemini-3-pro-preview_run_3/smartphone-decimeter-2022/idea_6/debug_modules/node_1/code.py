import os
import numpy as np
import pandas as pd
import warnings

# Import from provided libraries
from library.config import FEATURE_NAMES, LGBM_PARAMS, WORKING_DIR
from library.data_factory import load_metadata, _process_trip
from library.quantile_regressor import QuantileLGBMWrapper
from library.adaptive_smoother import AdaptiveKalmanFilter
from library.utils import enu_to_wgs84, wgs84_to_enu
from library.metrics import competition_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Data Loading (Subset)
    # -------------------------------------------------------------------------
    print("\n[1] Loading Metadata and Generating Features...")

    # Load train metadata
    meta_df = load_metadata("train")

    # Select a small subset of drives for demonstration speed
    # We pick 2 unique drives to have a distinct train and validation set
    unique_drives = meta_df["drive_id"].unique()
    if len(unique_drives) < 2:
        raise ValueError("Not enough unique drives in metadata for this demo.")

    selected_drives = unique_drives[:2]
    subset_meta = meta_df[meta_df["drive_id"].isin(selected_drives)].copy()

    # Process trips to get features and targets
    # Group by tripId to process each trip once
    unique_trips = subset_meta.drop_duplicates(subset=["tripId"])

    processed_dfs = []
    print(f"Processing {len(unique_trips)} trips from drives: {selected_drives}")

    for _, row in unique_trips.iterrows():
        # _process_trip loads GNSS/IMU, merges them, and computes ENU targets
        df_trip = _process_trip(row, include_gt=True)
        if df_trip is not None:
            processed_dfs.append(df_trip)

    if not processed_dfs:
        raise RuntimeError("No data could be processed.")

    full_df = pd.concat(processed_dfs, ignore_index=True)
    print(f"Total rows processed: {len(full_df)}")

    # -------------------------------------------------------------------------
    # 2. Train/Validation Split
    # -------------------------------------------------------------------------
    print("\n[2] Splitting Data...")

    # Split by drive_id to simulate unseen test data
    train_drive = selected_drives[0]
    val_drive = selected_drives[1]

    train_df = full_df[full_df["drive_id"] == train_drive].copy()
    val_df = full_df[full_df["drive_id"] == val_drive].copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")

    # Fill NaNs in features
    train_df[FEATURE_NAMES] = train_df[FEATURE_NAMES].fillna(0)
    val_df[FEATURE_NAMES] = val_df[FEATURE_NAMES].fillna(0)

    # -------------------------------------------------------------------------
    # 3. Model Training (LightGBM Quantile Regression)
    # -------------------------------------------------------------------------
    print("\n[3] Training Quantile Models...")

    # Override params for speed
    fast_params = LGBM_PARAMS.copy()
    fast_params.update({"n_estimators": 10, "num_leaves": 10, "verbose": -1})

    # Initialize wrappers for East and North components
    model_east = QuantileLGBMWrapper(
        model_path_base=os.path.join(WORKING_DIR, "demo_east"), params=fast_params
    )
    model_north = QuantileLGBMWrapper(
        model_path_base=os.path.join(WORKING_DIR, "demo_north"), params=fast_params
    )

    # Train East Model
    print("Training East Model...")
    model_east.fit(
        train_df[FEATURE_NAMES],
        train_df["target_east"],
        val_df[FEATURE_NAMES],
        val_df["target_east"],
        feature_names=FEATURE_NAMES,
    )

    # Train North Model
    print("Training North Model...")
    model_north.fit(
        train_df[FEATURE_NAMES],
        train_df["target_north"],
        val_df[FEATURE_NAMES],
        val_df["target_north"],
        feature_names=FEATURE_NAMES,
    )

    # -------------------------------------------------------------------------
    # 4. Prediction
    # -------------------------------------------------------------------------
    print("\n[4] Generating Predictions...")

    # Predict Median and Uncertainty
    pred_east, unc_east = model_east.predict_with_uncertainty(val_df[FEATURE_NAMES])
    pred_north, unc_north = model_north.predict_with_uncertainty(val_df[FEATURE_NAMES])

    # Store raw predictions (Baseline + Correction)
    # Note: Target was (GT - WLS), so Pred_GT = WLS + Pred_Residual
    # However, we will use the smoother which takes the residual prediction and WLS

    # Prepare data for smoothing
    # We need to organize by trip because Kalman Filter is sequential
    val_df["pred_east_res"] = pred_east
    val_df["pred_north_res"] = pred_north
    val_df["unc_east"] = unc_east
    val_df["unc_north"] = unc_north

    # -------------------------------------------------------------------------
    # 5. Adaptive Kalman Smoothing
    # -------------------------------------------------------------------------
    print("\n[5] Applying Adaptive Kalman Smoothing...")

    smoother = AdaptiveKalmanFilter(process_noise_std=0.5)

    smoothed_lats = []
    smoothed_lons = []

    # Process each trip in validation set
    for trip_id, trip_data in val_df.groupby("tripId"):
        trip_data = trip_data.sort_values("UnixTimeMillis")

        # Observations: WLS Position + Predicted Residual
        # The Kalman Filter state is [East, North, Vel_E, Vel_N] relative to the WLS reference?
        # No, usually KF tracks absolute position or position relative to a fixed point.
        # Here, WLS changes every step.
        # Standard approach: KF tracks the *Residual* (Error) or the *Absolute Position*.
        # Let's track the Absolute Position (ENU) relative to the *first* WLS point of the trip to maintain a local cartesian frame.

        # Reference point for ENU conversion (First WLS point of the trip)
        ref_lat = trip_data["WlsLat"].iloc[0]
        ref_lon = trip_data["WlsLon"].iloc[0]
        ref_alt = trip_data["WlsAlt"].iloc[0]

        # Convert WLS sequence to ENU relative to ref
        wls_e, wls_n, _ = wgs84_to_enu(
            trip_data["WlsLat"].values,
            trip_data["WlsLon"].values,
            trip_data["WlsAlt"].values,
            ref_lat,
            ref_lon,
            ref_alt,
        )

        # Corrected Observation (WLS + Predicted Residual)
        obs_e = wls_e + trip_data["pred_east_res"].values
        obs_n = wls_n + trip_data["pred_north_res"].values
        observations = np.column_stack((obs_e, obs_n))

        # Uncertainties
        uncertainties = np.column_stack(
            (trip_data["unc_east"].values, trip_data["unc_north"].values)
        )

        # Timestamps
        timestamps = trip_data["UnixTimeMillis"].values

        # Run Smoother
        smoothed_enu = smoother.smooth(observations, uncertainties, timestamps)

        # Convert Smoothed ENU back to WGS84
        # Note: We assume 'Up' component is same as WLS 'Up' (converted to ENU) for simplicity,
        # or just 0 relative to ref if we ignore altitude changes for lat/lon.
        # Let's use the WLS altitude relative to ref.
        _, _, wls_u = wgs84_to_enu(
            trip_data["WlsLat"].values,
            trip_data["WlsLon"].values,
            trip_data["WlsAlt"].values,
            ref_lat,
            ref_lon,
            ref_alt,
        )

        s_lats, s_lons, _ = enu_to_wgs84(
            smoothed_enu[:, 0], smoothed_enu[:, 1], wls_u, ref_lat, ref_lon, ref_alt
        )

        # Store results aligned with the trip dataframe index
        val_df.loc[trip_data.index, "LatitudeDegrees_pred"] = s_lats
        val_df.loc[trip_data.index, "LongitudeDegrees_pred"] = s_lons

    # -------------------------------------------------------------------------
    # 6. Evaluation
    # -------------------------------------------------------------------------
    print("\n[6] Evaluating Performance...")

    # Calculate Baseline Score (WLS)
    baseline_score = competition_score(
        val_df,
        pred_lat_col="WlsLat",
        pred_lon_col="WlsLon",
        gt_lat_col="LatitudeDegrees",
        gt_lon_col="LongitudeDegrees",
    )
    print(f"Baseline WLS Score: {baseline_score:.4f} meters")

    # Calculate Model Score
    final_score = competition_score(
        val_df,
        pred_lat_col="LatitudeDegrees_pred",
        pred_lon_col="LongitudeDegrees_pred",
        gt_lat_col="LatitudeDegrees",
        gt_lon_col="LongitudeDegrees",
    )
    print(f"Final Model Score:  {final_score:.4f} meters")

    # Validation assertions
    assert not val_df["LatitudeDegrees_pred"].isna().any(), "NaNs in predicted latitude"
    assert (
        not val_df["LongitudeDegrees_pred"].isna().any()
    ), "NaNs in predicted longitude"
    assert final_score > 0, "Score must be positive"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
