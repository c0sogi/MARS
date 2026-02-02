import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config
from library.utils import GeoUtils
from library.model import LGBMResidualModel


class KinematicKalmanSmoother:
    """
    Implements a Kalman Filter with Control Input (Doppler Velocity) and Innovation Gating.
    State: [East, North]
    Control: [v_East, v_North]
    Measurement: [East_ML, North_ML]
    """

    def __init__(
        self, process_noise_std=0.5, measurement_noise_std=5.0, gate_threshold=None
    ):
        """
        Args:
            process_noise_std (float): Standard deviation of the process noise (meters/sec).
                                       Represents uncertainty in the velocity integration.
            measurement_noise_std (float): Standard deviation of the measurement noise (meters).
                                           Represents uncertainty in the ML-corrected position.
            gate_threshold (float): Threshold in meters for innovation gating.
                                    If None, uses Config.KF_GATE_THRESHOLD.
        """
        self.Q = np.eye(2) * (process_noise_std**2)
        self.R = np.eye(2) * (measurement_noise_std**2)
        self.gate_threshold = (
            gate_threshold if gate_threshold is not None else Config.KF_GATE_THRESHOLD
        )

    def smooth_trip(self, df_trip):
        """
        Apply smoothing to a single trip's dataframe.
        Expects columns: 'UnixTimeMillis', 'meas_east', 'meas_north', 'v_east', 'v_north'
        """
        # Sort by time
        df_trip = df_trip.sort_values("UnixTimeMillis").reset_index(drop=True)

        n_steps = len(df_trip)
        if n_steps == 0:
            return np.array([]), np.array([])

        # State: [e, n]
        # Initialize state with the first measurement
        x_est = np.array([df_trip.at[0, "meas_east"], df_trip.at[0, "meas_north"]])
        P_est = np.eye(2) * 10.0  # Initial covariance

        smoothed_east = np.zeros(n_steps)
        smoothed_north = np.zeros(n_steps)

        smoothed_east[0] = x_est[0]
        smoothed_north[0] = x_est[1]

        # Time deltas in seconds
        times = df_trip["UnixTimeMillis"].values
        dt_arr = np.diff(times) / 1000.0

        # Measurements and Controls
        meas = df_trip[["meas_east", "meas_north"]].values
        ctrl = df_trip[["v_east", "v_north"]].values

        for k in range(1, n_steps):
            dt = dt_arr[k - 1]

            # 1. Prediction Step
            # x_pred = F * x_est + B * u
            # F is Identity, B is Identity * dt
            # We use the previous velocity as control input
            u = ctrl[k - 1]
            x_pred = x_est + u * dt

            # P_pred = F * P_est * F.T + Q * dt
            # Process noise grows with time step
            P_pred = P_est + self.Q * dt

            # 2. Gating & Update Step
            z = meas[k]

            # Innovation (Residual)
            y = z - x_pred

            # Innovation Covariance
            S = P_pred + self.R

            # Mahalanobis distance (simplified to Euclidean for thresholding as per prompt strategy)
            # Or just simple Euclidean distance of the residual
            dist = np.linalg.norm(y)

            if dist < self.gate_threshold:
                # Valid measurement: Update
                K = P_pred @ np.linalg.inv(S)
                x_est = x_pred + K @ y
                P_est = (np.eye(2) - K) @ P_pred
            else:
                # Outlier: Reject measurement, trust prediction
                # "Coast" on the velocity
                x_est = x_pred
                P_est = P_pred
                # Note: P_est grows, reflecting increased uncertainty without measurement

            smoothed_east[k] = x_est[0]
            smoothed_north[k] = x_est[1]

        return smoothed_east, smoothed_north


def generate_submission(load_cached_features=True):
    """
    Generates the submission file by:
    1. Loading Test Features.
    2. Loading Trained Models.
    3. Predicting Residuals.
    4. Applying Kinematic Kalman Smoothing.
    5. Saving to CSV.
    """
    print("\n" + "=" * 60)
    print(" GENERATING SUBMISSION")
    print("=" * 60)

    # 1. Load Test Features
    features_path = Config.TEST_FEATURES_PATH
    if load_cached_data and os.path.exists(features_path):
        print(f"Loading test features from {features_path}...")
        test_df = pd.read_parquet(features_path)
    else:
        raise FileNotFoundError(
            f"Test features not found at {features_path}. Please run feature engineering first."
        )

    # 2. Predict Residuals
    print("Loading models and predicting residuals...")
    model = LGBMResidualModel()
    # Ensure models exist
    if not os.path.exists(model.model_east_path) or not os.path.exists(
        model.model_north_path
    ):
        raise FileNotFoundError(
            "Trained models not found. Please train the model first."
        )

    preds_df = model.predict(test_df)

    # Merge predictions back with necessary feature columns (WLS, Doppler)
    # We need: tripId, UnixTimeMillis, WlsLat, WlsLon, WlsAlt, v_east, v_north
    cols_to_merge = [
        "tripId",
        "UnixTimeMillis",
        "WlsLat",
        "WlsLon",
        "WlsAlt",
        "v_east",
        "v_north",
    ]
    # Ensure we don't duplicate columns if they are already in preds_df (unlikely for predict output)
    cols_to_merge = [
        c
        for c in cols_to_merge
        if c not in preds_df.columns or c in ["tripId", "UnixTimeMillis"]
    ]

    full_df = pd.merge(
        preds_df, test_df[cols_to_merge], on=["tripId", "UnixTimeMillis"], how="left"
    )

    # 3. Apply Smoothing per Trip
    print(f"Applying Kinematic Kalman Smoothing (Gate: {Config.KF_GATE_THRESHOLD}m)...")

    results = []
    smoother = KinematicKalmanSmoother()

    unique_trips = full_df["tripId"].unique()

    for trip in tqdm(unique_trips, desc="Smoothing Trips"):
        trip_data = full_df[full_df["tripId"] == trip].copy()

        # A. Convert WLS LLA to ENU
        # Use the first point as the local anchor
        anchor_lat = trip_data.iloc[0]["WlsLat"]
        anchor_lon = trip_data.iloc[0]["WlsLon"]
        anchor_alt = trip_data.iloc[0]["WlsAlt"]

        # Convert WLS Baseline to ENU
        wls_e, wls_n, wls_u = GeoUtils.lla_to_ecef(
            trip_data["WlsLat"].values,
            trip_data["WlsLon"].values,
            trip_data["WlsAlt"].values,
        )
        # Actually utils has ecef_to_enu, let's use that.
        # First LLA -> ECEF
        x, y, z = GeoUtils.lla_to_ecef(
            trip_data["WlsLat"].values,
            trip_data["WlsLon"].values,
            trip_data["WlsAlt"].values,
        )
        e, n, u = GeoUtils.ecef_to_enu(x, y, z, anchor_lat, anchor_lon, anchor_alt)

        # B. Add Predicted Residuals to get "Measured" ENU
        # Target was: GT - WLS. So GT = WLS + Target.
        # Here: Meas = WLS_ENU + Pred_Residuals
        trip_data["meas_east"] = e + trip_data["pred_east"]
        trip_data["meas_north"] = n + trip_data["pred_north"]

        # C. Run Smoother
        # Note: v_east and v_north from features are already in ENU frame (rotated in feature_engineering)
        # However, they were rotated relative to the WLS position at THAT timestamp.
        # For small distances (a few km), the change in ENU frame orientation is negligible.
        # We assume the velocity vectors are valid in this local trip frame.

        s_e, s_n = smoother.smooth_trip(trip_data)

        # D. Convert Smoothed ENU back to LLA
        # Use the same anchor
        # We keep the original Up component (u) because we didn't smooth altitude
        # (Task is horizontal only, but we need 3D for conversion)

        sx, sy, sz = GeoUtils.enu_to_ecef(
            s_e, s_n, u, anchor_lat, anchor_lon, anchor_alt
        )
        slat, slon, salt = GeoUtils.ecef_to_lla(sx, sy, sz)

        trip_data["LatitudeDegrees"] = slat
        trip_data["LongitudeDegrees"] = slon

        results.append(
            trip_data[
                ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
            ]
        )

    submission_df = pd.concat(results, ignore_index=True)

    # 4. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


# Note: The main execution block is omitted as per instructions.
