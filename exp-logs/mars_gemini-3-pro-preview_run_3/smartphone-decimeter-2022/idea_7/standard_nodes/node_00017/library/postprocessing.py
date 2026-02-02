import numpy as np
import pandas as pd
import os
from library.config import INNOVATION_THRESHOLD, SUBMISSION_DIR
from library.utils import enu_to_wgs84


class KinematicSmoother:
    """
    Implements an Innovation-Gated Kalman Filter for trajectory smoothing.
    Operates in the local ENU Cartesian frame.
    """

    def __init__(
        self, innovation_threshold=INNOVATION_THRESHOLD, r_sigma=5.0, q_sigma=0.5
    ):
        """
        Args:
            innovation_threshold (float): Threshold in meters to reject outliers.
            r_sigma (float): Measurement noise standard deviation (meters).
            q_sigma (float): Process noise spectral density (m/s^2 / sqrt(Hz)).
        """
        self.innovation_threshold = innovation_threshold
        # Measurement noise covariance R (assuming independent East/North errors)
        self.R = np.diag([r_sigma**2, r_sigma**2])
        self.Q_sigma = q_sigma

    def _smooth_trip(self, group):
        """
        Apply Gated Kalman Filter to a single trip's data.
        """
        # Ensure data is sorted by time
        group = group.sort_values("UnixTimeMillis").reset_index(drop=True)

        times = group["UnixTimeMillis"].values
        obs_e = group["corr_e"].values
        obs_n = group["corr_n"].values

        n_samples = len(group)

        # State Vector: [e, n, ve, vn]
        # Initialize state with the first observation and zero velocity
        x_est = np.zeros((4, 1))
        x_est[0] = obs_e[0]
        x_est[1] = obs_n[0]

        # Initial State Covariance P
        # High uncertainty for velocity, low for position (trusting first fix relatively well)
        P_est = np.eye(4)
        P_est[0, 0] = self.R[0, 0]
        P_est[1, 1] = self.R[1, 1]
        P_est[2, 2] = 100.0
        P_est[3, 3] = 100.0

        # Observation Matrix H (we observe position e, n)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        I = np.eye(4)

        # Arrays to store smoothed results
        smoothed_e = np.zeros(n_samples)
        smoothed_n = np.zeros(n_samples)

        # First point is initialization
        smoothed_e[0] = obs_e[0]
        smoothed_n[0] = obs_n[0]

        for k in range(1, n_samples):
            # Calculate time step in seconds
            dt = (times[k] - times[k - 1]) / 1000.0
            if dt <= 0:
                dt = 1.0  # Fallback for duplicate timestamps if any

            # State Transition Matrix F (Constant Velocity Model)
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

            # Process Noise Covariance Q
            # Based on Discrete White Noise Acceleration Model
            sp = (dt**3 / 3) * (self.Q_sigma**2)
            sv = dt * (self.Q_sigma**2)
            spv = (dt**2 / 2) * (self.Q_sigma**2)

            Q = np.array(
                [[sp, 0, spv, 0], [0, sp, 0, spv], [spv, 0, sv, 0], [0, spv, 0, sv]]
            )

            # 1. Predict Step
            x_pred = F @ x_est
            P_pred = F @ P_est @ F.T + Q

            # 2. Gating Step
            # Current Observation
            z = np.array([[obs_e[k]], [obs_n[k]]])

            # Innovation (Residual)
            y = z - H @ x_pred

            # Innovation Magnitude (Euclidean distance)
            innov_mag = np.sqrt(y[0, 0] ** 2 + y[1, 0] ** 2)

            # Check Gate
            if innov_mag > self.innovation_threshold:
                # Reject Observation: Treat as missing data
                # State update is purely based on prediction
                x_est = x_pred
                P_est = P_pred
            else:
                # 3. Update Step (Standard Kalman)
                # Innovation Covariance
                S = H @ P_pred @ H.T + self.R
                # Kalman Gain
                K = P_pred @ H.T @ np.linalg.inv(S)
                # Update State
                x_est = x_pred + K @ y
                # Update Covariance
                P_est = (I - K @ H) @ P_pred

            # Store smoothed position
            smoothed_e[k] = x_est[0, 0]
            smoothed_n[k] = x_est[1, 0]

        return pd.DataFrame(
            {"UnixTimeMillis": times, "smooth_e": smoothed_e, "smooth_n": smoothed_n}
        )

    def apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply innovation-gated Kalman smoothing to the entire dataset.

        Args:
            df: DataFrame containing metadata and predictions.
                Required columns: 'tripId', 'UnixTimeMillis', 'wls_e', 'wls_n',
                                  'pred_e', 'pred_n', 'wls_u',
                                  'anchor_lat', 'anchor_lon', 'anchor_alt'

        Returns:
            DataFrame with columns: ['tripId', 'UnixTimeMillis', 'LatitudeDegrees', 'LongitudeDegrees']
        """
        # Calculate Base Corrected Coordinates (WLS + Prediction)
        df_proc = df.copy()
        df_proc["corr_e"] = df_proc["wls_e"] + df_proc["pred_e"]
        df_proc["corr_n"] = df_proc["wls_n"] + df_proc["pred_n"]

        # Apply smoothing per trip
        smoothed_results = []
        unique_trips = df_proc["tripId"].unique()

        print(f"Smoothing {len(unique_trips)} trips...")

        for trip in unique_trips:
            trip_df = df_proc[df_proc["tripId"] == trip]
            smoothed_trip = self._smooth_trip(trip_df)
            smoothed_trip["tripId"] = trip
            smoothed_results.append(smoothed_trip)

        smoothed_all = pd.concat(smoothed_results, ignore_index=True)

        # Merge back with original metadata to get anchors and altitude for conversion
        # Use left join to preserve original order and metadata
        final_df = pd.merge(
            df_proc, smoothed_all, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Convert Smoothed ENU back to WGS84 (Lat/Lon)
        # We use the original WLS Up component ('wls_u') as we only smoothed horizontal components
        lats, lons, _ = enu_to_wgs84(
            final_df["smooth_e"].values,
            final_df["smooth_n"].values,
            final_df["wls_u"].values,
            final_df["anchor_lat"].values,
            final_df["anchor_lon"].values,
            final_df["anchor_alt"].values,
        )

        final_df["LatitudeDegrees"] = lats
        final_df["LongitudeDegrees"] = lons

        return final_df[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]


def generate_submission(
    test_meta: pd.DataFrame, pred_e: np.ndarray, pred_n: np.ndarray
) -> pd.DataFrame:
    """
    Generates the submission file using the KinematicSmoother.

    Args:
        test_meta (pd.DataFrame): Test metadata with WLS and Anchor info.
        pred_e (np.array): Predicted East residuals.
        pred_n (np.array): Predicted North residuals.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    # Attach predictions to metadata
    test_meta_df = test_meta.copy()
    test_meta_df["pred_e"] = pred_e
    test_meta_df["pred_n"] = pred_n

    # Initialize Smoother
    smoother = KinematicSmoother()

    print("Applying Kinematic Smoothing to Test Predictions...")
    submission_df = smoother.apply_smoothing(test_meta_df)

    # Ensure directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Save to CSV
    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return submission_df
