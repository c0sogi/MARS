import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from library.config import Config
from library.utils import get_logger
from library.coords import enu_to_ecef, ecef_to_geodetic, geodetic_to_ecef

logger = get_logger(__name__)


class KalmanSmoother:
    """
    A simple Kalman Smoother implementation for 1D state (Position, Velocity)
    assuming Constant Velocity model.
    Can be applied independently to X, Y, Z coordinates.
    """

    def __init__(self, process_noise_std=0.5, measurement_noise_std=5.0, dt=1.0):
        self.dt = dt

        # State Transition Matrix (F)
        # x_k = x_{k-1} + v_{k-1} * dt
        # v_k = v_{k-1}
        self.F = np.array([[1, dt], [0, 1]])

        # Observation Matrix (H)
        # z_k = x_k
        self.H = np.array([[1, 0]])

        # Process Noise Covariance (Q)
        # Assumes continuous white noise acceleration model discretized
        # Q = [[dt^3/3, dt^2/2], [dt^2/2, dt]] * sigma_a^2
        sigma_a = process_noise_std
        self.Q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * (sigma_a**2)

        # Measurement Noise Covariance (R)
        self.R = np.array([[measurement_noise_std**2]])

        # Initial State Covariance (P)
        self.P = np.eye(2) * 100.0

    def smooth(self, observations):
        """
        Apply RTS Smoother to a sequence of observations.
        Handles NaNs in observations by skipping update step.

        Args:
            observations (np.array): 1D array of measurements (N,). NaNs indicate missing data.

        Returns:
            np.array: Smoothed state estimates (N,).
        """
        n_samples = len(observations)

        # Forward Pass (Filter)
        x_est = np.zeros((n_samples, 2))  # State estimates
        P_est = np.zeros((n_samples, 2, 2))  # Covariance estimates

        x_pred = np.zeros((n_samples, 2))
        P_pred = np.zeros((n_samples, 2, 2))

        # Initialize
        x_curr = np.array([observations[0] if not np.isnan(observations[0]) else 0, 0])
        P_curr = self.P.copy()

        for k in range(n_samples):
            # Prediction
            if k > 0:
                x_curr = self.F @ x_curr
                P_curr = self.F @ P_curr @ self.F.T + self.Q

            x_pred[k] = x_curr
            P_pred[k] = P_curr

            # Update (if observation exists)
            z = observations[k]
            if not np.isnan(z):
                y = z - self.H @ x_curr  # Residual
                S = self.H @ P_curr @ self.H.T + self.R  # Residual covariance
                K = P_curr @ self.H.T @ np.linalg.inv(S)  # Kalman gain

                x_curr = x_curr + (K @ y)
                P_curr = (np.eye(2) - K @ self.H) @ P_curr

            x_est[k] = x_curr
            P_est[k] = P_curr

        # Backward Pass (Smoother)
        x_smooth = np.zeros_like(x_est)
        P_smooth = np.zeros_like(P_est)

        x_smooth[-1] = x_est[-1]
        P_smooth[-1] = P_est[-1]

        for k in range(n_samples - 2, -1, -1):
            # Smoother Gain
            # C_k = P_k|k * F.T * (P_{k+1}|k)^-1
            # Note: Using pseudoinverse for stability
            C = P_est[k] @ self.F.T @ np.linalg.pinv(P_pred[k + 1])

            x_smooth[k] = x_est[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
            P_smooth[k] = P_est[k] + C @ (P_smooth[k + 1] - P_pred[k + 1]) @ C.T

        return x_smooth[:, 0]  # Return position component


def convert_enu_to_geodetic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts predicted ENU residuals + Reference WLS coordinates to final Geodetic coordinates.
    """
    logger.info("Converting ENU predictions to Geodetic coordinates...")

    # Ensure required columns exist
    req_cols = [Config.TARGET_EAST, Config.TARGET_NORTH, "RefLat", "RefLon", "RefAlt"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} for coordinate conversion.")

    # Inputs
    d_east = df[Config.TARGET_EAST].values
    d_north = df[Config.TARGET_NORTH].values
    d_up = np.zeros_like(d_east)  # We assume 0 vertical error correction for 2D task

    ref_lat = df["RefLat"].values
    ref_lon = df["RefLon"].values
    ref_alt = df["RefAlt"].values

    # 1. ENU -> ECEF
    # This gives us the corrected position in ECEF
    x_ecef, y_ecef, z_ecef = enu_to_ecef(
        d_east, d_north, d_up, ref_lat, ref_lon, ref_alt
    )

    # 2. ECEF -> Geodetic
    pred_lat, pred_lon, pred_alt = ecef_to_geodetic(x_ecef, y_ecef, z_ecef)

    # Assign back to DataFrame
    df["LatitudeDegrees"] = pred_lat
    df["LongitudeDegrees"] = pred_lon

    # Store ECEF for smoothing
    df["x_ecef"] = x_ecef
    df["y_ecef"] = y_ecef
    df["z_ecef"] = z_ecef

    return df


def apply_kalman_smoothing(
    df: pd.DataFrame, process_noise=0.5, meas_noise=3.0
) -> pd.DataFrame:
    """
    Applies Kalman Smoothing to the reconstructed trajectory.
    Operates on ECEF coordinates to avoid curvature issues.
    Re-indexes to 1Hz to handle gaps.
    """
    logger.info("Applying Kalman Smoothing to trajectories...")

    # Initialize Smoother
    # We apply the same smoother parameters to X, Y, Z independently
    kf = KalmanSmoother(
        process_noise_std=process_noise, measurement_noise_std=meas_noise, dt=1.0
    )

    smoothed_rows = []

    # Process each trip
    trips = df["tripId"].unique()

    for trip in tqdm(trips, desc="Smoothing Trips"):
        trip_df = df[df["tripId"] == trip].copy()
        trip_df = trip_df.sort_values("UnixTimeMillis")

        # Create 1Hz grid
        t_min = trip_df["UnixTimeMillis"].min()
        t_max = trip_df["UnixTimeMillis"].max()

        # Create index covering the full range
        full_index = np.arange(t_min, t_max + 1000, 1000)

        # Reindex to fill gaps with NaN
        trip_df_indexed = trip_df.set_index("UnixTimeMillis")
        trip_df_resampled = trip_df_indexed.reindex(full_index)

        # Extract coordinates (with NaNs for gaps)
        obs_x = trip_df_resampled["x_ecef"].values
        obs_y = trip_df_resampled["y_ecef"].values
        obs_z = trip_df_resampled["z_ecef"].values

        # Smooth
        smooth_x = kf.smooth(obs_x)
        smooth_y = kf.smooth(obs_y)
        smooth_z = kf.smooth(obs_z)

        # Convert smoothed ECEF back to Geodetic
        lat_smooth, lon_smooth, _ = ecef_to_geodetic(smooth_x, smooth_y, smooth_z)

        # Assign back to the resampled dataframe
        trip_df_resampled["LatitudeDegrees"] = lat_smooth
        trip_df_resampled["LongitudeDegrees"] = lon_smooth

        # Filter back to original timestamps (remove the filled gaps)
        # We only want predictions for the timestamps requested in the test set
        # For train/val, we also only care about the original rows
        original_indices = trip_df_indexed.index
        final_trip_df = trip_df_resampled.loc[original_indices].reset_index()

        # Restore tripId (lost during reindex if it wasn't the index)
        final_trip_df["tripId"] = trip

        smoothed_rows.append(final_trip_df)

    result_df = pd.concat(smoothed_rows, ignore_index=True)

    # Ensure we return columns in expected order/format
    return result_df


def apply_postprocessing(
    df: pd.DataFrame, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Main post-processing function.
    1. Converts model ENU predictions to Geodetic.
    2. Applies Kalman Smoothing.
    3. Caches results.

    Args:
        df: DataFrame containing 'tripId', 'UnixTimeMillis', 'delta_east', 'delta_north',
            and Reference coordinates ('RefLat', 'RefLon', 'RefAlt').
        load_cached_data: Whether to load from cache if available.

    Returns:
        DataFrame with 'LatitudeDegrees' and 'LongitudeDegrees' columns ready for submission.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "submission_smoothed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading processed predictions from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Starting post-processing...")

    # 1. Reconstruct Trajectory (ENU -> Geodetic)
    df_geo = convert_enu_to_geodetic(df)

    # 2. Apply Kinematic Smoothing
    # Hyperparameters for smoothing can be tuned.
    # Process noise roughly 0.5 m/s^2, Measurement noise roughly 3-5 meters.
    df_smoothed = apply_kalman_smoothing(df_geo, process_noise=0.5, meas_noise=4.0)

    # Select final columns
    final_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    result = df_smoothed[final_cols]

    # 3. Cache
    logger.info(f"Saving processed predictions to cache: {cache_path}")
    result.to_parquet(cache_path, index=False)

    return result
