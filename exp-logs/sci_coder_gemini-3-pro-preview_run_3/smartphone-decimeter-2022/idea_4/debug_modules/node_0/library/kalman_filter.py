import numpy as np
import pandas as pd
from library.config import Config


def apply_kalman_smoothing(df):
    """
    Applies Kalman Smoothing to the predicted latitudes and longitudes in the DataFrame.
    Uses a Constant Velocity (CV) motion model.

    Args:
        df (pd.DataFrame): DataFrame containing 'tripId', 'UnixTimeMillis',
                           'LatitudeDegrees', 'LongitudeDegrees'.

    Returns:
        pd.DataFrame: DataFrame with smoothed 'LatitudeDegrees' and 'LongitudeDegrees'.
    """
    # Create a copy to avoid modifying the original dataframe
    smoothed_df = df.copy()

    # Process each trip independently
    trips = smoothed_df["tripId"].unique()

    # Constants for coordinate conversion
    # Approximation of meters per degree
    R_EARTH = 6371000.0
    DEG_TO_RAD = np.pi / 180.0

    # Kalman Filter Parameters from Config
    # Measurement noise std dev (meters)
    R_std = Config.KF_R
    # Process noise std dev (acceleration magnitude, m/s^2)
    Q_std = Config.KF_Q

    for trip_id in trips:
        trip_mask = smoothed_df["tripId"] == trip_id
        trip_data = smoothed_df.loc[trip_mask].sort_values("UnixTimeMillis")

        if len(trip_data) < 2:
            continue

        timestamps = trip_data["UnixTimeMillis"].values
        lats = trip_data["LatitudeDegrees"].values
        lons = trip_data["LongitudeDegrees"].values

        # 1. Convert Lat/Lon to Local Cartesian Coordinates (Meters)
        # Use the mean position as the origin for the local projection
        lat_mean = np.mean(lats)
        lon_mean = np.mean(lons)

        # Meters per degree longitude depends on latitude
        m_per_deg_lat = R_EARTH * DEG_TO_RAD
        m_per_deg_lon = R_EARTH * DEG_TO_RAD * np.cos(lat_mean * DEG_TO_RAD)

        # Project to x (East), y (North)
        y_obs = (lats - lat_mean) * m_per_deg_lat
        x_obs = (lons - lon_mean) * m_per_deg_lon

        # Stack observations: [x, y]
        measurements = np.column_stack((x_obs, y_obs))

        # 2. Kalman Filter Initialization
        n_timesteps = len(timestamps)
        n_dim_state = 4  # [x, y, vx, vy]
        n_dim_obs = 2  # [x, y]

        # State: [x, y, vx, vy]
        # Initialize state mean (x0) and covariance (P0)
        # Assume starting velocity is 0
        x_est = np.zeros((n_timesteps, n_dim_state))
        P_est = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        # Initial state: observation at t=0, velocity=0
        x_est[0] = [measurements[0, 0], measurements[0, 1], 0, 0]

        # Initial covariance: high uncertainty for velocity, measurement noise for position
        P_est[0] = np.eye(n_dim_state) * 1000.0
        P_est[0, 0, 0] = R_std**2
        P_est[0, 1, 1] = R_std**2

        # Measurement matrix H: Maps state [x, y, vx, vy] to [x, y]
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurement noise covariance R
        R = np.eye(n_dim_obs) * (R_std**2)

        # Store predictions for smoothing (Forward Pass)
        x_pred_all = np.zeros_like(x_est)
        P_pred_all = np.zeros_like(P_est)
        F_all = np.zeros((n_timesteps, n_dim_state, n_dim_state))

        # 3. Forward Pass (Filtering)
        for t in range(1, n_timesteps):
            # Calculate time delta in seconds
            dt = (timestamps[t] - timestamps[t - 1]) / 1000.0

            # State Transition Matrix F (Constant Velocity Model)
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            F_all[t] = F

            # Process Noise Covariance Q (Discrete White Noise Acceleration)
            # Models uncertainty in velocity changes (acceleration)
            # Q = G * Q_std^2 * G.T
            # G = [0.5*dt^2, 0.5*dt^2, dt, dt]^T (roughly)
            # Standard block form for CV model:
            q_pos = (dt**4) / 4
            q_pos_vel = (dt**3) / 2
            q_vel = dt**2

            Q = np.array(
                [
                    [q_pos, 0, q_pos_vel, 0],
                    [0, q_pos, 0, q_pos_vel],
                    [q_pos_vel, 0, q_vel, 0],
                    [0, q_pos_vel, 0, q_vel],
                ]
            ) * (Q_std**2)

            # Predict
            x_pred = F @ x_est[t - 1]
            P_pred = F @ P_est[t - 1] @ F.T + Q

            # Store predictions
            x_pred_all[t] = x_pred
            P_pred_all[t] = P_pred

            # Update
            z = measurements[t]
            y_residual = z - H @ x_pred
            S = H @ P_pred @ H.T + R  # Residual covariance
            K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman Gain

            x_est[t] = x_pred + K @ y_residual
            P_est[t] = (np.eye(n_dim_state) - K @ H) @ P_pred

        # 4. Backward Pass (Rauch-Tung-Striebel Smoothing)
        x_smooth = np.zeros_like(x_est)
        P_smooth = np.zeros_like(P_est)

        # Initialize with the last filtered estimate
        x_smooth[-1] = x_est[-1]
        P_smooth[-1] = P_est[-1]

        for t in range(n_timesteps - 2, -1, -1):
            F = F_all[t + 1]
            P_pred_next = P_pred_all[t + 1]

            # Smoothing Gain C
            # C_k = P_{k|k} F^T P_{k+1|k}^-1
            C = P_est[t] @ F.T @ np.linalg.inv(P_pred_next)

            # Smoothed state
            # x_{k|N} = x_{k|k} + C_k (x_{k+1|N} - x_{k+1|k})
            x_smooth[t] = x_est[t] + C @ (x_smooth[t + 1] - x_pred_all[t + 1])

            # Smoothed covariance (optional, not strictly needed for point estimates)
            # P_{k|N} = P_{k|k} + C_k (P_{k+1|N} - P_{k+1|k}) C_k^T
            P_smooth[t] = P_est[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

        # 5. Convert back to Lat/Lon
        x_smooth_vals = x_smooth[:, 0]
        y_smooth_vals = x_smooth[:, 1]

        lats_smooth = (y_smooth_vals / m_per_deg_lat) + lat_mean
        lons_smooth = (x_smooth_vals / m_per_deg_lon) + lon_mean

        # Update DataFrame using the index to ensure alignment
        smoothed_df.loc[trip_mask, "LatitudeDegrees"] = lats_smooth
        smoothed_df.loc[trip_mask, "LongitudeDegrees"] = lons_smooth

    return smoothed_df
