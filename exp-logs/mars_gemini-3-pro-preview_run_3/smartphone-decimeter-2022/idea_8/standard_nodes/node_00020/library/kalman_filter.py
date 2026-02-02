import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from library.config import Config
from library.coord_utils import geodetic_to_enu, enu_to_geodetic


class RobustKalmanSmoother:
    def __init__(self):
        self.q_sigma = Config.KF_Q_SIGMA
        self.r_sigma = Config.KF_R_SIGMA
        self.gate_threshold = Config.KF_GATE_THRESHOLD
        self.ransac_window = Config.RANSAC_WINDOW_SECONDS
        self.ransac_min_samples = Config.RANSAC_MIN_SAMPLES
        self.ransac_residual = Config.RANSAC_RESIDUAL_THRESHOLD

    def initialize_state_ransac(self, t, z):
        """
        Estimate initial state [x, y, vx, vy] using RANSAC on the first few seconds.

        Args:
            t: Timestamps (relative seconds)
            z: Measurements (N, 2) [x, y]

        Returns:
            x_init: Initial state vector (4,)
            P_init: Initial covariance matrix (4, 4)
        """
        # Select window
        mask = t <= self.ransac_window
        if np.sum(mask) < self.ransac_min_samples:
            # Fallback if not enough points: assume static start or trust first measurement
            return np.array([z[0, 0], z[0, 1], 0, 0]), np.eye(4) * self.r_sigma**2

        t_window = t[mask].reshape(-1, 1)
        z_window = z[mask]

        # Fit X position/velocity
        try:
            ransac_x = RANSACRegressor(
                min_samples=self.ransac_min_samples,
                residual_threshold=self.ransac_residual,
                random_state=Config.SEED,
            )
            ransac_x.fit(t_window, z_window[:, 0])
            pos_x = ransac_x.predict([[0]])[0]
            vel_x = ransac_x.estimator_.coef_[0]
        except Exception:
            pos_x = z[0, 0]
            vel_x = 0

        # Fit Y position/velocity
        try:
            ransac_y = RANSACRegressor(
                min_samples=self.ransac_min_samples,
                residual_threshold=self.ransac_residual,
                random_state=Config.SEED,
            )
            ransac_y.fit(t_window, z_window[:, 1])
            pos_y = ransac_y.predict([[0]])[0]
            vel_y = ransac_y.estimator_.coef_[0]
        except Exception:
            pos_y = z[0, 1]
            vel_y = 0

        x_init = np.array([pos_x, pos_y, vel_x, vel_y])

        # Initial covariance: High uncertainty on velocity if fit was poor, but let's standardise
        P_init = np.eye(4) * self.r_sigma**2
        P_init[2, 2] *= 10
        P_init[3, 3] *= 10

        return x_init, P_init

    def filter_with_gating(self, t, z):
        """
        Forward Kalman Filter with Innovation Gating.

        Args:
            t: Timestamps (seconds)
            z: Measurements (N, 2)

        Returns:
            means: Filtered state means (N, 4)
            covs: Filtered state covariances (N, 4, 4)
            preds: Predicted state means (N, 4) - for smoothing
            pred_covs: Predicted state covariances (N, 4, 4) - for smoothing
        """
        n_steps = len(t)
        dim_x = 4

        # Storage
        x_filt = np.zeros((n_steps, dim_x))
        P_filt = np.zeros((n_steps, dim_x, dim_x))
        x_pred = np.zeros((n_steps, dim_x))
        P_pred = np.zeros((n_steps, dim_x, dim_x))

        # Initialization
        x_curr, P_curr = self.initialize_state_ransac(t - t[0], z)

        # Store initial
        x_filt[0] = x_curr
        P_filt[0] = P_curr
        x_pred[0] = x_curr
        P_pred[0] = P_curr

        # Measurement matrix (Observing Position X, Y)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurement noise
        R = np.eye(2) * self.r_sigma**2

        for i in range(1, n_steps):
            dt = t[i] - t[i - 1]

            # 1. Predict
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

            # Process noise (Discrete White Noise Acceleration)
            q_var = self.q_sigma**2
            # Block form for Q
            Q = (
                np.array(
                    [
                        [0.25 * dt**4, 0, 0.5 * dt**3, 0],
                        [0, 0.25 * dt**4, 0, 0.5 * dt**3],
                        [0.5 * dt**3, 0, dt**2, 0],
                        [0, 0.5 * dt**3, 0, dt**2],
                    ]
                )
                * q_var
            )

            x_p = F @ x_curr
            P_p = F @ P_curr @ F.T + Q

            # Store predictions for smoother
            x_pred[i] = x_p
            P_pred[i] = P_p

            # 2. Update (with Gating)
            measurement = z[i]

            # Check for NaN measurement
            if np.isnan(measurement).any():
                # Missing data: skip update
                x_curr = x_p
                P_curr = P_p
            else:
                # Innovation
                y = measurement - H @ x_p

                # Gating
                # Simple Euclidean distance gate on innovation
                innov_mag = np.linalg.norm(y)

                if innov_mag > self.gate_threshold:
                    # Gate triggered: Treat as missing measurement (outlier)
                    # Trust prediction
                    x_curr = x_p
                    P_curr = P_p
                else:
                    # Standard Update
                    S = H @ P_p @ H.T + R
                    K = P_p @ H.T @ np.linalg.inv(S)
                    x_curr = x_p + K @ y
                    P_curr = (np.eye(dim_x) - K @ H) @ P_p

            # Store filtered
            x_filt[i] = x_curr
            P_filt[i] = P_curr

        return x_filt, P_filt, x_pred, P_pred

    def rts_smoother(self, t, x_filt, P_filt, x_pred, P_pred):
        """
        Rauch-Tung-Striebel (RTS) Smoother.
        """
        n_steps = len(t)

        x_smooth = np.zeros_like(x_filt)
        P_smooth = np.zeros_like(P_filt)

        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for i in range(n_steps - 2, -1, -1):
            dt = t[i + 1] - t[i]
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

            # Smoother gain
            # C = P_filt * F.T * P_pred_next^-1

            # Numerical stability check for inversion
            try:
                P_pred_inv = np.linalg.inv(P_pred[i + 1])
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_pred[i + 1])

            C = P_filt[i] @ F.T @ P_pred_inv

            x_smooth[i] = x_filt[i] + C @ (x_smooth[i + 1] - x_pred[i + 1])
            P_smooth[i] = P_filt[i] + C @ (P_smooth[i + 1] - P_pred[i + 1]) @ C.T

        return x_smooth

    def apply(self, df):
        """
        Apply Robust Kalman Smoothing to a dataframe representing a single trip.
        Expects columns: 'UnixTimeMillis', 'lat', 'lon'.
        Returns dataframe with 'lat', 'lon' updated.
        """
        if len(df) < 2:
            return df

        df = df.sort_values("UnixTimeMillis").copy()

        # 1. Convert to Local ENU
        # Use first point as reference
        ref_lat = df["lat"].iloc[0]
        ref_lon = df["lon"].iloc[0]
        ref_alt = 0  # Assume 0 for local plane

        # Vectorized conversion
        lats = df["lat"].values
        lons = df["lon"].values
        alts = np.zeros_like(lats)  # Ignore altitude for horizontal smoothing

        e, n, u = geodetic_to_enu(lats, lons, alts, ref_lat, ref_lon, ref_alt)

        # Prepare inputs
        t = df["UnixTimeMillis"].values / 1000.0  # Convert to seconds
        z = np.column_stack((e, n))

        # 2. Filter
        x_filt, P_filt, x_pred, P_pred = self.filter_with_gating(t, z)

        # 3. Smooth
        x_smooth = self.rts_smoother(t, x_filt, P_filt, x_pred, P_pred)

        # 4. Convert back to Geodetic
        e_smooth = x_smooth[:, 0]
        n_smooth = x_smooth[:, 1]
        u_smooth = np.zeros_like(e_smooth)  # Keep 0

        lat_smooth, lon_smooth, _ = enu_to_geodetic(
            e_smooth, n_smooth, u_smooth, ref_lat, ref_lon, ref_alt
        )

        df["lat"] = lat_smooth
        df["lon"] = lon_smooth

        return df
