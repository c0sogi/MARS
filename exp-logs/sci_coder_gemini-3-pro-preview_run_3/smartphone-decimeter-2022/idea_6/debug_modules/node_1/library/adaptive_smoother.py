import numpy as np
from library.config import (
    PROCESS_NOISE_STD,
    UNCERTAINTY_SCALE_FACTOR,
    BASE_MEASUREMENT_NOISE,
)


class AdaptiveKalmanFilter:
    """
    Implements an Uncertainty-Gated Kalman Smoother (Forward Filter + RTS Smoother).
    The measurement noise matrix R is scaled dynamically based on the predicted uncertainty.
    """

    def __init__(self, process_noise_std=PROCESS_NOISE_STD):
        """
        Initialize the Kalman Filter parameters.

        Args:
            process_noise_std (float): Standard deviation for the process noise (acceleration).
        """
        self.process_noise_std = process_noise_std
        # State vector: [East, North, Vel_East, Vel_North]
        self.state_dim = 4
        # Measurement vector: [East, North]
        self.meas_dim = 2

        # Measurement matrix H (static): We observe position directly
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

        # Initial State Covariance P (High uncertainty initially)
        self.P_init = np.eye(self.state_dim) * 100.0

    def _get_F(self, dt):
        """
        Construct the State Transition Matrix F for a given time step dt.
        Constant Velocity Model.
        """
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
        )

    def _get_Q(self, dt):
        """
        Construct the Process Noise Covariance Matrix Q for a given time step dt.
        Discrete White Noise Acceleration Model.
        """
        # Block for one dimension (position, velocity)
        # Variance = sigma_a^2
        # Matrix = [dt^4/4, dt^3/2; dt^3/2, dt^2]

        q_var = self.process_noise_std**2
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        q_block = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]]) * q_var

        Q = np.zeros((4, 4))
        # East dimension
        Q[0, 0] = q_block[0, 0]
        Q[0, 2] = q_block[0, 1]
        Q[2, 0] = q_block[1, 0]
        Q[2, 2] = q_block[1, 1]

        # North dimension
        Q[1, 1] = q_block[0, 0]
        Q[1, 3] = q_block[0, 1]
        Q[3, 1] = q_block[1, 0]
        Q[3, 3] = q_block[1, 1]

        return Q

    def smooth(self, observations, uncertainties, timestamps):
        """
        Apply the Uncertainty-Gated Kalman Smoother to a sequence of observations.

        Args:
            observations (np.ndarray): Shape (N, 2). Predicted [East, North] positions.
            uncertainties (np.ndarray): Shape (N, 2). Uncertainty estimates (IQR) for [East, North].
            timestamps (np.ndarray): Shape (N,). Timestamps in milliseconds.

        Returns:
            np.ndarray: Shape (N, 2). Smoothed [East, North] positions.
        """
        n_samples = len(observations)
        if n_samples == 0:
            return np.empty((0, 2))

        # Storage for Forward Pass
        # x_{t|t-1} (Prior state)
        xs_pred = [None] * n_samples
        # P_{t|t-1} (Prior covariance)
        Ps_pred = [None] * n_samples
        # x_{t|t} (Posterior state)
        xs_upd = [None] * n_samples
        # P_{t|t} (Posterior covariance)
        Ps_upd = [None] * n_samples
        # Transition matrices F used at each step (needed for backward pass)
        Fs = [None] * n_samples

        # --- Initialization ---
        # Initialize state with the first observation, velocity = 0
        x_curr = np.zeros(self.state_dim)
        x_curr[0] = observations[0, 0]
        x_curr[1] = observations[0, 1]

        P_curr = self.P_init.copy()

        # --- Forward Pass (Filtering) ---
        for i in range(n_samples):
            # 1. Prediction Step
            if i == 0:
                # First step: No transition, just initialization
                dt = 0
                F = np.eye(4)
                x_pred = x_curr
                P_pred = P_curr
            else:
                # Calculate time delta in seconds
                dt = (timestamps[i] - timestamps[i - 1]) / 1000.0
                # Handle potential duplicate timestamps or zero dt
                if dt <= 0:
                    dt = 0.001  # Small epsilon to avoid singular Q if needed, or just 0

                F = self._get_F(dt)
                Q = self._get_Q(dt)

                x_pred = F @ x_curr
                P_pred = F @ P_curr @ F.T + Q

            # Store priors and transition matrix
            xs_pred[i] = x_pred
            Ps_pred[i] = P_pred
            Fs[i] = F

            # 2. Update Step (Gated by Uncertainty)
            # Construct Dynamic Measurement Noise Matrix R
            unc_e = uncertainties[i, 0]
            unc_n = uncertainties[i, 1]

            # Scaling logic: R = (Base + Scale * Uncertainty)^2
            # This makes R explode for high uncertainty, causing the filter to ignore the measurement
            sigma_e = BASE_MEASUREMENT_NOISE + UNCERTAINTY_SCALE_FACTOR * unc_e
            sigma_n = BASE_MEASUREMENT_NOISE + UNCERTAINTY_SCALE_FACTOR * unc_n

            R = np.array([[sigma_e**2, 0], [0, sigma_n**2]])

            # Measurement Residual (Innovation)
            z = observations[i]
            y = z - self.H @ x_pred

            # Innovation Covariance
            S = self.H @ P_pred @ self.H.T + R

            # Kalman Gain
            # K = P H^T S^-1
            try:
                K = P_pred @ self.H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # Fallback for singular matrix (rare with diagonal R)
                K = np.zeros((self.state_dim, self.meas_dim))

            # Update State
            x_curr = x_pred + K @ y
            # Update Covariance: P = (I - KH)P
            I = np.eye(self.state_dim)
            P_curr = (I - K @ self.H) @ P_pred

            # Store posteriors
            xs_upd[i] = x_curr
            Ps_upd[i] = P_curr

        # --- Backward Pass (RTS Smoothing) ---
        # Initialize smoothed estimates with the last filtered estimate
        xs_smooth = [None] * n_samples
        xs_smooth[-1] = xs_upd[-1]

        # Iterate backwards from N-2 to 0
        for k in range(n_samples - 2, -1, -1):
            # We need:
            # P_{k|k} -> Ps_upd[k]
            # F_{k+1} -> Fs[k+1] (Transition from k to k+1)
            # P_{k+1|k} -> Ps_pred[k+1]
            # x_{k+1|k} -> xs_pred[k+1]
            # x_{k+1|N} -> xs_smooth[k+1] (Smoothed state of next step)

            P_curr_upd = Ps_upd[k]
            F_next = Fs[k + 1]
            P_next_pred = Ps_pred[k + 1]

            # Smoother Gain C_k = P_{k|k} F^T P_{k+1|k}^-1
            try:
                C = P_curr_upd @ F_next.T @ np.linalg.inv(P_next_pred)
            except np.linalg.LinAlgError:
                C = np.zeros_like(P_curr_upd)

            # Smooth State: x_{k|N} = x_{k|k} + C_k (x_{k+1|N} - x_{k+1|k})
            x_next_smooth = xs_smooth[k + 1]
            x_next_pred = xs_pred[k + 1]

            xs_smooth[k] = xs_upd[k] + C @ (x_next_smooth - x_next_pred)

        # Extract smoothed positions (East, North)
        smoothed_positions = np.array([x[0:2] for x in xs_smooth])

        return smoothed_positions
