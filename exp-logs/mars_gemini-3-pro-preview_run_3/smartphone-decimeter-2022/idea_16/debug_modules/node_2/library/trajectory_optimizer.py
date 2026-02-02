import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.config import Config
from library.utils import WGS84_to_ECEF, ECEF_to_WGS84


class TrajectoryAligner:
    """
    Aligns a noisy absolute trajectory (ML predictions) with a precise relative trajectory (TDCP).
    Uses global optimization to minimize a composite cost function:
    J(X) = sum( Huber(X_t - ML_t) ) + lambda * sum( || (X_t - X_{t-1}) - TDCP_delta_t ||^2 )
    """

    def __init__(self):
        self.lambda_param = Config.TDCP_LAMBDA
        # Huber loss delta parameter (meters).
        # Residuals smaller than this are squared (L2), larger are linear (L1).
        # 5.0m is a reasonable threshold for GNSS urban canyon errors.
        self.huber_delta = 5.0

    def _huber_loss(self, residuals):
        """
        Compute Huber loss for a vector of residuals (distances).
        """
        abs_r = np.abs(residuals)
        quadratic = np.minimum(abs_r, self.huber_delta)
        linear = abs_r - quadratic
        return 0.5 * quadratic**2 + self.huber_delta * linear

    def _objective_function(
        self, x_flat, n_points, ml_pos, tdcp_displacements, tdcp_weights
    ):
        """
        Objective function for optimization.

        Args:
            x_flat: Flattened array of optimization variables (ECEF coordinates), shape (N*3,).
            n_points: Number of epochs (N).
            ml_pos: ML predicted positions (centered ECEF), shape (N, 3).
            tdcp_displacements: TDCP delta vectors, shape (N-1, 3).
                                Entry i corresponds to displacement from t_i to t_{i+1}.
                                (Note: Input logic ensures alignment).
            tdcp_weights: Weights for shape constraints, shape (N-1,). 1 if valid, 0 else.

        Returns:
            float: Total cost.
        """
        # Reshape to (N, 3)
        X = x_flat.reshape((n_points, 3))

        # 1. Anchor Cost (Robust Loss against ML predictions)
        # Calculate Euclidean distance between Optimized X and ML Anchor
        diff = X - ml_pos
        dist = np.linalg.norm(diff, axis=1)
        anchor_cost = np.sum(self._huber_loss(dist))

        # 2. Shape Cost (L2 Loss against TDCP displacements)
        # Constraint: X[i] - X[i-1] should approximate TDCP[i]
        # Note on indexing:
        # X has indices 0 to N-1.
        # delta_X[i] = X[i+1] - X[i] (displacement from i to i+1)
        # tdcp_displacements should match this structure.

        delta_X = X[1:] - X[:-1]

        # Residuals between optimized displacement and measured TDCP displacement
        shape_residuals = delta_X - tdcp_displacements

        # Squared Euclidean norm of residuals
        shape_sq_error = np.sum(shape_residuals**2, axis=1)

        # Weighted sum (lambda * weight * error)
        shape_cost = np.sum(shape_sq_error * tdcp_weights)

        return anchor_cost + self.lambda_param * shape_cost

    def _objective_gradient(
        self, x_flat, n_points, ml_pos, tdcp_displacements, tdcp_weights
    ):
        """
        Analytical gradient of the objective function w.r.t X.
        """
        X = x_flat.reshape((n_points, 3))
        grad = np.zeros_like(X)

        # --- 1. Gradient of Anchor Term ---
        diff = X - ml_pos
        dist = np.linalg.norm(diff, axis=1)

        # Derivative of Huber Loss w.r.t distance d:
        # h'(d) = d            if d <= delta
        # h'(d) = delta        if d > delta
        d_huber = np.where(dist <= self.huber_delta, dist, self.huber_delta)

        # Chain rule: dJ/dX = h'(d) * (d_dist/dX)
        # d_dist/dX = diff / dist

        # Safe division for dist=0
        safe_dist = np.where(dist < 1e-9, 1e-9, dist)
        scale = d_huber / safe_dist

        # Broadcasting scale to (N, 3)
        grad += diff * scale[:, np.newaxis]

        # --- 2. Gradient of Shape Term ---
        # Term k (for edge k -> k+1): lambda * w_k * || (X_{k+1} - X_k) - D_k ||^2
        # Let R_k = (X_{k+1} - X_k) - D_k
        # Cost_k = lambda * w_k * (R_k dot R_k)
        # Grad w.r.t X_{k+1}: + 2 * lambda * w_k * R_k
        # Grad w.r.t X_k    : - 2 * lambda * w_k * R_k

        delta_X = X[1:] - X[:-1]
        R = delta_X - tdcp_displacements  # Shape (N-1, 3)

        # Weights (N-1, 1)
        W = (tdcp_weights * self.lambda_param)[:, np.newaxis]

        term_grad = 2 * W * R

        # Add gradient contributions
        # Contribution to X[1:] (as the "head" of the vector)
        grad[1:] += term_grad
        # Contribution to X[:-1] (as the "tail" of the vector)
        grad[:-1] -= term_grad

        return grad.flatten()

    def optimize_drive(
        self, drive_id, phone_name, df_ml, df_tdcp, load_cached_data=True
    ):
        """
        Run the optimization for a single drive.

        Args:
            drive_id (str): Drive Identifier.
            phone_name (str): Phone Name.
            df_ml (pd.DataFrame): ML predictions with [UnixTimeMillis, LatitudeDegrees, LongitudeDegrees].
            df_tdcp (pd.DataFrame): TDCP data with [UnixTimeMillis, dx, dy, dz].
            load_cached_data (bool): Whether to load/save result from cache.

        Returns:
            pd.DataFrame: Optimized trajectory [UnixTimeMillis, LatitudeDegrees, LongitudeDegrees].
        """
        # 1. Cache Check
        cache_file = os.path.join(
            Config.WORKING_DIR, f"opt_{drive_id}_{phone_name}.parquet"
        )
        if load_cached_data and os.path.exists(cache_file):
            # print(f"Loading optimized trajectory from cache: {cache_file}")
            return pd.read_parquet(cache_file)

        # 2. Data Preparation
        # Ensure ML data is sorted
        df_ml = df_ml.sort_values("UnixTimeMillis").reset_index(drop=True)

        # Merge TDCP data onto ML timestamps
        # TDCP row at time T contains displacement from T-1 to T.
        # For optimization logic X[i+1] - X[i], we need displacement from i to i+1.
        # So if we have points 0, 1, 2...
        # TDCP at 1 is disp(0->1). TDCP at 2 is disp(1->2).
        # We need to align them such that index i contains disp(i -> i+1).
        # This effectively means shifting the TDCP column backwards by 1 relative to the points.

        # Let's merge first
        merged = pd.merge(
            df_ml,
            df_tdcp[["UnixTimeMillis", "dx", "dy", "dz"]],
            on="UnixTimeMillis",
            how="left",
        )

        n_points = len(merged)
        if n_points < 2:
            return df_ml  # Can't optimize single point

        # Extract ML Positions (Lat/Lon -> ECEF)
        # Assume Altitude = 0 relative to ellipsoid for conversion stability,
        # as we don't have predicted altitude.
        ml_lats = merged["LatitudeDegrees"].values
        ml_lons = merged["LongitudeDegrees"].values
        ml_alts = np.zeros(n_points)

        ml_x, ml_y, ml_z = WGS84_to_ECEF(ml_lats, ml_lons, ml_alts)
        ml_pos = np.column_stack((ml_x, ml_y, ml_z))

        # Center coordinates to avoid floating point precision issues with large ECEF values
        center = np.mean(ml_pos, axis=0)
        ml_pos_centered = ml_pos - center

        # Extract TDCP Displacements
        # merged['dx'] at index i is displacement from i-1 to i.
        # We need displacement from i to i+1 for the shape constraint between X[i] and X[i+1].
        # So we take values from index 1 to N.
        tdcp_dx = merged["dx"].values[1:]
        tdcp_dy = merged["dy"].values[1:]
        tdcp_dz = merged["dz"].values[1:]

        # Create validity mask
        valid_mask = ~np.isnan(tdcp_dx)

        # Fill NaNs with 0 to prevent errors (they will be weighted 0 anyway)
        tdcp_dx = np.nan_to_num(tdcp_dx)
        tdcp_dy = np.nan_to_num(tdcp_dy)
        tdcp_dz = np.nan_to_num(tdcp_dz)

        tdcp_disp_arr = np.column_stack((tdcp_dx, tdcp_dy, tdcp_dz))
        tdcp_weights_arr = valid_mask.astype(float)

        # Initial Guess: ML predictions
        x0 = ml_pos_centered.flatten()

        # 3. Run Optimization
        # Use L-BFGS-B for efficiency with large number of variables
        res = minimize(
            fun=self._objective_function,
            x0=x0,
            args=(n_points, ml_pos_centered, tdcp_disp_arr, tdcp_weights_arr),
            jac=self._objective_gradient,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-6, "disp": False},
        )

        if not res.success:
            print(
                f"Warning: Optimization failed for {drive_id}-{phone_name}: {res.message}"
            )

        # 4. Reconstruction
        opt_pos_centered = res.x.reshape((n_points, 3))
        opt_pos = opt_pos_centered + center

        # Convert back to Lat/Lon
        opt_lat, opt_lon, _ = ECEF_to_WGS84(opt_pos[:, 0], opt_pos[:, 1], opt_pos[:, 2])

        # Create result DataFrame
        result_df = pd.DataFrame(
            {
                "tripId": (
                    df_ml["tripId"]
                    if "tripId" in df_ml.columns
                    else f"{drive_id}-{phone_name}"
                ),
                "UnixTimeMillis": merged["UnixTimeMillis"],
                "LatitudeDegrees": opt_lat,
                "LongitudeDegrees": opt_lon,
            }
        )

        # 5. Save Cache
        try:
            result_df.to_parquet(cache_file, index=False)
        except Exception as e:
            print(f"Failed to save optimization cache: {e}")

        return result_df
