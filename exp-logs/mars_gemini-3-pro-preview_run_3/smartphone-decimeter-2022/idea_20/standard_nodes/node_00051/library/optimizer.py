import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tqdm import tqdm
from library.utils import CoordinateTransformer, IOHelper


class TrajectoryOptimizer:
    def __init__(self, config=None):
        self.config = config or {}
        # Standard deviations for weights (1/sigma)
        self.sigma_anchor = self.config.get("sigma_anchor", 5.0)
        self.sigma_tdcp = self.config.get("sigma_tdcp", 0.05)
        self.sigma_doppler = self.config.get("sigma_doppler", 1.0)
        self.cache_dir = "./working/idea_20/"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _optimize_component(
        self, init_vals, anchors, deltas, weights_anc, weights_odom
    ):
        """
        Optimize a single component (e.g., Easting) using least squares.

        Args:
            init_vals: Initial guess (ML predictions)
            anchors: Anchor values (ML predictions)
            deltas: Relative changes (x_t - x_{t-1}) from kinematics
            weights_anc: Weights for anchor terms
            weights_odom: Weights for odometry terms (0 if no constraint)
        """
        n = len(init_vals)

        # We construct residuals vector and sparse Jacobian
        # Residuals structure:
        # 0..n-1: Anchor residuals (x_i - anchor_i)
        # n..2n-2: Odometry residuals (x_i - x_{i-1} - delta_i)

        # However, least_squares takes a function computing residuals.
        # For efficiency with sparse jacobian, we define the structure explicitly.

        # Filter valid odometry indices
        valid_odom_idx = np.where(weights_odom > 0)[0]
        n_odom = len(valid_odom_idx)

        # Total residuals
        m = n + n_odom

        # Pre-compute scaling factors
        scale_anc = 1.0 / self.sigma_anchor
        # weights_odom passed in are already 1/sigma for that specific edge

        def fun(x):
            # Anchor residuals
            r_anc = (x - anchors) * scale_anc

            # Odometry residuals
            # x[i] - x[i-1] - delta[i]
            # valid_odom_idx contains 'i' such that constraint exists between i-1 and i
            if n_odom > 0:
                idx = valid_odom_idx
                r_odom = (x[idx] - x[idx - 1] - deltas[idx]) * weights_odom[idx]
                return np.concatenate([r_anc, r_odom])
            else:
                return r_anc

        # Construct Sparse Jacobian Structure
        # J is m x n matrix
        J = lil_matrix((m, n), dtype=float)

        # Anchor block (Identity scaled)
        # We fill with 1s, scaling happens inside solver or we scale J manually?
        # least_squares handles f(x). If we want to provide J, it must match f'(x).

        # Anchor rows: J[i, i] = scale_anc
        J[:n, :n] = np.eye(n) * scale_anc

        # Odometry rows
        if n_odom > 0:
            # Row offset
            row_start = n
            idx = valid_odom_idx

            # r = w * (x[i] - x[i-1] - d)
            # dr/dx[i] = w
            # dr/dx[i-1] = -w

            # We can't vector assign to lil_matrix easily with fancy indexing in older scipy,
            # but loop is fine for structure construction (done once)
            # Actually, let's use coo or csr for construction if possible, but lil is good for building.

            for k, i in enumerate(idx):
                w = weights_odom[i]
                row = row_start + k
                J[row, i] = w
                J[row, i - 1] = -w

        # Convert to CSR for solver
        J_csr = J.tocsr()

        # Solve
        # loss='huber' applies robust loss to residuals.
        # We want it mainly for anchors. Odom constraints are usually trusted if valid.
        # But applying to all is generally safe if sigma is tuned.
        res = least_squares(
            fun, init_vals, jac=lambda x: J_csr, loss="huber", f_scale=1.0, verbose=0
        )

        return res.x

    def _optimize_trip(self, trip_id, df_p, df_k):
        """
        Process a single trip.
        """
        # 1. Merge Predictions and Kinematics
        # df_p has [UnixTimeMillis, LatitudeDegrees, LongitudeDegrees]
        # df_k has [utcTimeMillis, d_east_tdcp, ..., valid_tdcp, ...]

        # Ensure sorted
        df_p = df_p.sort_values("UnixTimeMillis").reset_index(drop=True)

        # Merge
        # Note: df_k might have gaps or extra rows. We align to df_p (submission targets).
        # We assume df_k's 'utcTimeMillis' matches 'UnixTimeMillis'
        merged = pd.merge(
            df_p, df_k, left_on="UnixTimeMillis", right_on="utcTimeMillis", how="left"
        )

        n = len(merged)
        if n == 0:
            return df_p

        # 2. Convert Anchors to ENU
        # Reference point: First epoch WLS or Pred?
        # Using first Pred is fine.
        ref_lat = merged["LatitudeDegrees"].iloc[0]
        ref_lon = merged["LongitudeDegrees"].iloc[0]
        ref_alt = 0.0  # Assumption for 2D optimization

        # Convert all preds to ECEF then ENU
        x, y, z = CoordinateTransformer.wgs84_to_ecef(
            merged["LatitudeDegrees"].values,
            merged["LongitudeDegrees"].values,
            np.zeros(n),
        )
        e_anc, n_anc, u_anc = CoordinateTransformer.ecef_to_enu(
            x, y, z, ref_lat, ref_lon, ref_alt
        )

        # 3. Prepare Kinematic Constraints
        # Constraints exist for index i if:
        # a) Kinematics data exists at i
        # b) Time gap matches (current - prev approx 1s)
        # c) Valid flag is true

        # Check time continuity
        time_diff = merged["UnixTimeMillis"].diff()
        is_continuous = (time_diff > 900) & (time_diff < 1100)  # Expect ~1000ms

        # Initialize weights and deltas
        # Default 0 (no constraint)
        w_e = np.zeros(n)
        w_n = np.zeros(n)

        d_e = np.zeros(n)
        d_n = np.zeros(n)

        # Logic: Prioritize TDCP, fallback to Doppler
        # Note: Kinematics at row i represents motion from i-1 to i

        # Vectorized logic
        has_tdcp = (merged["valid_tdcp"] == True) & is_continuous
        has_dop = (merged["valid_dop"] == True) & is_continuous

        # Apply TDCP
        idx_tdcp = np.where(has_tdcp)[0]
        w_e[idx_tdcp] = 1.0 / self.sigma_tdcp
        w_n[idx_tdcp] = 1.0 / self.sigma_tdcp
        d_e[idx_tdcp] = merged.loc[idx_tdcp, "d_east_tdcp"]
        d_n[idx_tdcp] = merged.loc[idx_tdcp, "d_north_tdcp"]

        # Apply Doppler where TDCP is missing
        idx_dop = np.where(has_dop & (~has_tdcp))[0]
        # Doppler gives Velocity. Delta = V * dt
        dt = time_diff.iloc[idx_dop].values / 1000.0
        w_e[idx_dop] = 1.0 / self.sigma_doppler
        w_n[idx_dop] = 1.0 / self.sigma_doppler
        d_e[idx_dop] = merged.loc[idx_dop, "v_east_dop"] * dt
        d_n[idx_dop] = merged.loc[idx_dop, "v_north_dop"] * dt

        # 4. Optimize
        opt_e = self._optimize_component(
            e_anc, e_anc, d_e, w_e, w_e
        )  # Pass w_e twice? No, func sig is weights_anc, weights_odom
        # Actually _optimize_component takes weights_anc (scalar implicitly handled) and weights_odom (vector)
        # My implementation of _optimize_component uses self.sigma_anchor internally.
        # So we just pass the odom weights.

        opt_e = self._optimize_component(e_anc, e_anc, d_e, None, w_e)
        opt_n = self._optimize_component(n_anc, n_anc, d_n, None, w_n)

        # 5. Convert back to Lat/Lon
        # Assume Up is 0 (or optimized if we wanted 3D)
        opt_x, opt_y, opt_z = CoordinateTransformer.enu_to_ecef(
            opt_e, opt_n, np.zeros(n), ref_lat, ref_lon, ref_alt
        )
        opt_lat, opt_lon, _ = CoordinateTransformer.ecef_to_wgs84(opt_x, opt_y, opt_z)

        # Update dataframe
        df_res = df_p.copy()
        df_res["LatitudeDegrees"] = opt_lat
        df_res["LongitudeDegrees"] = opt_lon

        return df_res

    def optimize_all(self, df_pred, df_kinematics, load_cached_data=True):
        """
        Run optimization for all trips in the prediction dataframe.
        """
        cache_file = "optimized_predictions.parquet"

        if load_cached_data:
            cached = IOHelper.load_parquet(cache_file)
            if cached is not None:
                return cached

        print("Starting Global Graph Optimization...")

        results = []
        unique_trips = df_pred["tripId"].unique()

        for trip_id in tqdm(unique_trips, desc="Optimizing Trips"):
            # Slice data
            trip_pred = df_pred[df_pred["tripId"] == trip_id].copy()
            trip_kin = df_kinematics[df_kinematics["tripId"] == trip_id].copy()

            if trip_kin.empty:
                # No kinematics, return anchors as is
                results.append(trip_pred)
                continue

            try:
                opt_trip = self._optimize_trip(trip_id, trip_pred, trip_kin)
                results.append(opt_trip)
            except Exception as e:
                print(f"Optimization failed for {trip_id}: {e}. Using raw predictions.")
                results.append(trip_pred)

        final_df = pd.concat(results, ignore_index=True)

        # Cache result
        IOHelper.save_parquet(final_df, cache_file)

        return final_df


def run_optimization(df_pred, df_kinematics, load_cached_data=True):
    optimizer = TrajectoryOptimizer()
    return optimizer.optimize_all(
        df_pred, df_kinematics, load_cached_data=load_cached_data
    )
