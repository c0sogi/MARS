import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.config import OUTPUT_DIR, OPT_LAMBDA, HUBER_DELTA, SEED
from library.gnss_utils import ecef2lla, lla2ecef, ecef2enu, enu2ecef


class TrajectoryOptimizer:
    """
    Implements Reliability-Weighted Graph Optimization to fuse ML predictions (Anchors)
    with Time-Differenced Carrier Phase kinematics (Odometry).
    """

    def __init__(self):
        self.cache_dir = os.path.join(OUTPUT_DIR, "opt_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _huber_loss(self, r, delta):
        """
        Computes Huber loss for a vector of residuals r.
        L(r) = 0.5 * r^2            if |r| <= delta
             = delta * (|r| - 0.5*delta)  otherwise
        """
        abs_r = np.abs(r)
        mask = abs_r <= delta
        loss = np.zeros_like(r)
        loss[mask] = 0.5 * r[mask] ** 2
        loss[~mask] = delta * (abs_r[~mask] - 0.5 * delta)
        return np.sum(loss)

    def _objective_1d(self, x, anchors, deltas, weights, lam, huber_delta):
        """
        Objective function for 1D optimization (East or North).

        Args:
            x: Vector of positions to optimize (N,)
            anchors: Vector of ML predicted positions (N,)
            deltas: Vector of kinematic displacements (N,). deltas[i] is move from i-1 to i.
            weights: Vector of reliability weights (N,).
            lam: Regularization strength for smoothness.
            huber_delta: Delta parameter for Huber loss.

        Returns:
            Scalar cost.
        """
        # 1. Anchor Cost (Huber)
        # Penalize deviation from ML predictions
        residuals_anchor = x - anchors
        cost_anchor = self._huber_loss(residuals_anchor, huber_delta)

        # 2. Kinematic/Smoothness Cost (Weighted L2)
        # Penalize deviation from TDCP displacement
        # x[i] - x[i-1] should be close to deltas[i]
        # We start from i=1
        diff = x[1:] - x[:-1]
        residuals_kin = diff - deltas[1:]

        # Weights align with the 'current' timestamp of the delta
        # weights[i] corresponds to the reliability of transition i-1 -> i
        w = weights[1:]

        # Weighted Squared Error
        cost_kin = np.sum(w * residuals_kin**2)

        return cost_anchor + lam * cost_kin

    def _optimize_component(self, init_pos, anchors, deltas, weights):
        """
        Optimizes a single component (East or North) using L-BFGS-B.
        """
        # Initial guess is the ML anchor trajectory
        x0 = anchors.copy()

        # Optimization
        res = minimize(
            fun=self._objective_1d,
            x0=x0,
            args=(anchors, deltas, weights, OPT_LAMBDA, HUBER_DELTA),
            method="L-BFGS-B",
            options={"maxiter": 15000, "ftol": 1e-6, "disp": False},
        )

        return res.x

    def optimize_drive(self, drive_df):
        """
        Optimizes the trajectory for a single drive.
        """
        # Sort by time just in case
        drive_df = drive_df.sort_values("UnixTimeMillis").reset_index(drop=True)

        # 1. Establish Reference Frame (First epoch WLS)
        # We need a local ENU frame. Use the first valid WLS position as origin.
        # Note: We use WLS of the first epoch, not GT, to be realistic for test set.
        ref_row = drive_df.iloc[0]
        ref_x = ref_row["WlsPositionXEcefMeters"]
        ref_y = ref_row["WlsPositionYEcefMeters"]
        ref_z = ref_row["WlsPositionZEcefMeters"]

        # If WLS is NaN at start, find first valid
        if np.isnan(ref_x):
            valid_rows = drive_df.dropna(subset=["WlsPositionXEcefMeters"])
            if not valid_rows.empty:
                ref_row = valid_rows.iloc[0]
                ref_x, ref_y, ref_z = ref_row[
                    [
                        "WlsPositionXEcefMeters",
                        "WlsPositionYEcefMeters",
                        "WlsPositionZEcefMeters",
                    ]
                ]
            else:
                # Fallback if absolutely no WLS (should not happen based on data loader)
                return drive_df[
                    ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
                ]  # Return as is (likely placeholders)

        lat0, lon0, alt0 = ecef2lla(ref_x, ref_y, ref_z)

        # 2. Prepare ML Anchors in ENU
        # The ML model predicted residuals (pred_E, pred_N) relative to the instantaneous WLS.
        # We need to convert instantaneous WLS to the *Reference* ENU frame, then add residuals.

        wls_x = drive_df["WlsPositionXEcefMeters"].values
        wls_y = drive_df["WlsPositionYEcefMeters"].values
        wls_z = drive_df["WlsPositionZEcefMeters"].values

        # Convert all WLS ECEF to ENU relative to Ref
        wls_e, wls_n, wls_u = ecef2enu(wls_x, wls_y, wls_z, lat0, lon0, alt0)

        # Add predicted residuals to get Anchor Positions
        # Fill NaNs in predictions with 0 (fallback to WLS)
        pred_e = drive_df["pred_E"].fillna(0.0).values
        pred_n = drive_df["pred_N"].fillna(0.0).values

        anchors_e = wls_e + pred_e
        anchors_n = wls_n + pred_n

        # 3. Prepare Kinematics
        # d_E, d_N are relative displacements in ENU
        deltas_e = drive_df["d_E"].fillna(0.0).values
        deltas_n = drive_df["d_N"].fillna(0.0).values
        weights = drive_df["weight"].fillna(0.0).values

        # 4. Optimize East and North independently
        opt_e = self._optimize_component(anchors_e, anchors_e, deltas_e, weights)
        opt_n = self._optimize_component(anchors_n, anchors_n, deltas_n, weights)

        # 5. Convert Optimized ENU back to LLA
        # We use the original WLS Up component (wls_u) to reconstruct 3D position,
        # assuming vertical error correction isn't the primary target or is handled by altitude correction in features.
        # Ideally, we'd optimize Up too, but horizontal metric is 2D.
        # Using wls_u preserves the vertical geometry roughly.

        opt_x, opt_y, opt_z = enu2ecef(opt_e, opt_n, wls_u, lat0, lon0, alt0)
        opt_lat, opt_lon, _ = ecef2lla(opt_x, opt_y, opt_z)

        # 6. Construct Result
        result = drive_df[["tripId", "UnixTimeMillis"]].copy()
        result["LatitudeDegrees"] = opt_lat
        result["LongitudeDegrees"] = opt_lon

        return result

    def optimize(self, dataset_df, predictions_df, load_cached_data=True):
        """
        Main entry point. Merges predictions and runs optimization for all drives.

        Args:
            dataset_df: DataFrame containing features and kinematics (from data_loader)
            predictions_df: DataFrame containing ML predictions (pred_E, pred_N)
            load_cached_data: Whether to load optimized results from cache

        Returns:
            DataFrame with final trajectory [tripId, UnixTimeMillis, LatitudeDegrees, LongitudeDegrees]
        """
        # Merge predictions
        # Ensure we don't duplicate columns if they exist
        cols_to_use = predictions_df.columns.difference(dataset_df.columns).tolist()
        cols_to_use.append("UnixTimeMillis")

        # We need to merge carefully. The dataset_df might have multiple drives.
        # predictions_df should align by timestamp.
        # However, timestamps might not be unique across different drives (though unlikely to overlap exactly in test).
        # Safer to merge on index if aligned, or merge on tripId/Time if available.
        # dataset_df has 'tripId' from data_loader.

        if "tripId" in predictions_df.columns:
            full_df = pd.merge(
                dataset_df, predictions_df, on=["tripId", "UnixTimeMillis"], how="left"
            )
        else:
            # If predictions_df only has time, we assume it matches the test set structure
            # But dataset_df is the source of truth for structure.
            # Let's assume predictions_df was generated from dataset_df and has same length/order or join on Time
            # Warning: If multiple trips share timestamps, simple join is risky.
            # Best practice: predictions_df should be generated per trip or have tripId.
            # Assuming predictions_df comes from model.predict(dataset_df), it aligns row-by-row.
            full_df = dataset_df.copy()
            full_df["pred_E"] = predictions_df["pred_E"]
            full_df["pred_N"] = predictions_df["pred_N"]

        # Group by TripID
        trips = full_df["tripId"].unique()
        results = []

        print(f"Optimizing trajectories for {len(trips)} trips...")

        for trip_id in trips:
            trip_data = full_df[full_df["tripId"] == trip_id].copy()

            # Cache check per trip
            cache_file = os.path.join(self.cache_dir, f"opt_{trip_id}.parquet")

            if load_cached_data and os.path.exists(cache_file):
                # print(f"Loading cached trajectory for {trip_id}")
                opt_res = pd.read_parquet(cache_file)
            else:
                # Optimize
                opt_res = self.optimize_drive(trip_data)
                # Save cache
                opt_res.to_parquet(cache_file)

            results.append(opt_res)

        final_df = pd.concat(results, ignore_index=True)
        return final_df
