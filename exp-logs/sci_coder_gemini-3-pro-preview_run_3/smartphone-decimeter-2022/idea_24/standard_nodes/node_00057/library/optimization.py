import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from library.config import Config
from library.utils import llh_to_ecef, ecef_to_llh, ecef_to_enu, enu_to_ecef
from library.data_io import load_metadata, load_drive_data
from library.kinematics import compute_tdcp_odometry
from library.model import generate_submission


class GraphOptimizer:
    """
    Implements Global Graph Optimization for GNSS trajectory refinement.
    Fuses absolute position estimates (ML Anchors) with relative displacement estimates (Odometry).
    """

    def __init__(self):
        self.huber_delta_anchor = Config.HUBER_DELTA_ANCHOR
        self.huber_delta_odom = Config.HUBER_DELTA_ODOM
        self.lambda_odom = Config.ODOMETRY_LAMBDA

    def _huber_loss(self, residuals, delta):
        """
        Computes the Huber loss for a vector of residuals.
        L = 0.5 * r^2           if |r| <= delta
        L = delta * (|r| - 0.5 * delta)  otherwise
        """
        abs_r = np.abs(residuals)
        quadratic = np.minimum(abs_r, delta)
        linear = abs_r - quadratic
        return 0.5 * quadratic**2 + delta * linear

    def _objective_function(
        self, x_flat, anchors, odom_deltas, odom_weights, n_samples
    ):
        """
        Objective function for the optimizer.
        x_flat: Flattened state vector [e0, n0, u0, e1, n1, u1, ...]
        """
        # Reshape state vector to (N, 3)
        X = x_flat.reshape((n_samples, 3))

        # 1. Anchor Cost (Unary)
        # Difference between optimized positions and ML predictions
        anchor_diff = X - anchors
        # Compute robust loss sum
        cost_anchor = np.sum(self._huber_loss(anchor_diff, self.huber_delta_anchor))

        # 2. Odometry Cost (Binary)
        # Difference between consecutive optimized positions
        # X[t] - X[t-1]
        x_diff = X[1:] - X[:-1]

        # Error relative to measured odometry
        # (X[t] - X[t-1]) - dOdom[t]
        # Note: odom_deltas[t] corresponds to step t-1 -> t.
        # odom_deltas has length N. Index 0 is usually 0 (start).
        # We align indices: x_diff[i] corresponds to step i -> i+1.
        # odom_deltas[1:] corresponds to steps 0->1, 1->2, etc.

        odom_residuals = x_diff - odom_deltas[1:]

        # Apply weights (if weight is 0, edge is ignored)
        # Weights shape: (N-1, 1) to broadcast across E, N, U
        weights = odom_weights[1:].reshape(-1, 1)

        # Compute robust loss
        # We apply the lambda factor here
        cost_odom = self.lambda_odom * np.sum(
            weights * self._huber_loss(odom_residuals, self.huber_delta_odom)
        )

        return cost_anchor + cost_odom

    def _jacobian(self, x_flat, anchors, odom_deltas, odom_weights, n_samples):
        """
        Analytical gradient of the objective function.
        Speeds up optimization significantly compared to numerical differentiation.
        """
        X = x_flat.reshape((n_samples, 3))
        grad = np.zeros_like(X)

        # --- Anchor Gradient ---
        # d/dx (Huber(x - anchor))
        anchor_diff = X - anchors
        abs_anchor_diff = np.abs(anchor_diff)

        # Mask for quadratic region
        mask_quad = abs_anchor_diff <= self.huber_delta_anchor

        # Derivative of 0.5*r^2 is r
        grad_anchor = np.where(mask_quad, anchor_diff, 0.0)

        # Derivative of delta*(|r| - 0.5*delta) is delta * sign(r)
        grad_anchor += np.where(
            ~mask_quad, self.huber_delta_anchor * np.sign(anchor_diff), 0.0
        )

        grad += grad_anchor

        # --- Odometry Gradient ---
        # Terms involve x_t and x_{t-1}
        # Residual r_t = (x_t - x_{t-1}) - delta_t
        # Cost = sum L(r_t)

        x_diff = X[1:] - X[:-1]
        odom_res = x_diff - odom_deltas[1:]
        weights = odom_weights[1:].reshape(-1, 1)

        # Compute Huber derivative for residuals
        abs_odom_res = np.abs(odom_res)
        mask_odom_quad = abs_odom_res <= self.huber_delta_odom

        d_huber = np.where(mask_odom_quad, odom_res, 0.0)
        d_huber += np.where(
            ~mask_odom_quad, self.huber_delta_odom * np.sign(odom_res), 0.0
        )

        # Scale by lambda and weights
        weighted_grad = self.lambda_odom * weights * d_huber

        # Accumulate gradients
        # For x_t (where t > 0): contributes to term t (positive)
        grad[1:] += weighted_grad

        # For x_{t-1} (where t > 0): contributes to term t (negative)
        grad[:-1] -= weighted_grad

        return grad.flatten()

    def optimize_trip(self, trip_id, drive_id, phone_name, df_ml, df_odom, df_wls):
        """
        Optimizes a single trip.
        """
        # 1. align data
        # We need ML predictions, Odometry, and WLS Altitude for all test timestamps

        # Filter for this trip
        trip_ml = df_ml[df_ml["tripId"] == trip_id].sort_values("UnixTimeMillis").copy()

        # Timestamps required
        timestamps = trip_ml["UnixTimeMillis"].values

        # Filter Odometry
        # We merge to ensure we have odom for exactly the timestamps in submission
        # If odom missing, fill with 0
        trip_odom = df_odom[df_odom["UnixTimeMillis"].isin(timestamps)].copy()
        trip_odom = trip_odom.drop_duplicates(subset=["UnixTimeMillis"])

        # Merge to align rows
        merged = pd.merge(trip_ml, trip_odom, on="UnixTimeMillis", how="left")

        # Fill missing odom
        merged["dE"] = merged["dE"].fillna(0.0)
        merged["dN"] = merged["dN"].fillna(0.0)
        merged["dU"] = merged["dU"].fillna(0.0)
        merged["weight_odom"] = merged["weight_odom"].fillna(0.0)

        # Filter WLS for Altitude
        # We need altitude to convert Lat/Lon to ENU
        trip_wls = df_wls[df_wls["UnixTimeMillis"].isin(timestamps)].copy()
        trip_wls = trip_wls.drop_duplicates(subset=["UnixTimeMillis"])

        # Merge WLS
        merged = pd.merge(
            merged,
            trip_wls[
                [
                    "UnixTimeMillis",
                    "WlsPositionXEcefMeters",
                    "WlsPositionYEcefMeters",
                    "WlsPositionZEcefMeters",
                ]
            ],
            on="UnixTimeMillis",
            how="left",
        )

        # If WLS missing, interpolate or forward fill?
        # WLS is usually dense. If missing, we can't define local frame well.
        # But we can use the first valid point as reference for the whole trip.

        if merged.empty:
            return trip_ml  # Should not happen

        # Define Reference Point (First valid WLS or ML)
        # Using first ML point as origin for ENU is safer if WLS is gaps
        ref_lat = merged["LatitudeDegrees"].iloc[0]
        ref_lon = merged["LongitudeDegrees"].iloc[0]

        # Approximate altitude if WLS missing
        # Calculate WLS LLH
        wls_xyz = merged[
            [
                "WlsPositionXEcefMeters",
                "WlsPositionYEcefMeters",
                "WlsPositionZEcefMeters",
            ]
        ].values

        # Fill missing WLS with 0 (will result in bad alt, but we only care about Lat/Lon)
        # Better: use mean of valid WLS
        valid_wls = ~np.isnan(wls_xyz[:, 0])
        if np.sum(valid_wls) > 0:
            mean_x = np.mean(wls_xyz[valid_wls, 0])
            mean_y = np.mean(wls_xyz[valid_wls, 1])
            mean_z = np.mean(wls_xyz[valid_wls, 2])
            # Get ref alt
            _, _, ref_alt = ecef_to_llh(mean_x, mean_y, mean_z)
        else:
            ref_alt = 0.0

        # Convert ML Lat/Lon to ENU
        anchors_enu = []
        for i in range(len(merged)):
            lat = merged["LatitudeDegrees"].iloc[i]
            lon = merged["LongitudeDegrees"].iloc[i]
            # Use ref_alt for projection surface
            x, y, z = llh_to_ecef(lat, lon, ref_alt)
            e, n, u = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
            anchors_enu.append([e, n, u])

        anchors_enu = np.array(anchors_enu)

        # Prepare Odometry arrays
        odom_deltas = merged[["dE", "dN", "dU"]].values
        odom_weights = merged["weight_odom"].values

        # Initial Guess: ML Anchors
        x0 = anchors_enu.flatten()

        # Optimize
        res = minimize(
            fun=self._objective_function,
            x0=x0,
            args=(anchors_enu, odom_deltas, odom_weights, len(merged)),
            method="L-BFGS-B",
            jac=self._jacobian,
            options={"maxiter": 1500, "disp": False},
        )

        # Convert result back to Lat/Lon
        opt_enu = res.x.reshape((-1, 3))

        opt_lats = []
        opt_lons = []

        for i in range(len(opt_enu)):
            e, n, u = opt_enu[i]
            x, y, z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)
            lat, lon, _ = ecef_to_llh(x, y, z)
            opt_lats.append(lat)
            opt_lons.append(lon)

        # Update DataFrame
        result_df = trip_ml.copy()
        result_df["LatitudeDegrees"] = opt_lats
        result_df["LongitudeDegrees"] = opt_lons

        return result_df


def run_global_optimization(load_cached_data=True):
    """
    Main entry point for the optimization module.
    1. Ensures ML predictions exist.
    2. Loads Test Metadata.
    3. Computes Kinematics.
    4. Runs Graph Optimization per trip.
    5. Saves final submission.
    """
    print("\nStarting Global Graph Optimization...")

    # 1. Ensure ML predictions exist
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        print("ML submission not found. Generating...")
        generate_submission(load_cached_data=load_cached_data)

    df_ml = pd.read_csv(submission_path)
    print(f"Loaded ML predictions: {len(df_ml)} rows")

    # 2. Load Test Metadata to get drive/phone mapping
    test_meta = load_metadata("test")

    # 3. Process per Trip
    optimizer = GraphOptimizer()
    final_dfs = []

    unique_trips = df_ml["tripId"].unique()

    for i, trip_id in enumerate(unique_trips):
        # Get metadata for this trip
        trip_meta = test_meta[test_meta["tripId"] == trip_id].iloc[0]
        drive_id = trip_meta["drive_id"]
        phone_name = trip_meta["phone_name"]
        gnss_path = trip_meta["gnss_path"]
        imu_path = trip_meta["imu_path"]

        # Load/Compute Kinematics
        # Note: kinematics returns data for ALL GNSS epochs.
        # We will filter inside optimize_trip.
        df_odom = compute_tdcp_odometry(
            drive_id, phone_name, gnss_path, imu_path, load_cached_data=load_cached_data
        )

        # Load WLS for Altitude
        data = load_drive_data(
            drive_id, phone_name, gnss_path, imu_path, load_cached_data=load_cached_data
        )
        df_wls = data["gnss"].rename(columns={"utcTimeMillis": "UnixTimeMillis"})

        # Optimize
        print(f"Optimizing trip {i+1}/{len(unique_trips)}: {trip_id}")
        df_opt = optimizer.optimize_trip(
            trip_id, drive_id, phone_name, df_ml, df_odom, df_wls
        )

        final_dfs.append(df_opt)

    # 4. Concatenate and Save
    final_submission = pd.concat(final_dfs, ignore_index=True)

    # Ensure correct column order for submission
    final_submission = final_submission[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    # Sort by trip and time just in case
    final_submission = final_submission.sort_values(["tripId", "UnixTimeMillis"])

    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    final_submission.to_csv(output_path, index=False)
    print(f"Final optimized submission saved to {output_path}")

    return final_submission
