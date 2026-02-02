import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from library.config import WORKING_DIR, HUBER_DELTA, ODOM_WEIGHT, SEED, SUBMISSION_DIR
from library.utils import wgs84_to_ecef, ecef_to_wgs84, process_with_cache
from library.data_loader import load_dataset


class GraphOptimizer:
    def __init__(self):
        self.huber_delta = HUBER_DELTA
        self.odom_weight_base = ODOM_WEIGHT

    def _huber_loss(self, residuals):
        """
        Computes Huber loss for a vector of residuals.
        L(r) = 0.5 * r^2               if |r| <= delta
             = delta * (|r| - 0.5*delta)  if |r| > delta
        """
        abs_r = np.abs(residuals)
        quadratic = np.minimum(abs_r, self.huber_delta)
        linear = abs_r - quadratic
        return 0.5 * quadratic**2 + self.huber_delta * linear

    def _objective_function(self, x_flat, anchors, odom_deltas, odom_weights):
        """
        Objective function for the optimization.
        J(X) = Sum(Huber(X_t - Anchor_t)) + Sum(Weight_t * ||(X_t - X_{t-1}) - Odom_t||^2)

        Args:
            x_flat: Flattened state vector (N*3,).
            anchors: Array of anchor positions (N, 3).
            odom_deltas: Array of odometry deltas (N-1, 3).
            odom_weights: Array of odometry weights (N-1,).
        """
        n_points = len(anchors)
        X = x_flat.reshape((n_points, 3))

        # 1. Anchor Cost (Huber)
        # Calculate component-wise residuals
        anchor_residuals = X - anchors
        # Sum of Huber loss over all components and all points
        anchor_cost = np.sum(self._huber_loss(anchor_residuals))

        # 2. Odometry Cost (Weighted L2)
        # Calculate relative motion: X_t - X_{t-1}
        rel_motion = X[1:] - X[:-1]

        # Residual: Estimated Motion - Measured Odometry
        odom_residuals = rel_motion - odom_deltas

        # Squared Euclidean norm of residuals: sum(dx^2 + dy^2 + dz^2)
        squared_errors = np.sum(odom_residuals**2, axis=1)

        # Weighted sum
        odom_cost = np.sum(odom_weights * squared_errors)

        return anchor_cost + odom_cost

    def _optimize_trip(self, args):
        """
        Worker function to optimize a single trip.
        """
        trip_id, df_trip = args

        # Sort by time to ensure sequential order
        df_trip = df_trip.sort_values("UnixTimeMillis").reset_index(drop=True)

        # Extract Anchors (ML Preds converted to ECEF)
        # We need WLS Altitude to convert ML Lat/Lon to ECEF
        # Assuming df_trip has merged WLS info

        # Fallback for missing WLS altitude? Use 0 or mean.
        # WLS columns should be present from load_dataset merge.
        wls_x = df_trip["WlsPositionXEcefMeters"].values
        wls_y = df_trip["WlsPositionYEcefMeters"].values
        wls_z = df_trip["WlsPositionZEcefMeters"].values

        # Convert WLS to LLA to get Altitude
        _, _, wls_alt = ecef_to_wgs84(wls_x, wls_y, wls_z)

        # Convert ML Lat/Lon + WLS Alt to ECEF Anchors
        ml_lat = df_trip["lat_pred"].values
        ml_lon = df_trip["lon_pred"].values

        anchor_x, anchor_y, anchor_z = wgs84_to_ecef(ml_lat, ml_lon, wls_alt)
        anchors = np.column_stack((anchor_x, anchor_y, anchor_z))

        # Extract Odometry
        # odom_x/y/z at index i is displacement from i-1 to i.
        # So for N points, we have N entries, but the first one is usually 0/invalid for delta.
        # We need deltas for indices 1 to N-1.
        odom_deltas = df_trip[["odom_x", "odom_y", "odom_z"]].values[1:]

        # Extract Reliability and compute weights
        # Weight = Base_Lambda * Reliability
        reliability = df_trip["reliability"].values[1:]
        odom_weights = self.odom_weight_base * reliability

        # Initial Guess: Use ML Anchors
        x0 = anchors.flatten()

        # Optimization
        # L-BFGS-B is memory efficient and handles bound constraints if we needed them (we don't here)
        # It approximates the Hessian which is sufficient.
        res = minimize(
            self._objective_function,
            x0,
            args=(anchors, odom_deltas, odom_weights),
            method="L-BFGS-B",
            options={"maxiter": 1000, "disp": False},
        )

        # Reshape result
        X_opt = res.x.reshape((-1, 3))

        # Convert back to LLA
        opt_lat, opt_lon, _ = ecef_to_wgs84(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2])

        # Create result DataFrame
        df_res = pd.DataFrame(
            {
                "tripId": df_trip["tripId"],
                "UnixTimeMillis": df_trip["UnixTimeMillis"],
                "LatitudeDegrees": opt_lat,
                "LongitudeDegrees": opt_lon,
            }
        )

        return df_res

    def optimize_trajectory(self, df_ml, df_odom, split="test", max_drives=None):
        """
        Main driver to run optimization on all trips.
        """
        print(f"Starting Graph Optimization for split: {split}")

        # 1. Load Raw Dataset (for WLS positions)
        # We need WLS ECEF coordinates to construct 3D anchors from 2D ML predictions
        df_raw = load_dataset(split, load_cached_data=True, max_drives=max_drives)

        # Keep only necessary columns from raw data
        cols_raw = [
            "tripId",
            "UnixTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ]

        # The raw dataset might have multiple rows per epoch (one per sat).
        # We need to drop duplicates to get one row per epoch.
        df_raw_unique = df_raw[cols_raw].drop_duplicates(
            subset=["tripId", "UnixTimeMillis"]
        )

        # 2. Merge Data
        # Merge ML predictions with Raw WLS
        df_merged = pd.merge(
            df_ml, df_raw_unique, on=["tripId", "UnixTimeMillis"], how="inner"
        )

        # Merge with Odometry
        # Odometry should exist for these timestamps
        df_merged = pd.merge(
            df_merged, df_odom, on=["tripId", "UnixTimeMillis"], how="left"
        )

        # Fill missing odometry with 0 (no constraint)
        df_merged["odom_x"] = df_merged["odom_x"].fillna(0.0)
        df_merged["odom_y"] = df_merged["odom_y"].fillna(0.0)
        df_merged["odom_z"] = df_merged["odom_z"].fillna(0.0)
        df_merged["reliability"] = df_merged["reliability"].fillna(0.0)

        # 3. Process Trips
        trips = list(df_merged.groupby("tripId"))
        print(f"Optimizing {len(trips)} trips...")

        results = []

        # Use Parallel Processing
        # We pass 'self' (the optimizer instance) which contains the config parameters
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = executor.map(self._optimize_trip, trips)

            for res_df in tqdm(
                futures, total=len(trips), desc="Optimizing Trajectories"
            ):
                results.append(res_df)

        final_df = pd.concat(results, ignore_index=True)
        return final_df


def process_optimization(df_ml, df_odom, split, load_cached_data=True, max_drives=None):
    """
    Wrapper for caching the optimization result.
    """
    optimizer = GraphOptimizer()

    def _compute():
        return optimizer.optimize_trajectory(df_ml, df_odom, split, max_drives)

    suffix = f"_{max_drives}" if max_drives else ""
    cache_name = f"optimized_{split}{suffix}.parquet"

    return process_with_cache(cache_name, _compute, load_cached_data=load_cached_data)
