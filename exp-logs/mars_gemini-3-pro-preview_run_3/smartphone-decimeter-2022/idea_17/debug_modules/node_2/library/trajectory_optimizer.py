import numpy as np
import pandas as pd
import os
from scipy.optimize import least_squares
from library.utils import GeodeticToEcef, EcefToGeodetic


class GraphOptimizer:
    """
    Constructs and solves a Factor Graph to fuse ML-predicted anchors with
    Robust Odometry constraints (RTGO Strategy).

    The optimization minimizes a cost function consisting of:
    1. Anchor Residuals: Huber loss between optimized position and ML prediction.
    2. Odometry Residuals: Weighted L2 loss between relative displacement and TDCP/Doppler measurement.
    """

    def __init__(self, cache_dir="./working/idea_17"):
        """
        Args:
            cache_dir (str): Directory to store cached optimization results.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Hyperparameters
        self.anchor_weight = 1.0
        # Scale odometry weight: High confidence in carrier phase (cm-level precision)
        # vs ML prediction (meter-level precision).
        self.odom_weight_scale = 50.0
        # Huber loss scale parameter (meters). Errors below this are quadratic, above are linear.
        self.huber_scale = 10.0

    def _optimize_trip(self, trip_id, df_trip):
        """
        Optimizes a single trip trajectory using scipy.optimize.least_squares.

        Args:
            trip_id (str): Identifier for the trip.
            df_trip (pd.DataFrame): Data for the trip containing anchors and odometry.

        Returns:
            pd.DataFrame: Optimized trajectory.
        """
        # Sort by time to ensure sequential constraints are applied correctly
        df_trip = df_trip.sort_values("UnixTimeMillis").reset_index(drop=True)

        n = len(df_trip)
        # Need at least 2 points for odometry constraints to make sense
        if n < 2:
            return df_trip[
                ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
            ]

        # 1. Prepare Anchors (ML Predictions)
        # Convert Geodetic (Lat, Lon, 0) to ECEF.
        # We assume Altitude=0 for the anchor prior as we primarily optimize horizontal position.
        # The optimization happens in ECEF (3D), but is constrained by 2D anchors + 3D odometry.
        lat_anc = df_trip["LatitudeDegrees"].values
        lon_anc = df_trip["LongitudeDegrees"].values
        alt_anc = np.zeros_like(lat_anc)

        x_anc, y_anc, z_anc = GeodeticToEcef.transform(lat_anc, lon_anc, alt_anc)

        # 2. Prepare Odometry Constraints
        # dx, dy, dz are displacements from t-1 to t in ECEF
        dx_odom = df_trip["dx_ecef"].values
        dy_odom = df_trip["dy_ecef"].values
        dz_odom = df_trip["dz_ecef"].values

        # Weights: 0 for first point (no prev), else derived from RANSAC inliers
        w_odom = df_trip["weight"].values * self.odom_weight_scale

        # 3. Initial Guess
        # Use ML anchors as the initial guess for the solver
        x0 = np.concatenate([x_anc, y_anc, z_anc])

        # 4. Define Residual Function
        # The solver minimizes sum(residuals^2).
        # We define residuals such that:
        # Anchor term: (x - x_anc)
        # Odom term: (x_t - x_{t-1} - dx)

        def fun(x):
            # Unpack state vector (flattened [x_0...x_n, y_0...y_n, z_0...z_n])
            xs = x[:n]
            ys = x[n : 2 * n]
            zs = x[2 * n :]

            # Anchor Residuals: Pull towards ML prediction
            # We apply constant weight. Huber loss handling is done by least_squares 'loss' parameter.
            rx_anc = (xs - x_anc) * self.anchor_weight
            ry_anc = (ys - y_anc) * self.anchor_weight
            rz_anc = (zs - z_anc) * self.anchor_weight

            # Odometry Residuals: Maintain shape
            # Constraint: Pos[t] - Pos[t-1] = Delta[t]
            # Residual: Pos[t] - Pos[t-1] - Delta[t]
            # We skip index 0 for odometry residuals as there is no t-1
            rx_odom = (xs[1:] - xs[:-1] - dx_odom[1:]) * np.sqrt(w_odom[1:])
            ry_odom = (ys[1:] - ys[:-1] - dy_odom[1:]) * np.sqrt(w_odom[1:])
            rz_odom = (zs[1:] - zs[:-1] - dz_odom[1:]) * np.sqrt(w_odom[1:])

            return np.concatenate([rx_anc, ry_anc, rz_anc, rx_odom, ry_odom, rz_odom])

        # 5. Solve
        # 'trf' is a robust Trust Region Reflective algorithm suitable for large sparse problems.
        # 'huber' loss makes it robust against large outliers in ML anchors (e.g. multipath jumps).
        # Odometry outliers are already filtered by RANSAC in the previous stage.
        res = least_squares(
            fun, x0, loss="huber", f_scale=self.huber_scale, method="trf", verbose=0
        )

        # 6. Extract Result
        x_opt = res.x[:n]
        y_opt = res.x[n : 2 * n]
        z_opt = res.x[2 * n :]

        # Convert back to Geodetic
        lat_opt, lon_opt, _ = EcefToGeodetic.transform(x_opt, y_opt, z_opt)

        # Return optimized dataframe
        res_df = pd.DataFrame(
            {
                "tripId": df_trip["tripId"],
                "UnixTimeMillis": df_trip["UnixTimeMillis"],
                "LatitudeDegrees": lat_opt,
                "LongitudeDegrees": lon_opt,
            }
        )

        return res_df

    def run(self, anchors_df, odom_df, load_cached_data=True):
        """
        Runs the optimization for all trips provided in the dataframes.

        Args:
            anchors_df (pd.DataFrame): DataFrame with ML predictions (LatitudeDegrees, LongitudeDegrees).
            odom_df (pd.DataFrame): DataFrame with Odometry constraints (dx_ecef, dy_ecef, dz_ecef, weight).
            load_cached_data (bool): Whether to load result from cache if available.

        Returns:
            pd.DataFrame: Optimized trajectory containing LatitudeDegrees and LongitudeDegrees.
        """
        cache_file = os.path.join(self.cache_dir, "optimized_trajectory.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print("Loading cached optimized trajectories...")
            return pd.read_parquet(cache_file)

        print("Starting Factor Graph Optimization...")

        # Ensure timestamp types match for merging
        anchors_df["UnixTimeMillis"] = anchors_df["UnixTimeMillis"].astype(int)
        odom_df["UnixTimeMillis"] = odom_df["UnixTimeMillis"].astype(int)

        # Merge Data
        # We perform a left join on anchors to ensure we output predictions for all required timestamps.
        # Odometry data might have gaps; these will be filled with 0 weight (no constraint).
        merged = pd.merge(
            anchors_df[
                ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
            ],
            odom_df[
                ["tripId", "UnixTimeMillis", "dx_ecef", "dy_ecef", "dz_ecef", "weight"]
            ],
            on=["tripId", "UnixTimeMillis"],
            how="left",
        )

        # Fill missing odometry with 0 (indicates no constraint for that step)
        merged["dx_ecef"] = merged["dx_ecef"].fillna(0.0)
        merged["dy_ecef"] = merged["dy_ecef"].fillna(0.0)
        merged["dz_ecef"] = merged["dz_ecef"].fillna(0.0)
        merged["weight"] = merged["weight"].fillna(0.0)

        results = []

        # Process each trip independently
        unique_trips = merged["tripId"].unique()
        print(f"Optimizing {len(unique_trips)} trips...")

        # Iterate without tqdm to comply with requirements
        for trip_id in unique_trips:
            trip_data = merged[merged["tripId"] == trip_id].copy()
            opt_data = self._optimize_trip(trip_id, trip_data)
            results.append(opt_data)

        final_df = pd.concat(results, ignore_index=True)

        # Cache result
        final_df.to_parquet(cache_file)
        print("Optimization complete.")

        return final_df


def save_submission(df, output_path="./submission/submission.csv"):
    """
    Saves the dataframe to the submission format required by the competition.

    Args:
        df (pd.DataFrame): DataFrame containing predictions.
        output_path (str): Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Ensure columns order
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    df[cols].to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
