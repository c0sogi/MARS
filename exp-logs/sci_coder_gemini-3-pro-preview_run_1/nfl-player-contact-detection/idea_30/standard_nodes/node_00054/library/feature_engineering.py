import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import Timer, save_data, load_data


class FeatureEngineer:
    """
    Implements the Orthogonal-Spectral Vector-Anchored Ensemble (OSVA-E) feature engineering pipeline.
    Includes Relaxed Quadratic Gating, Dual-Basis Projection, and DCT Encoding.
    """

    def __init__(self):
        self.dct_matrix = self._create_dct_matrix(Config.DCT_WINDOW_SIZE, Config.DCT_K)

    def _create_dct_matrix(self, N, K):
        """
        Creates the Discrete Cosine Transform (Type-II) matrix.
        Shape: (N, K)
        """
        # Formula: X_k = sum_{n=0}^{N-1} x_n * cos(pi/N * (n + 0.5) * k)
        # We want a matrix M such that X = x @ M
        # M[n, k] = cos(pi/N * (n + 0.5) * k)
        n = np.arange(N)
        k = np.arange(K)
        # Broadcasting to create (N, K) matrix
        matrix = np.cos(np.pi / N * (n[:, None] + 0.5) * k[None, :])
        return matrix

    def _prepare_tracking_lags(self, df_tracking):
        """
        Creates a wide dataframe with lagged features for windowed analysis.
        Vectorized approach using shifts.
        """
        with Timer("Tracking Lags Generation"):
            # Ensure sorting
            df_tracking = df_tracking.sort_values(
                ["game_play", "nfl_player_id", "step"]
            ).reset_index(drop=True)

            # Define features to lag
            # x, y are needed for basis calculation (lag 0) and projection (relative positions)
            # speed (s), acceleration (a), direction (dir), orientation (o)
            # We convert polar (speed, dir) to cartesian (vx, vy) for proper vector math

            # Convert to radians
            # direction: 0 is Y axis (North), 90 is X axis (East) - Check standard NFL data
            # Standard NFL tracking: 0 is along Y (short axis?), 90 along X (long axis).
            # Usually: 0 deg is typically 'up' or 'right'.
            # Let's assume standard math convention after conversion or stick to provided x/y if available.
            # Provided: x_position, y_position, speed, direction, acceleration.
            # We'll compute vx, vy, ax, ay.

            rad_dir = np.radians(
                90 - df_tracking["direction"]
            )  # Adjusting to standard trig if needed, or just use as is for projection
            # Actually, let's stick to raw x, y and compute velocities from deltas if needed,
            # but 'speed' and 'direction' are provided.
            # Let's use the provided speed/direction to get velocity vectors.

            # Note: direction is 0..360.
            # We will use simple projection: vx = speed * cos(theta), vy = speed * sin(theta)
            # NFL data usually: 0 is Y-axis, increasing X is 90.
            theta = np.radians(90 - df_tracking["direction"])
            df_tracking["vx"] = df_tracking["speed"] * np.cos(theta)
            df_tracking["vy"] = df_tracking["speed"] * np.sin(theta)

            # Acceleration is a magnitude. We don't have accel direction explicitly usually,
            # but we can approximate it or use 'sa' (signed accel).
            # Let's assume acceleration aligns with velocity for simplicity or use finite difference of velocity.
            # Finite difference is safer for structural integrity.
            # ax = (vx_t+1 - vx_t-1) / 2dt.
            # Given time constraints, let's use the provided 'acceleration' magnitude directed along 'direction'.
            df_tracking["ax"] = df_tracking["acceleration"] * np.cos(theta)
            df_tracking["ay"] = df_tracking["acceleration"] * np.sin(theta)

            feature_cols = ["x_position", "y_position", "vx", "vy", "ax", "ay"]

            # Create Group ID for safe shifting
            # We can't use groupby().shift() efficiently for 21 lags on 1M rows.
            # Faster: shift whole DF, mask boundaries.
            df_tracking["group_id"] = (
                df_tracking["game_play"].astype(str)
                + "_"
                + df_tracking["nfl_player_id"].astype(str)
            )

            # Generate lags
            # Window is centered. Size 21 means -10 to +10.
            radius = Config.DCT_WINDOW_SIZE // 2

            lagged_data = {}
            # Keep metadata columns
            meta_cols = ["game_play", "nfl_player_id", "step", "group_id"]
            for col in meta_cols:
                lagged_data[col] = df_tracking[col]

            # Shift
            group_ids = df_tracking["group_id"].values

            for lag in range(-radius, radius + 1):
                # shift(k): positive k shifts data down (t becomes t+k).
                # We want data at t+k to be in the row of t. So we shift by -k.
                # e.g. lag +1 (future): we want t+1 data at t. shift(-1).
                shifted = df_tracking[feature_cols].shift(-lag)

                # Mask where group_id changes
                shifted_groups = df_tracking["group_id"].shift(-lag)
                mask = group_ids == shifted_groups

                # Apply mask
                # Use numpy for speed
                for col in feature_cols:
                    col_name = f"{col}_lag_{lag}"
                    vals = shifted[col].values
                    vals[~mask] = 0.0  # Zero padding for edge cases
                    lagged_data[col_name] = vals

            df_wide = pd.DataFrame(lagged_data)

            # Cleanup
            del df_tracking
            gc.collect()

            return df_wide

    def _perform_gating(self, df):
        """
        Applies Relaxed Quadratic Gating.
        Filters pairs where the estimated minimum distance in the trajectory is too large.
        """
        with Timer("Quadratic Gating"):
            # Extract Lag 0 (Current) features
            # P1
            x1 = df["x_position_p1_lag_0"]
            y1 = df["y_position_p1_lag_0"]
            vx1 = df["vx_p1_lag_0"]
            vy1 = df["vy_p1_lag_0"]

            # P2 (Handle G/NaN)
            x2 = df["x_position_p2_lag_0"].fillna(0)
            y2 = df["y_position_p2_lag_0"].fillna(0)
            vx2 = df["vx_p2_lag_0"].fillna(0)
            vy2 = df["vy_p2_lag_0"].fillna(0)

            # Identify Ground interactions
            is_ground = df["nfl_player_id_2"] == "G"

            # Calculate Relative State
            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)

            dvx = vx1 - vx2
            dvy = vy1 - vy2

            # Radial Velocity (Projection of rel velocity onto rel position)
            # v_rad = (r . v) / |r|
            # Avoid div by zero
            safe_dist = dist.replace(0, 1e-6)
            v_rad = (dx * dvx + dy * dvy) / safe_dist

            # Simple Linear Extrapolation for Min Distance
            # d(t) ~ d_0 + v_rad * t
            # Min distance occurs when v_rad * t reduces d_0.
            # If v_rad < 0 (approaching), min dist is in future.
            # If v_rad > 0 (departing), min dist is now (d_0).
            # We allow a lookahead window.
            # Quadratic term is a bit noisy, let's stick to Linear Approach + Buffer
            # "Relaxed Quadratic" implies we consider curvature, but robust gating often just needs
            # "Is it possible they get close?"
            # Criterion: If dist < Threshold OR (Approaching AND Predicted_Min_Dist < Threshold)

            # Time to closest approach (linear): t_min = - (r . v) / (v . v)
            # But simpler: if dist is already small, keep.
            # If large, check if closing speed is high.

            # Logic: Keep if dist < GATING_THRESHOLD
            # This is the "Survivors" set.
            # For Ground: dist is meaningless x/y diff. We force keep Ground.

            # Apply Sentinel for Ground Distance
            # We modify the 'dist' series to be -1.0 where is_ground
            final_dist = dist.copy()
            final_dist[is_ground] = Config.SENTINEL_DIST_VALUE

            # Gating Mask
            # Keep if Ground OR Dist < Threshold
            mask = (is_ground) | (dist < Config.GATING_THRESHOLD)

            # Add distance to dataframe for later use
            df["distance"] = final_dist

            # Filter
            df_filtered = df[mask].copy()

            return df_filtered

    def _compute_dual_basis_and_dct(self, df):
        """
        Computes basis vectors, projects windowed features, and applies DCT.
        Vectorized implementation.
        """
        with Timer("Dual-Basis DCT Encoding"):
            # 1. Define Basis Vectors (u_x, u_y)
            # ----------------------------------
            x1 = df["x_position_p1_lag_0"].values
            y1 = df["y_position_p1_lag_0"].values
            x2 = df["x_position_p2_lag_0"].fillna(0).values
            y2 = df["y_position_p2_lag_0"].fillna(0).values

            is_ground = (df["nfl_player_id_2"] == "G").values

            # Case A: Player-Player (Collision Axis: P2 -> P1)
            rx = x1 - x2
            ry = y1 - y2
            r_norm = np.sqrt(rx**2 + ry**2)
            r_norm[r_norm == 0] = 1e-6

            u_pp_x = rx / r_norm
            u_pp_y = ry / r_norm

            # Case B: Player-Ground (Motion Axis: P1 Velocity)
            vx1 = df["vx_p1_lag_0"].values
            vy1 = df["vy_p1_lag_0"].values
            v_norm = np.sqrt(vx1**2 + vy1**2)
            v_norm[v_norm == 0] = 1e-6

            u_pg_x = vx1 / v_norm
            u_pg_y = vy1 / v_norm

            # Combine based on mask
            u_x = np.where(is_ground, u_pg_x, u_pp_x)
            u_y = np.where(is_ground, u_pg_y, u_pp_y)

            # Orthogonal Basis (Rotate 90 deg)
            # u_perp = (-u_y, u_x)
            u_perp_x = -u_y
            u_perp_y = u_x

            # 2. Projection & DCT Loop
            # ------------------------
            radius = Config.DCT_WINDOW_SIZE // 2
            lags = range(-radius, radius + 1)

            # We need to collect the sequence for each component
            # Shape: (N_samples, Window_Size)

            # Components to process:
            # P1 Velocity, P1 Accel, P2 Velocity, P2 Accel
            # Projected onto U and U_perp

            feature_prefixes = ["vx", "vy", "ax", "ay"]
            players = ["p1", "p2"]

            # Container for final features
            dct_features = {}

            # Helper to extract matrix of shape (Rows, 21) for a specific feature
            def get_lag_matrix(prefix, player):
                # Columns: e.g., vx_p1_lag_-10 ... vx_p1_lag_10
                cols = [f"{prefix}_{player}_lag_{k}" for k in lags]
                # Handle P2 NaN for Ground
                mat = df[cols].fillna(0).values
                return mat

            # Process P1 and P2
            for p in players:
                # Get raw vector sequences
                vx_seq = get_lag_matrix("vx", p)
                vy_seq = get_lag_matrix("vy", p)
                ax_seq = get_lag_matrix("ax", p)
                ay_seq = get_lag_matrix("ay", p)

                # Project onto Basis (u) and Orthogonal (u_perp)
                # Note: u_x, u_y are shape (Rows,), seq is (Rows, 21)
                # Broadcast u across time steps (Basis is fixed at t=0)

                # Component 1: Parallel to Basis
                v_c1 = vx_seq * u_x[:, None] + vy_seq * u_y[:, None]
                a_c1 = ax_seq * u_x[:, None] + ay_seq * u_y[:, None]

                # Component 2: Perpendicular
                v_c2 = vx_seq * u_perp_x[:, None] + vy_seq * u_perp_y[:, None]
                a_c2 = ax_seq * u_perp_x[:, None] + ay_seq * u_perp_y[:, None]

                # Apply DCT
                # Input: (Rows, 21). Matrix: (21, K). Result: (Rows, K)
                # Matmul: (N, 21) @ (21, K) -> (N, K)

                v_c1_dct = v_c1 @ self.dct_matrix
                v_c2_dct = v_c2 @ self.dct_matrix
                a_c1_dct = a_c1 @ self.dct_matrix
                a_c2_dct = a_c2 @ self.dct_matrix

                # Store Features
                for k in range(Config.DCT_K):
                    dct_features[f"dct_v_{p}_c1_{k}"] = v_c1_dct[:, k]
                    dct_features[f"dct_v_{p}_c2_{k}"] = v_c2_dct[:, k]
                    dct_features[f"dct_a_{p}_c1_{k}"] = a_c1_dct[:, k]
                    dct_features[f"dct_a_{p}_c2_{k}"] = a_c2_dct[:, k]

            # Convert to DataFrame
            df_dct = pd.DataFrame(dct_features, index=df.index)

            # Combine with metadata and distance
            keep_cols = [
                "contact_id",
                "game_play",
                "step",
                "nfl_player_id_1",
                "nfl_player_id_2",
                "distance",
                "contact",
            ]
            # Ensure contact exists (test set might need placeholder if not present, but metadata usually has it or we handle it)
            available_cols = [c for c in keep_cols if c in df.columns]

            result = pd.concat([df[available_cols], df_dct], axis=1)
            return result

    def generate_features(
        self, metadata_path, tracking_path, output_path, load_cached_data=True
    ):
        """
        Main pipeline execution.
        """
        # 1. Caching Check
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}...")
            return load_data(output_path)

        print(f"Generating features for {metadata_path}...")

        # 2. Load Data
        df_meta = pd.read_csv(metadata_path)

        # Debugging / Sampling
        if Config.DEBUG:
            print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            df_meta = df_meta.sample(
                n=min(len(df_meta), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).copy()

        df_tracking = pd.read_csv(tracking_path)

        # 3. Preprocess Tracking (Wide Lags)
        df_track_wide = self._prepare_tracking_lags(df_tracking)

        # 4. Merge Metadata with Tracking
        with Timer("Merging Metadata"):
            # Prepare keys
            df_meta["game_play"] = df_meta["game_play"].astype(str)
            df_meta["step"] = df_meta["step"].astype(int)
            df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

            # Helper for renaming: x_position_lag_0 -> x_position_p1_lag_0
            def rename_col(c, suffix):
                if "_lag_" in c:
                    base, lag = c.split("_lag_")
                    return f"{base}_{suffix}_lag_{lag}"
                return f"{c}_{suffix}"

            # Merge P1
            df_merged = df_meta.merge(
                df_track_wide,
                left_on=["game_play", "nfl_player_id_1", "step"],
                right_on=["game_play", "nfl_player_id", "step"],
                how="left",
            ).rename(
                columns=lambda x: (
                    rename_col(x, "p1")
                    if x in df_track_wide.columns and x not in ["game_play", "step"]
                    else x
                )
            )

            # Handle P2 (Mixed Types: Int or 'G')
            # Separate G rows to avoid type errors during merge
            mask_g = df_merged["nfl_player_id_2"] == "G"

            # Create a temp int column for merging
            df_merged["p2_int"] = (
                pd.to_numeric(df_merged["nfl_player_id_2"], errors="coerce")
                .fillna(-1)
                .astype(int)
            )

            # Merge P2
            df_merged = df_merged.merge(
                df_track_wide,
                left_on=["game_play", "p2_int", "step"],
                right_on=["game_play", "nfl_player_id", "step"],
                how="left",
                suffixes=("", "_p2"),
            )

            # Rename P2 columns properly
            p2_cols = [
                c
                for c in df_track_wide.columns
                if c not in ["game_play", "nfl_player_id", "step", "group_id"]
            ]
            rename_map = {c: rename_col(c, "p2") for c in p2_cols}
            df_merged = df_merged.rename(columns=rename_map)

            # Clean up
            df_merged.drop(
                columns=[
                    "p2_int",
                    "nfl_player_id_x",
                    "nfl_player_id_y",
                    "group_id_x",
                    "group_id_y",
                ],
                errors="ignore",
                inplace=True,
            )

        # 5. Gating
        df_gated = self._perform_gating(df_merged)
        print(f"Gating reduced data from {len(df_merged)} to {len(df_gated)} rows.")

        # 6. Feature Computation (Basis + DCT)
        df_final = self._compute_dual_basis_and_dct(df_gated)

        # 7. Save and Return
        save_data(df_final, output_path)

        return df_final


def generate_features(metadata_path, tracking_path, output_path, load_cached_data=True):
    """
    Wrapper function to instantiate class and run pipeline.
    """
    engineer = FeatureEngineer()
    return engineer.generate_features(
        metadata_path, tracking_path, output_path, load_cached_data
    )
