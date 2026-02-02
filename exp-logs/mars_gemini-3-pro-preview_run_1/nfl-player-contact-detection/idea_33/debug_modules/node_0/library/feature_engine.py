import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import CacheManager, reduce_mem_usage
from library.data_manager import DataManager


class FeatureEngine:
    """
    Implements the Decoupled-Entity Interaction-Basis Anchored-Mining Ensemble (DEIB-AME)
    feature engineering pipeline.

    Includes:
    1. Basis Projection (Collision Axis vs Motion Axis).
    2. Decoupled Vector Decomposition.
    3. Explicit Time-Domain Lag Generation.
    4. Relaxed Quadratic Reachability Gating.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        self.data_manager = DataManager()

        # Features to be lagged
        self.lag_features = [
            "distance",
            "proj_u_p1",
            "proj_v_p1",
            "acc_u_p1",
            "jerk_u_p1",
            "speed_p1",
            "proj_u_p2",
            "proj_v_p2",
            "acc_u_p2",
            "jerk_u_p2",
            "speed_p2",
            "orientation_u_p1",
            "orientation_u_p2",
        ]

    def process_train(self, load_cached_data=True, sample_size=None):
        """
        Generates features for the training set.
        """
        cache_name = "features_train.parquet"
        if sample_size:
            cache_name = f"features_train_sample_{sample_size}.parquet"

        # Check cache first
        if load_cached_data and self.cache_manager.exists(cache_name):
            print(f"Loading cached training features from {cache_name}...")
            return self.cache_manager.load(cache_name)

        # Load raw merged data
        df = self.data_manager.load_train_data(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        # Generate features
        df_features = self._generate_features(df)

        # Save to cache
        self.cache_manager.save(df_features, cache_name)

        return df_features

    def process_val(self, load_cached_data=True, sample_size=None):
        """
        Generates features for the validation set.
        """
        cache_name = "features_val.parquet"
        if sample_size:
            cache_name = f"features_val_sample_{sample_size}.parquet"

        if load_cached_data and self.cache_manager.exists(cache_name):
            print(f"Loading cached validation features from {cache_name}...")
            return self.cache_manager.load(cache_name)

        df = self.data_manager.load_val_data(
            load_cached_data=load_cached_data, sample_size=sample_size
        )
        df_features = self._generate_features(df)
        self.cache_manager.save(df_features, cache_name)

        return df_features

    def process_test(self, load_cached_data=True, sample_size=None):
        """
        Generates features for the test set.
        """
        cache_name = "features_test.parquet"

        if load_cached_data and self.cache_manager.exists(cache_name):
            print(f"Loading cached test features from {cache_name}...")
            return self.cache_manager.load(cache_name)

        df = self.data_manager.load_test_data(
            load_cached_data=load_cached_data, sample_size=sample_size
        )
        df_features = self._generate_features(df)
        self.cache_manager.save(df_features, cache_name)

        return df_features

    def _generate_features(self, df):
        """
        Core logic for feature engineering.
        """
        print("Starting feature engineering...")

        # 1. Sort for temporal continuity
        # Create a unique group identifier for sorting and shifting
        # Note: nfl_player_id_2 can be 'G', so we treat it as string in the ID
        df["group_id"] = (
            df["game_play"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        )
        df = df.sort_values(by=["group_id", "step"]).reset_index(drop=True)

        # 2. Pre-computation: Jerk (Derivative of Acceleration)
        # We assume constant time step of 0.1s, but for ML features, raw diff is sufficient
        print("Computing derivatives (Jerk)...")
        # Mask for valid shifts (same group)
        valid_shift = df["group_id"] == df["group_id"].shift(1)

        # P1 Jerk
        df["jerk_p1"] = df["acceleration_p1"].diff()
        df.loc[~valid_shift, "jerk_p1"] = 0.0

        # P2 Jerk (Handle NaNs for Ground)
        df["jerk_p2"] = df["acceleration_p2"].diff()
        df.loc[~valid_shift, "jerk_p2"] = 0.0
        df["jerk_p2"] = df["jerk_p2"].fillna(0.0)

        # 3. Basis Projection
        print("Computing Decoupled Interaction-Basis Projections...")
        df = self._compute_basis_projections(df)

        # 4. Lag Generation
        print(f"Generating temporal lags (Window Size: {Config.WINDOW_SIZE})...")
        df = self._generate_lags(df)

        # 5. Relaxed Quadratic Gating
        # Only apply gating if we are in training/validation mode (i.e., 'contact' column exists)
        # For test set, we generally keep all rows provided in sample_submission,
        # but the prompt implies we predict for "every allowable contact_id".
        # However, to save inference time, we can predict 0 for gated rows.
        # Here, we will filter the dataset. The caller must handle alignment with submission file if needed.
        # NOTE: For this competition task, we usually predict for all rows in submission.
        # But the architecture describes "Gating" as a stage.
        # We will apply gating to filter the dataframe. The model will only train/predict on survivors.
        # For submission, we merge predictions back.
        print("Applying Relaxed Quadratic Gating...")
        df = self._apply_quadratic_gating(df)

        # 6. Cleanup
        print("Cleaning up features...")
        cols_to_drop = [
            "group_id",
            "x_position_p1",
            "y_position_p1",
            "x_position_p2",
            "y_position_p2",
            "direction_p1",
            "direction_p2",
            "orientation_p1",
            "orientation_p2",
            "sa_p1",
            "sa_p2",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "datetime",
        ]
        # Remove intermediate projection columns (we use lags)
        cols_to_drop += [c for c in self.lag_features if c != "distance"]

        df = df.drop(
            columns=[c for c in cols_to_drop if c in df.columns], errors="ignore"
        )

        df = reduce_mem_usage(df)
        gc.collect()

        print(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def _compute_basis_projections(self, df):
        """
        Computes basis vectors and projects entity kinematics onto them.
        """
        # --- Define Basis Vectors (ux, uy) ---

        # Initialize with zeros
        ux = np.zeros(len(df), dtype=np.float32)
        uy = np.zeros(len(df), dtype=np.float32)

        # Mask for Ground vs Player-Player
        is_ground = df["nfl_player_id_2"] == "G"
        is_pp = ~is_ground

        # Case A: Player-Player (Collision Axis: P2 -> P1)
        # Vector R = P1 - P2
        rx = df.loc[is_pp, "x_position_p1"] - df.loc[is_pp, "x_position_p2"]
        ry = df.loc[is_pp, "y_position_p1"] - df.loc[is_pp, "y_position_p2"]
        dist = np.sqrt(rx**2 + ry**2)

        # Avoid division by zero
        dist = dist.replace(0, 1e-6)

        ux[is_pp] = rx / dist
        uy[is_pp] = ry / dist

        # Case B: Player-Ground (Motion Axis: P1 Velocity)
        # Vector V = V1. If speed is 0, use orientation.
        # Convert direction (degrees) to radians. 0 deg is Y axis (usually), 90 is X.
        # Standard NFL tracking: 0 is Y-axis (North), increasing clockwise.
        # But we have x, y components implied. Let's use direction column if available or compute from dx/dy.
        # The dataset provides 'direction'.
        # Direction is 0..360. 0 is Y (North), 90 is X (East).
        # dx = speed * sin(dir), dy = speed * cos(dir)

        dir_rad = np.radians(df.loc[is_ground, "direction_p1"].fillna(0))
        vx = np.sin(dir_rad)  # Unit vector components
        vy = np.cos(dir_rad)

        ux[is_ground] = vx
        uy[is_ground] = vy

        # Transverse Basis (Rotate 90 deg: -y, x)
        vx_basis = -uy
        vy_basis = ux

        # --- Helper for Projection ---
        def project_vector(mag, angle_deg, basis_x, basis_y, perp_x, perp_y):
            # Convert magnitude/angle to cartesian
            rad = np.radians(angle_deg.fillna(0))
            vec_x = mag * np.sin(rad)
            vec_y = mag * np.cos(rad)

            # Dot products
            proj_u = vec_x * basis_x + vec_y * basis_y
            proj_v = vec_x * perp_x + vec_y * perp_y
            return proj_u, proj_v

        # --- Project Player 1 ---
        # Velocity
        df["proj_u_p1"], df["proj_v_p1"] = project_vector(
            df["speed_p1"], df["direction_p1"], ux, uy, vx_basis, vy_basis
        )

        # Acceleration (Using 'acceleration' magnitude and assuming it aligns with motion or we need angle?
        # Tracking data usually gives 'acceleration' magnitude. It doesn't give acceleration direction explicitly
        # other than 'sa' (signed acceleration).
        # However, we can approximate acceleration direction is along the motion or use 'direction' if we assume tangential.
        # A better approx for 'acc_u' is simply the change in speed or signed acceleration if available.
        # Let's use 'sa' (Signed Acceleration) as the longitudinal component if available, else derive.
        # The prompt mentions "acceleration: magnitude".
        # Let's project the acceleration vector assuming it aligns with 'direction' (tangential)
        # OR use the scalar projection directly if we treat 'sa' as the component along motion.
        # But we need component along *Basis*.
        # Simplification: Treat acceleration vector as aligned with velocity direction for projection purposes,
        # or better, use the scalar 'acceleration' as magnitude and 'direction' as angle.
        df["acc_u_p1"], _ = project_vector(
            df["acceleration_p1"], df["direction_p1"], ux, uy, vx_basis, vy_basis
        )

        # Jerk
        df["jerk_u_p1"], _ = project_vector(
            df["jerk_p1"], df["direction_p1"], ux, uy, vx_basis, vy_basis
        )

        # Orientation relative to basis (Dot product of orientation unit vector and basis)
        o_rad = np.radians(df["orientation_p1"].fillna(0))
        ox, oy = np.sin(o_rad), np.cos(o_rad)
        df["orientation_u_p1"] = ox * ux + oy * uy

        # --- Project Player 2 ---
        # For Ground, P2 features are 0 or irrelevant.
        df["proj_u_p2"] = 0.0
        df["proj_v_p2"] = 0.0
        df["acc_u_p2"] = 0.0
        df["jerk_u_p2"] = 0.0
        df["orientation_u_p2"] = 0.0

        # Only calculate for P-P
        if is_pp.sum() > 0:
            p2_mask = is_pp

            # Velocity
            u_p2, v_p2 = project_vector(
                df.loc[p2_mask, "speed_p2"],
                df.loc[p2_mask, "direction_p2"],
                ux[p2_mask],
                uy[p2_mask],
                vx_basis[p2_mask],
                vy_basis[p2_mask],
            )
            df.loc[p2_mask, "proj_u_p2"] = u_p2
            df.loc[p2_mask, "proj_v_p2"] = v_p2

            # Acc
            a_u_p2, _ = project_vector(
                df.loc[p2_mask, "acceleration_p2"],
                df.loc[p2_mask, "direction_p2"],
                ux[p2_mask],
                uy[p2_mask],
                vx_basis[p2_mask],
                vy_basis[p2_mask],
            )
            df.loc[p2_mask, "acc_u_p2"] = a_u_p2

            # Jerk
            j_u_p2, _ = project_vector(
                df.loc[p2_mask, "jerk_p2"],
                df.loc[p2_mask, "direction_p2"],
                ux[p2_mask],
                uy[p2_mask],
                vx_basis[p2_mask],
                vy_basis[p2_mask],
            )
            df.loc[p2_mask, "jerk_u_p2"] = j_u_p2

            # Orientation
            o2_rad = np.radians(df.loc[p2_mask, "orientation_p2"].fillna(0))
            o2x, o2y = np.sin(o2_rad), np.cos(o2_rad)
            df.loc[p2_mask, "orientation_u_p2"] = o2x * ux[p2_mask] + o2y * uy[p2_mask]

        return df

    def _generate_lags(self, df):
        """
        Generates explicit time-domain lags for selected features.
        """
        # We need lags from -WINDOW to +WINDOW
        window = Config.WINDOW_SIZE
        lags = range(-window, window + 1)

        # Identify group boundaries
        # df is sorted by group_id, step
        group_ids = df["group_id"].values

        # We will iterate through features and lags.
        # To speed up, we can use shift but must mask out boundaries.

        total_rows = len(df)

        for feature in self.lag_features:
            if feature not in df.columns:
                continue

            values = df[feature].values

            for lag in lags:
                if lag == 0:
                    col_name = feature
                    # Feature already exists, no op (or rename if we wanted consistent naming)
                    continue

                col_name = f"{feature}_lag_{lag}"

                # Shift
                shifted_values = np.roll(values, lag)

                # Mask boundaries
                # If lag > 0 (looking back), we invalidate the first 'lag' elements of a group
                # If lag < 0 (looking forward), we invalidate the last 'abs(lag)' elements

                # Check group continuity
                shifted_groups = np.roll(group_ids, lag)

                # Valid mask: group matches
                valid_mask = group_ids == shifted_groups

                # Handle wrap-around at array ends
                if lag > 0:
                    valid_mask[:lag] = False
                elif lag < 0:
                    valid_mask[lag:] = False

                # Apply mask
                # Use 0 as fill value for kinematics, but for distance maybe something else?
                # For distance, filling with a large number or the current value is better.
                # Let's use 0 for simplicity as tree models handle it,
                # or forward/backward fill logic.
                # Here we strictly zero-out invalid lags to indicate "no data".

                result = np.zeros(total_rows, dtype=np.float32)
                result[valid_mask] = shifted_values[valid_mask]

                # Special handling for distance: if invalid, maybe set to sentinel or large value?
                # If we use 0, it looks like contact.
                # Let's use the current value (padding) or a large value.
                # Padding with current value is safer for distance.
                if feature == "distance":
                    result[~valid_mask] = values[~valid_mask]  # Pad with current
                else:
                    result[~valid_mask] = 0.0

                df[col_name] = result

        return df

    def _apply_quadratic_gating(self, df):
        """
        Filters the dataset based on a quadratic approximation of the minimum distance
        within the window.
        """
        # Constants
        WINDOW_SECONDS = 1.0  # +/- 1.0s window roughly matches +/- 10 steps
        MAX_DIST = Config.GATING_DISTANCE

        # Identify Ground interactions (Always keep)
        is_ground = df["nfl_player_id_2"] == "G"

        # Identify Player-Player interactions
        is_pp = ~is_ground

        # Calculate kinematics for quadratic approx
        # Relative Position Vector R
        rx = df["x_position_p1"] - df["x_position_p2"]
        ry = df["y_position_p1"] - df["y_position_p2"]
        r_sq = rx**2 + ry**2

        # Relative Velocity Vector V
        # Convert speed/direction to vx, vy
        # P1
        rad1 = np.radians(df["direction_p1"].fillna(0))
        vx1 = df["speed_p1"] * np.sin(rad1)
        vy1 = df["speed_p1"] * np.cos(rad1)

        # P2
        rad2 = np.radians(df["direction_p2"].fillna(0))
        vx2 = df["speed_p2"] * np.sin(rad2)
        vy2 = df["speed_p2"] * np.cos(rad2)

        dvx = vx1 - vx2
        dvy = vy1 - vy2
        v_sq = dvx**2 + dvy**2

        # Dot product R . V
        r_dot_v = rx * dvx + ry * dvy

        # Time of minimum distance: t* = -(R.V) / V^2
        # Avoid div by zero
        v_sq_safe = v_sq.replace(0, 1e-6)
        t_star = -r_dot_v / v_sq_safe

        # Clamp t_star to window [-1, 1] (seconds)
        # Assuming steps are 0.1s, window size 10 is 1.0s
        t_clamped = t_star.clip(-WINDOW_SECONDS, WINDOW_SECONDS)

        # Min Distance Squared at t_clamped
        # d^2(t) = r^2 + 2(r.v)t + v^2 t^2
        min_dist_sq = r_sq + 2 * r_dot_v * t_clamped + v_sq * (t_clamped**2)

        # Handle numerical errors (negative squared distance)
        min_dist_sq = min_dist_sq.clip(0, None)
        min_dist = np.sqrt(min_dist_sq)

        # Gating Condition
        keep_mask = (min_dist < MAX_DIST) | is_ground

        # Filter
        original_len = len(df)
        df_filtered = df[keep_mask].reset_index(drop=True)
        new_len = len(df_filtered)

        print(
            f"Gating Complete. Rows: {original_len} -> {new_len} ({(new_len/original_len)*100:.2f}%)"
        )

        return df_filtered
