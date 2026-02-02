import os
import gc
import numpy as np
import pandas as pd
import logging
from library.config import Config
from library.utils import setup_logger, generate_content_hash
from library.data_loader import load_tracking, reduce_mem_usage

# Initialize logger
logger = setup_logger("feature_engine")


class FeatureProcessor:
    """
    Handles feature engineering for NFL Contact Detection.
    Generates kinematic, contextual, and temporal features from tracking data.
    """

    def __init__(self):
        self.window_size = Config.WINDOW_SIZE
        self.neighbor_radius = Config.NEIGHBOR_RADIUS
        self.use_jerk = Config.USE_JERK
        self.use_cluster = Config.USE_CLUSTER_KINEMATICS

        # Base columns to extract from tracking data
        self.base_tracking_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "sa",
        ]
        # This list will be extended with derived features
        self.feature_cols = self.base_tracking_cols.copy()

    def _get_cache_path(self, split, metadata_len):
        """Generates a unique cache filename based on config and data."""
        config_dict = {
            "window": self.window_size,
            "radius": self.neighbor_radius,
            "jerk": self.use_jerk,
            "cluster": self.use_cluster,
            "split": split,
            "meta_len": metadata_len,
        }
        hash_str = generate_content_hash(config_dict)
        filename = f"features_{split}_{metadata_len}_{hash_str}.parquet"
        return os.path.join(Config.WORKING_DIR, filename)

    def compute_physics_derivatives(self, df):
        """
        Computes Jerk (d_acc/dt) and Angular Jerk (d_ang_vel/dt).
        Assumes df is sorted by game_play, nfl_player_id, step.
        """
        if not self.use_jerk:
            return df

        logger.info("Computing physics derivatives (Jerk)...")

        # Identify boundaries where play or player changes
        # We use shift to compare current row with previous row
        # Data must be sorted by [game_play, nfl_player_id, step]

        prev_game = df["game_play"].shift(1)
        prev_player = df["nfl_player_id"].shift(1)

        # Mask is True where the previous row belongs to the same player in the same play
        # We can only compute derivatives where this mask is True
        mask = (df["game_play"] == prev_game) & (df["nfl_player_id"] == prev_player)

        # 1. Linear Jerk = diff(acceleration) / dt (dt=0.1s, constant)
        # We use simple diff as a proxy for derivative
        df["jerk"] = df["acceleration"].diff().fillna(0)
        df.loc[~mask, "jerk"] = 0

        # 2. Angular Jerk
        # First, compute Angular Velocity = diff(orientation)
        # Orientation is 0-360 degrees. We need to handle the wrap-around.
        o_curr = df["orientation"]
        o_prev = df["orientation"].shift(1).fillna(0)

        # Calculate shortest arc difference
        # (x - y + 180) % 360 - 180 maps difference to [-180, 180]
        ang_diff = (o_curr - o_prev + 180) % 360 - 180

        # Assign angular velocity
        df["ang_vel"] = ang_diff
        df.loc[~mask, "ang_vel"] = 0

        # Angular Jerk = diff(ang_vel)
        df["ang_jerk"] = df["ang_vel"].diff().fillna(0)
        df.loc[~mask, "ang_jerk"] = 0

        # Drop intermediate column
        df.drop(columns=["ang_vel"], inplace=True)

        # Register new features
        if "jerk" not in self.feature_cols:
            self.feature_cols.extend(["jerk", "ang_jerk"])

        return df

    def compute_context_features(self, df):
        """
        Computes spatial density (neighbor count) and cluster kinematics.
        Uses vectorized NumPy broadcasting for performance.
        """
        if not self.use_cluster:
            return df

        logger.info("Computing context features (Density & Cluster Kinematics)...")

        # Ensure sorting for grouping
        df.sort_values(["game_play", "step"], inplace=True)

        # Prepare result arrays
        n_samples = len(df)
        neighbor_counts = np.zeros(n_samples, dtype=np.int32)
        cluster_speeds = np.zeros(n_samples, dtype=np.float32)
        cluster_accels = np.zeros(n_samples, dtype=np.float32)

        # Convert columns to numpy for fast access
        gp_arr = df["game_play"].values
        step_arr = df["step"].values
        x_arr = df["x_position"].values
        y_arr = df["y_position"].values
        s_arr = df["speed"].values
        a_arr = df["acceleration"].values

        # Identify group boundaries (where game_play or step changes)
        # Factorize game_play to integers for faster comparison
        gp_codes, _ = pd.factorize(gp_arr)

        # Create a combined ID: gp_code * multiplier + step
        # Assuming step < 100000
        group_ids = gp_codes.astype(np.int64) * 100000 + step_arr

        # Find indices where group_ids change
        # np.diff != 0 gives indices before the change
        change_indices = np.flatnonzero(np.diff(group_ids) != 0) + 1
        split_indices = np.concatenate(([0], change_indices, [n_samples]))

        # Iterate over each frame (game_play + step)
        # This loop runs ~50k times for the full train set.
        # Inside the loop is efficient numpy broadcasting on small arrays (~22 players).
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]

            # Extract data for this frame
            xs = x_arr[start:end]
            ys = y_arr[start:end]
            ss = s_arr[start:end]
            aas = a_arr[start:end]

            # Coordinate matrix: (N, 2)
            coords = np.column_stack((xs, ys))

            # Compute Pairwise Euclidean Distance using broadcasting
            # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists_sq = (diff**2).sum(axis=2)
            dists = np.sqrt(dists_sq)

            # Mask: Neighbors are within radius, excluding self (dist > 0)
            mask = (dists <= self.neighbor_radius) & (dists > 0)

            # Calculate metrics
            counts = mask.sum(axis=1)

            # Cluster Average Speed/Accel
            # dot product: sum(mask[i,j] * val[j])
            sum_speeds = mask @ ss
            sum_accels = mask @ aas

            # Avoid division by zero
            safe_counts = counts.copy()
            safe_counts[safe_counts == 0] = 1

            avg_speeds = sum_speeds / safe_counts
            avg_accels = sum_accels / safe_counts

            # Zero out metrics where count is 0
            avg_speeds[counts == 0] = 0
            avg_accels[counts == 0] = 0

            # Store results
            neighbor_counts[start:end] = counts
            cluster_speeds[start:end] = avg_speeds
            cluster_accels[start:end] = avg_accels

        # Assign back to DataFrame
        df["neighbor_count"] = neighbor_counts
        df["cluster_speed"] = cluster_speeds
        df["cluster_accel"] = cluster_accels

        if "neighbor_count" not in self.feature_cols:
            self.feature_cols.extend(
                ["neighbor_count", "cluster_speed", "cluster_accel"]
            )

        return df

    def generate_features(self, metadata_df, split="train", load_cached_data=True):
        """
        Main pipeline to generate features.
        1. Checks cache.
        2. Loads and preprocesses tracking data.
        3. Merges tracking data onto metadata for a temporal window (t-W to t+W).
        4. Saves to cache.
        """
        # 1. Check Cache
        cache_path = self._get_cache_path(split, len(metadata_df))
        if load_cached_data and os.path.exists(cache_path):
            logger.info(f"Loading cached features from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache ({e}). Regenerating...")

        # 2. Load Tracking Data
        logger.info(f"Loading tracking data for split: {split}")
        # Map 'val' logical split to 'train' physical source
        tracking_source = "train" if split == "val" else split
        tracking_df = load_tracking(tracking_source, load_cached_data=load_cached_data)

        # 3. Preprocess Tracking (Derivatives & Context)
        # Sort is critical for derivatives
        tracking_df.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

        tracking_df = self.compute_physics_derivatives(tracking_df)
        tracking_df = self.compute_context_features(tracking_df)

        # Optimize memory before the merge explosion
        tracking_df = reduce_mem_usage(tracking_df)

        # 4. Merge Loop (Temporal Window)
        logger.info(f"Generating features with window +/- {self.window_size} steps...")

        # Start with metadata
        result_df = metadata_df.copy()

        # Create a temporary integer column for Player 2 to handle 'G' (Ground)
        # 'G' will become -1 (or NaN converted to int), allowing us to merge on int types
        result_df["p2_int"] = (
            pd.to_numeric(result_df["nfl_player_id_2"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

        # Define offsets
        offsets = range(-self.window_size, self.window_size + 1)

        # We need a subset of tracking columns for merging
        cols_to_merge = ["game_play", "step", "nfl_player_id"] + self.feature_cols

        for offset in offsets:
            # We want tracking features at time T_event + offset.
            # The tracking dataframe has rows at time T_track.
            # We want to join Metadata(T_event) with Tracking(T_track) where T_track = T_event + offset.
            # Equivalently: T_track - offset = T_event.
            # So we create a temporary join key in tracking: 'step_join' = step - offset.

            # Create a lightweight subset for this offset
            track_subset = tracking_df[cols_to_merge].copy()
            track_subset["step"] = track_subset["step"] - offset

            # Define suffix
            suffix = f"_t{offset}"

            # --- Merge Player 1 ---
            # Rename columns for P1
            rename_p1 = {c: f"{c}_p1{suffix}" for c in self.feature_cols}
            track_p1 = track_subset.rename(columns=rename_p1)

            result_df = pd.merge(
                result_df,
                track_p1,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            result_df.drop(columns=["nfl_player_id"], inplace=True)

            # --- Merge Player 2 ---
            # Rename columns for P2
            rename_p2 = {c: f"{c}_p2{suffix}" for c in self.feature_cols}
            track_p2 = track_subset.rename(columns=rename_p2)

            # Merge using the integer ID. Ground ('G' -> -1) will fail to match, resulting in NaNs (correct).
            result_df = pd.merge(
                result_df,
                track_p2,
                left_on=["game_play", "step", "p2_int"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            result_df.drop(columns=["nfl_player_id"], inplace=True)

            # --- Compute Relative Features for this offset ---
            # Distance
            x1 = result_df[f"x_position_p1{suffix}"]
            y1 = result_df[f"y_position_p1{suffix}"]
            x2 = result_df[f"x_position_p2{suffix}"]
            y2 = result_df[f"y_position_p2{suffix}"]

            # Calculate Euclidean distance
            # NaNs (Ground) will propagate, which is handled later
            result_df[f"distance{suffix}"] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            # Relative Angles (Cite solution_lesson_node_00012: Higher-Order Derivatives and Context)
            # We capture the relative alignment of players (e.g., head-on vs parallel)
            d1 = result_df[f"direction_p1{suffix}"]
            d2 = result_df[f"direction_p2{suffix}"]
            result_df[f"direction_diff{suffix}"] = (d1 - d2 + 180) % 360 - 180

            o1 = result_df[f"orientation_p1{suffix}"]
            o2 = result_df[f"orientation_p2{suffix}"]
            result_df[f"orientation_diff{suffix}"] = (o1 - o2 + 180) % 360 - 180

            # Clean up
            del track_subset, track_p1, track_p2
            gc.collect()

        # Drop helper column
        result_df.drop(columns=["p2_int"], inplace=True)

        # Fill Missing Values
        # Missing values occur for:
        # 1. Ground interactions (P2 is G) -> P2 features are NaN
        # 2. Missing tracking data -> P1 or P2 features are NaN
        # We fill with 0. Tree-based models can handle this or we can use a specific value.
        # 0 is safe for distance/speed in this context (implies no movement/contact at origin).
        result_df.fillna(0, inplace=True)

        # Final memory reduction
        result_df = reduce_mem_usage(result_df)

        # 5. Save to Cache
        logger.info(f"Saving generated features to {cache_path}...")
        try:
            result_df.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        return result_df
