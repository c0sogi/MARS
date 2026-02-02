import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    CACHE_DIR,
    WINDOW_HALF,
    GATING_THRESHOLD,
    LGBM_PARAMS,
    XGB_PARAMS,
    SEED,
)
from library.utils import setup_logger, CacheManager


class FeatureEngineer:
    """
    Implements Kinematically-Aligned Feature Engineering with Time-Domain Lags
    and Relaxed Quadratic Gating.
    """

    def __init__(self, logger=None):
        self.logger = (
            logger
            if logger
            else setup_logger(
                os.path.join(os.getcwd(), "logs", "feature_engineering.log")
            )
        )
        self.cache_manager = CacheManager()
        self.window_size = 2 * WINDOW_HALF + 1

    def create_features(
        self,
        df_meta,
        df_tracking,
        split="train",
        load_cached_data=True,
        save_output=True,
    ):
        """
        Main entry point for feature engineering.

        Args:
            df_meta (pd.DataFrame): Metadata containing contact_id, game_play, step, players.
            df_tracking (pd.DataFrame): Tracking data.
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.
            save_output (bool): Whether to save the result to cache.

        Returns:
            pd.DataFrame: Feature dataframe ready for model training/inference.
        """
        # Determine cache key based on split strategy
        # Train is gated, Val/Test are full
        is_gated = split == "train"
        suffix = "gated" if is_gated else "full"
        cache_key = f"features_{split}_{suffix}.parquet"

        if load_cached_data:
            cached_df = self.cache_manager.load_parquet(cache_key)
            if cached_df is not None:
                self.logger.info(f"Loaded features from cache: {cache_key}")
                return cached_df

        self.logger.info(
            f"Generating features for {split} (Cache miss or force reload)..."
        )

        # 1. Generate Windowed Kinematic Features
        df_features = self._generate_windowed_features(df_meta, df_tracking)

        # 2. Apply Relaxed Quadratic Gating
        # We only gate train set to reduce imbalance. Val/Test set must keep all rows.
        if is_gated:
            df_features = self._apply_quadratic_gating(df_features)

        # 3. Final Cleanup
        # Ensure contact column is present (for train/val) or handled
        if "contact" not in df_features.columns and "contact" in df_meta.columns:
            # Merge target back if it was lost (though _generate keeps index/contact_id usually)
            # In this implementation, we construct df_features based on df_meta's index
            df_features = df_features.merge(
                df_meta[["contact_id", "contact"]], on="contact_id", how="left"
            )

        # Fill NaNs that might result from missing tracking data with 0 or sentinels
        # (Tree models handle NaNs, but explicit filling is safer for some features)
        df_features = df_features.fillna(0.0)

        if save_output:
            self.logger.info(f"Saving features to cache: {cache_key}")
            self.cache_manager.save_parquet(df_features, cache_key)

        return df_features

    def _generate_windowed_features(self, df_meta, df_tracking):
        """
        Iterates through time lags, merges tracking data, and computes kinematic features.
        """
        self.logger.info("Starting windowed feature generation...")

        # Prepare base metadata
        # We need to ensure we don't modify the original
        meta_base = df_meta.copy()

        # Pre-process tracking for faster merges
        # Keep only necessary columns
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ]
        # Filter tracking to only columns we need
        df_track = df_tracking[track_cols].copy()

        # Handle P2 ID conversion once
        meta_base["nfl_player_id_2_numeric"] = pd.to_numeric(
            meta_base["nfl_player_id_2"], errors="coerce"
        )

        # List to hold feature dataframes for each lag
        lag_features_list = []

        # We also need to keep track of contact_id to merge everything back together
        # We will index everything by the original index of df_meta

        # Iterate over lags
        lags = range(-WINDOW_HALF, WINDOW_HALF + 1)

        for lag in lags:
            # Create a temp merge key with shifted step
            meta_lag = meta_base[
                [
                    "game_play",
                    "step",
                    "nfl_player_id_1",
                    "nfl_player_id_2_numeric",
                    "contact_id",
                ]
            ].copy()
            meta_lag["step_shifted"] = meta_lag["step"] + lag

            # Merge P1
            merged = meta_lag.merge(
                df_track,
                left_on=["game_play", "step_shifted", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p1"),
            ).rename(
                columns={
                    c: f"{c}_p1"
                    for c in [
                        "x_position",
                        "y_position",
                        "speed",
                        "direction",
                        "orientation",
                        "acceleration",
                        "sa",
                    ]
                }
            )

            # Merge P2
            merged = merged.merge(
                df_track,
                left_on=["game_play", "step_shifted", "nfl_player_id_2_numeric"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p2"),
            ).rename(
                columns={
                    c: f"{c}_p2"
                    for c in [
                        "x_position",
                        "y_position",
                        "speed",
                        "direction",
                        "orientation",
                        "acceleration",
                        "sa",
                    ]
                }
            )

            # Compute Kinematics for this lag
            kinematics = self._compute_instantaneous_kinematics(
                merged, meta_base["nfl_player_id_2"]
            )

            # Rename columns to include lag suffix
            kinematics.columns = [f"{col}_lag{lag}" for col in kinematics.columns]

            # Add contact_id for joining
            kinematics["contact_id"] = merged["contact_id"]

            lag_features_list.append(kinematics)

            # periodic gc
            if lag % 5 == 0:
                gc.collect()

        self.logger.info("Merging all temporal lags...")

        # Start with the contact_ids
        final_df = df_meta[["contact_id"]].copy()

        for lag_df in lag_features_list:
            final_df = final_df.merge(lag_df, on="contact_id", how="left")

        return final_df

    def _compute_instantaneous_kinematics(self, df, p2_id_original_series):
        """
        Computes vector-based kinematic features for a single timestep.
        Handles Basis Alignment and Ground Sentinels.
        """
        # Fill missing tracking data with 0 to allow vector math (will be handled by sentinels later if needed)
        # Note: We must be careful not to introduce fake data.
        # However, for vector ops, NaNs propagate, which is fine.

        # 1. Identify Ground Interactions
        # We use the original p2_id series (aligned by index) to check for 'G'
        # Since 'df' is derived from a merge that preserves order/index of 'meta_lag',
        # and 'meta_lag' is derived from 'meta_base', the indices should align if we reset them properly.
        # However, merge operations might shuffle.
        # Safer: Check 'nfl_player_id_2_numeric' is NaN in df.
        # But NaN could also mean missing tracking for a real player.
        # Let's rely on the fact that for 'G', p2 columns are NaN.

        # Extract coordinates
        x1, y1 = df["x_position_p1"], df["y_position_p1"]
        x2, y2 = df["x_position_p2"], df["y_position_p2"]

        vx1 = df["speed_p1"] * np.sin(np.radians(df["direction_p1"]))
        vy1 = df["speed_p1"] * np.cos(np.radians(df["direction_p1"]))

        vx2 = df["speed_p2"] * np.sin(np.radians(df["direction_p2"]))
        vy2 = df["speed_p2"] * np.cos(np.radians(df["direction_p2"]))

        ax1 = df["acceleration_p1"] * np.sin(
            np.radians(df["direction_p1"])
        )  # Approx direction of accel along motion
        ay1 = df["acceleration_p1"] * np.cos(np.radians(df["direction_p1"]))

        ax2 = df["acceleration_p2"] * np.sin(np.radians(df["direction_p2"]))
        ay2 = df["acceleration_p2"] * np.cos(np.radians(df["direction_p2"]))

        # --- Handle Ground (P2 is NaN) ---
        # If P2 is missing (Ground or Missing Data), assume stationary at (0,0) relative or handle via sentinel
        # For Ground, we want Distance = -1.
        # For vector math, let's set P2 pos/vel/acc to 0 temporarily, then override.

        is_ground_or_missing = x2.isna()

        x2 = x2.fillna(0)
        y2 = y2.fillna(0)
        vx2 = vx2.fillna(0)
        vy2 = vy2.fillna(0)
        ax2 = ax2.fillna(0)
        ay2 = ay2.fillna(0)

        # --- Relative Vectors ---
        rx = x1 - x2
        ry = y1 - y2
        dist = np.sqrt(rx**2 + ry**2)

        vrx = vx1 - vx2
        vry = vy1 - vy2
        v_rel_mag = np.sqrt(vrx**2 + vry**2)

        arx = ax1 - ax2
        ary = ay1 - ay2

        # --- Basis Alignment (Relative Velocity Basis) ---
        # u = v_rel / |v_rel|
        # Avoid divide by zero
        epsilon = 1e-6
        u_x = vrx / (v_rel_mag + epsilon)
        u_y = vry / (v_rel_mag + epsilon)

        # Orthogonal basis u_perp (-u_y, u_x)
        up_x = -u_y
        up_y = u_x

        # --- Projections ---
        # r_long: Project r onto u
        r_long = rx * u_x + ry * u_y
        # r_trans: Project r onto u_perp
        r_trans = rx * up_x + ry * up_y

        # a_long: Project a_rel onto u
        a_long = arx * u_x + ary * u_y
        # a_trans: Project a_rel onto u_perp
        a_trans = arx * up_x + ary * up_y

        # --- Primitives ---
        # TTC: r / closing_speed. Closing speed is -projection of v_rel onto r?
        # Simpler: r / v_rel_mag (approx) or r_long / v_rel_mag
        # Let's use r / v_rel_mag
        ttc = dist / (v_rel_mag + epsilon)

        # Angular Jerk Proxy: Change in orientation?
        # We only have instantaneous here. We can use difference in orientation.
        # orientation is 0-360.
        o1 = df["orientation_p1"].fillna(0)
        o2 = df["orientation_p2"].fillna(0)
        # Handle periodicity
        diff = np.abs(o1 - o2)
        orientation_diff = np.minimum(diff, 360 - diff)

        # --- Construct Result DataFrame ---
        features = pd.DataFrame(
            {
                "dist": dist,
                "r_long": r_long,
                "r_trans": r_trans,
                "v_rel": v_rel_mag,
                "a_long": a_long,
                "a_trans": a_trans,
                "ttc": ttc,
                "orientation_diff": orientation_diff,
                "speed_p1": df["speed_p1"],
                "speed_p2": df["speed_p2"],
                "accel_p1": df["acceleration_p1"],
                "accel_p2": df["acceleration_p2"],
                "sa_p1": df["sa_p1"],
                "sa_p2": df["sa_p2"],
            }
        )

        # --- Apply Sentinels for Ground/Missing ---
        # If is_ground_or_missing, set distance related metrics to sentinel
        sentinel = -1.0
        features.loc[is_ground_or_missing, "dist"] = sentinel
        features.loc[is_ground_or_missing, "r_long"] = sentinel
        features.loc[is_ground_or_missing, "r_trans"] = sentinel
        features.loc[is_ground_or_missing, "ttc"] = sentinel

        # For Ground, v_rel is just v_p1 (since p2 is 0). This is physically meaningful.
        # But speed_p2 should be 0 (already filled).

        return features

    def _apply_quadratic_gating(self, df_features):
        """
        Applies Relaxed Quadratic Gating.
        Calculates min(distance) across all lags for each pair.
        Keeps pairs where min_dist < Threshold OR pair is Ground contact.
        """
        self.logger.info(
            f"Applying Relaxed Quadratic Gating (Threshold={GATING_THRESHOLD}y)..."
        )

        # Identify distance columns
        # They are named dist_lag-10, ..., dist_lag10
        dist_cols = [c for c in df_features.columns if c.startswith("dist_lag")]

        if not dist_cols:
            self.logger.warning("No distance columns found for gating. Skipping.")
            return df_features

        # Compute min distance across window
        # Note: Ground contacts have dist = -1.0, so min_dist will be -1.0, which is < 3.0.
        # So Ground contacts are automatically preserved.
        min_dists = df_features[dist_cols].min(axis=1)

        # Filter
        mask = min_dists < GATING_THRESHOLD

        original_len = len(df_features)
        df_gated = df_features[mask].copy()
        new_len = len(df_gated)

        self.logger.info(
            f"Gating reduced data from {original_len} to {new_len} rows ({new_len/original_len:.2%})"
        )

        return df_gated
