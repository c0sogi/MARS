import pandas as pd
import numpy as np
import os
import gc
from library.config import (
    WORKING_DIR,
    TIER_1_FEATURES,
    TIER_2_WINDOW_BASE_COLS,
    TIER_2_EXTRA_FEATURES,
    WINDOW_SIZE,
    SEED,
)
from library.utils import setup_logger, CacheManager


class FeatureFactory:
    """
    Implements the Tiered Feature Engineering strategy.
    Tier 1: Low-cost, instantaneous features for the entire dataset (Scout).
    Tier 2: High-cost, contextual features for mined candidates (Expert).
    """

    def __init__(self, cache_dir=WORKING_DIR):
        self.logger = setup_logger()
        self.cache_manager = CacheManager(cache_dir)

    def compute_tier1_features(self, df, load_cached_data=True):
        """
        Unified Feature Generation.
        Tier 1 now redirects to Tier 2 to ensure Scout model sees full temporal context.
        Cite Lesson 00022.
        """
        # We perform the basic Tier 1 calculations (instantaneous) inside Tier 2 logic
        # or as a pre-step. To keep it simple, we just call compute_tier2_features
        # which now handles everything.
        return self.compute_tier2_features(df, load_cached_data=load_cached_data)

    def compute_tier2_features(self, df, target_mask=None, load_cached_data=True):
        """
        Computes all features (Instantaneous + Contextual + Temporal).
        Optimized for large-scale processing using vectorized shifts.
        """
        # Cache key
        cache_params = {
            "rows": len(df),
            "window_size": WINDOW_SIZE,
            "version": "v2_unified_optimized",
        }
        cache_path = self.cache_manager.get_cache_path("features_unified", cache_params)

        if load_cached_data:
            full_features_df = self.cache_manager.load(cache_path)
            if full_features_df is not None:
                self.logger.info(f"Loaded unified features from cache: {cache_path}")
                if target_mask is not None:
                    return full_features_df.loc[target_mask].copy()
                return full_features_df

        self.logger.info("Computing unified features (Vectorized)...")

        # 1. Base Feature Calculation (Instantaneous)
        # We calculate these on the fly to ensure we have them for windowing
        work_df = df.copy()

        # Distance
        dx = work_df["x_position_p1"] - work_df["x_position_p2"].fillna(
            work_df["x_position_p1"]
        )
        dy = work_df["y_position_p1"] - work_df["y_position_p2"].fillna(
            work_df["y_position_p1"]
        )
        work_df["distance"] = np.sqrt(dx**2 + dy**2)

        # Speed/Acc Diffs
        work_df["speed_p1"] = work_df["speed_p1"].fillna(0)
        work_df["speed_p2"] = work_df["speed_p2"].fillna(0)
        work_df["acceleration_p1"] = work_df["acceleration_p1"].fillna(0)
        work_df["acceleration_p2"] = work_df["acceleration_p2"].fillna(0)
        work_df["speed_diff"] = np.abs(work_df["speed_p1"] - work_df["speed_p2"])
        work_df["acc_diff"] = np.abs(
            work_df["acceleration_p1"] - work_df["acceleration_p2"]
        )

        # Direction/Orientation
        d1 = work_df["direction_p1"].fillna(0)
        d2 = work_df["direction_p2"].fillna(0)
        work_df["direction_diff"] = np.abs(d1 - d2)
        work_df["direction_diff"] = np.minimum(
            work_df["direction_diff"], 360 - work_df["direction_diff"]
        )

        o1 = work_df["orientation_p1"].fillna(0)
        o2 = work_df["orientation_p2"].fillna(0)
        work_df["orientation_diff"] = np.abs(o1 - o2)
        work_df["orientation_diff"] = np.minimum(
            work_df["orientation_diff"], 360 - work_df["orientation_diff"]
        )

        work_df["is_ground"] = (work_df["nfl_player_id_2"] == "G").astype(int)

        # Ensure spatial density is present (calculated in data_loader)
        if "spatial_density_p1" not in work_df.columns:
            work_df["spatial_density_p1"] = 0.0
        if "spatial_density_p2" not in work_df.columns:
            work_df["spatial_density_p2"] = 0.0

        # 2. Vectorized Rolling Windows
        # Sort to align time series
        sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        work_df = work_df.sort_values(sort_cols)

        # Create masks for group boundaries
        # We check if the group identifier changes from the previous row
        # Group ID: game_play + p1 + p2
        # We can use pandas diff on the hash or just compare columns.
        # Comparing string columns is slow, so we rely on the sort order.

        # We will use shift() on the whole dataframe.
        # Then we mask out rows where the group changed.

        # Define columns to shift
        cols_to_shift = [c for c in TIER_2_WINDOW_BASE_COLS if c in work_df.columns]

        feature_cols = {}

        # Add Base Features to output
        base_features = [
            "distance",
            "speed_p1",
            "speed_p2",
            "speed_diff",
            "acceleration_p1",
            "acceleration_p2",
            "acc_diff",
            "direction_diff",
            "orientation_diff",
            "is_ground",
            "step",
            "spatial_density_p1",
            "spatial_density_p2",
        ]
        for c in base_features:
            if c in work_df.columns:
                feature_cols[c] = work_df[c]

        # Physics Derivatives (Diffs)
        # We use shift(1) and check boundary
        # Group identifiers
        g_play = work_df["game_play"]
        p1 = work_df["nfl_player_id_1"]
        p2 = work_df["nfl_player_id_2"].astype(str)  # Handle mixed types if any

        # Shifted identifiers
        g_play_s1 = g_play.shift(1)
        p1_s1 = p1.shift(1)
        p2_s1 = p2.shift(1)

        # Valid mask: True if current row belongs to same group as previous row
        valid_mask = (g_play == g_play_s1) & (p1 == p1_s1) & (p2 == p2_s1)

        # Jerk
        if "acceleration_p1" in work_df.columns:
            diff = work_df["acceleration_p1"] - work_df["acceleration_p1"].shift(1)
            feature_cols["jerk_p1"] = diff.where(valid_mask, 0).fillna(0)

        if "acceleration_p2" in work_df.columns:
            diff = work_df["acceleration_p2"] - work_df["acceleration_p2"].shift(1)
            feature_cols["jerk_p2"] = diff.where(valid_mask, 0).fillna(0)

        # Angular Jerk
        if "direction_p1" in work_df.columns:
            diff = work_df["direction_p1"] - work_df["direction_p1"].shift(1)
            feature_cols["angular_jerk_p1"] = diff.where(valid_mask, 0).fillna(0)

        if "direction_p2" in work_df.columns:
            diff = work_df["direction_p2"] - work_df["direction_p2"].shift(1)
            feature_cols["angular_jerk_p2"] = diff.where(valid_mask, 0).fillna(0)

        # Lags
        lags = range(-WINDOW_SIZE, WINDOW_SIZE + 1)

        for lag in lags:
            if lag == 0:
                continue

            # For lag k (positive = past in shift notation? No, shift(k) gets previous. shift(-k) gets future)
            # We want t-k. shift(k).
            # We want t+k. shift(-k).

            shifted_group_ids_match = (
                (g_play == g_play.shift(lag))
                & (p1 == p1.shift(lag))
                & (p2 == p2.shift(lag))
            )

            for col in cols_to_shift:
                col_name = f"{col}_lag_{lag}"
                shifted_vals = work_df[col].shift(lag)
                feature_cols[col_name] = shifted_vals.where(
                    shifted_group_ids_match, 0
                ).fillna(0)

        # Construct Result
        res_df = pd.DataFrame(feature_cols, index=work_df.index)

        # Reindex to original order
        res_df = res_df.reindex(df.index)
        res_df = res_df.fillna(0)
        res_df = res_df.astype(np.float32)

        self.logger.info(f"Saving unified features to cache: {cache_path}")
        self.cache_manager.save(res_df, cache_path)

        if target_mask is not None:
            return res_df.loc[target_mask].copy()

        return res_df
