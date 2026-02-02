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
        Computes instantaneous, low-cost features for the Scout model.

        Args:
            df (pd.DataFrame): The base merged table.
            load_cached_data (bool): Whether to use caching.

        Returns:
            pd.DataFrame: DataFrame containing TIER_1_FEATURES.
        """
        # Generate a unique cache key based on dataframe shape and head (proxy for content)
        cache_params = {
            "rows": len(df),
            "cols": sorted(list(df.columns)),
            "first_timestamp": (
                str(df["datetime"].iloc[0]) if "datetime" in df.columns else "N/A"
            ),
        }
        cache_path = self.cache_manager.get_cache_path("features_tier1", cache_params)

        if load_cached_data:
            cached_df = self.cache_manager.load(cache_path)
            if cached_df is not None:
                self.logger.info(f"Loaded Tier 1 features from cache: {cache_path}")
                return cached_df

        self.logger.info("Computing Tier 1 features...")

        # Create a copy to avoid modifying the input
        out_df = pd.DataFrame(index=df.index)

        # 1. Distance
        # Fill NaNs with 0 for calculation (Ground contacts have NaN p2 coords)
        dx = df["x_position_p1"] - df["x_position_p2"].fillna(df["x_position_p1"])
        dy = df["y_position_p1"] - df["y_position_p2"].fillna(df["y_position_p1"])
        out_df["distance"] = np.sqrt(dx**2 + dy**2)

        # 2. Speed & Acceleration
        out_df["speed_p1"] = df["speed_p1"].fillna(0)
        out_df["speed_p2"] = df["speed_p2"].fillna(0)
        out_df["acceleration_p1"] = df["acceleration_p1"].fillna(0)
        out_df["acceleration_p2"] = df["acceleration_p2"].fillna(0)

        out_df["speed_diff"] = np.abs(out_df["speed_p1"] - out_df["speed_p2"])
        out_df["acc_diff"] = np.abs(
            out_df["acceleration_p1"] - out_df["acceleration_p2"]
        )

        # 3. Direction & Orientation
        d1 = df["direction_p1"].fillna(0)
        d2 = df["direction_p2"].fillna(0)
        out_df["direction_diff"] = np.abs(d1 - d2)
        # Handle circularity (360 degrees)
        out_df["direction_diff"] = np.minimum(
            out_df["direction_diff"], 360 - out_df["direction_diff"]
        )

        o1 = df["orientation_p1"].fillna(0)
        o2 = df["orientation_p2"].fillna(0)
        out_df["orientation_diff"] = np.abs(o1 - o2)
        out_df["orientation_diff"] = np.minimum(
            out_df["orientation_diff"], 360 - out_df["orientation_diff"]
        )

        # 4. Is Ground
        # Check if nfl_player_id_2 is 'G'
        out_df["is_ground"] = (df["nfl_player_id_2"] == "G").astype(int)

        # 5. Metadata
        out_df["step"] = df["step"]

        # Ensure we only return the requested columns
        for col in TIER_1_FEATURES:
            if col not in out_df.columns:
                out_df[col] = 0.0

        out_df = out_df[TIER_1_FEATURES]

        # Save to cache
        self.logger.info(f"Saving Tier 1 features to cache: {cache_path}")
        self.cache_manager.save(out_df, cache_path)

        return out_df

    def compute_tier2_features(self, df, target_mask=None, load_cached_data=True):
        """
        Computes contextual, high-cost features for the Expert model.
        Uses a 'compute-all-then-filter' strategy to ensure time-series continuity
        for rolling windows, while leveraging caching to avoid re-computation.

        Args:
            df (pd.DataFrame): The full base merged table.
            target_mask (pd.Series/np.array, optional): Boolean mask of rows to return.
                                                      If None, returns all rows.
            load_cached_data (bool): Whether to use caching.

        Returns:
            pd.DataFrame: DataFrame containing Tier 2 features for the masked rows.
        """
        # Cache key based on full dataframe content and window config
        cache_params = {
            "rows": len(df),
            "window_size": WINDOW_SIZE,
            "version": "v1_full_rolling",
        }
        cache_path = self.cache_manager.get_cache_path(
            "features_tier2_full", cache_params
        )

        features_computed = False
        full_features_df = None

        if load_cached_data:
            full_features_df = self.cache_manager.load(cache_path)
            if full_features_df is not None:
                self.logger.info(
                    f"Loaded full Tier 2 features from cache: {cache_path}"
                )
                features_computed = True

        if not features_computed:
            self.logger.info("Computing full Tier 2 features (this may take time)...")

            # Prepare working dataframe
            # We need to sort by game_play, player pair, and step to ensure correct rolling
            sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]

            # Select only necessary columns to save memory
            req_cols = list(set(sort_cols + TIER_2_WINDOW_BASE_COLS))
            req_cols = [c for c in req_cols if c in df.columns]

            work_df = df[req_cols].copy()
            work_df = work_df.fillna(0)

            # Sort for GroupBy
            work_df = work_df.sort_values(sort_cols)

            # Group by unique contact pair within a play
            grp = work_df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

            feature_cols = {}

            # 1. Physics Derivatives (Jerk)
            if "acceleration_p1" in work_df.columns:
                feature_cols["jerk_p1"] = grp["acceleration_p1"].diff().fillna(0)
            if "acceleration_p2" in work_df.columns:
                feature_cols["jerk_p2"] = grp["acceleration_p2"].diff().fillna(0)

            # Angular Jerk (using direction as proxy if available)
            if "direction_p1" in work_df.columns:
                feature_cols["angular_jerk_p1"] = grp["direction_p1"].diff().fillna(0)
            if "direction_p2" in work_df.columns:
                feature_cols["angular_jerk_p2"] = grp["direction_p2"].diff().fillna(0)

            # 2. Flattened Temporal Windows (Lags)
            # Create lags from -WINDOW_SIZE to +WINDOW_SIZE
            lags = range(-WINDOW_SIZE, WINDOW_SIZE + 1)

            for col in TIER_2_WINDOW_BASE_COLS:
                if col not in work_df.columns:
                    continue

                for lag in lags:
                    if lag == 0:
                        continue

                    col_name = f"{col}_lag_{lag}"
                    # shift(k) shifts data down k periods.
                    # To get t-k (past), we use shift(k).
                    # To get t+k (future), we use shift(-k).
                    feature_cols[col_name] = grp[col].shift(lag).fillna(0)

            # 3. Spatial Density
            # Placeholder: Requires full frame tracking not available in pair-wise table
            if "spatial_density" in TIER_2_EXTRA_FEATURES:
                feature_cols["spatial_density"] = 0.0

            # Construct result dataframe
            res_df = pd.DataFrame(feature_cols, index=work_df.index)

            # Reindex to original df index to undo the sorting effect
            res_df = res_df.reindex(df.index)
            res_df = res_df.fillna(0)
            res_df = res_df.astype(np.float32)

            full_features_df = res_df

            # Save to cache
            self.logger.info(f"Saving full Tier 2 features to cache: {cache_path}")
            self.cache_manager.save(full_features_df, cache_path)

            # Clean up
            del work_df, grp, res_df
            gc.collect()

        # Apply Mask if provided
        if target_mask is not None:
            self.logger.info(f"Filtering Tier 2 features to {sum(target_mask)} rows...")
            return full_features_df.loc[target_mask].copy()

        return full_features_df
