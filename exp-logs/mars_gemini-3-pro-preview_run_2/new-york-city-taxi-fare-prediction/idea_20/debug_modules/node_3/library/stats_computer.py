import os
import numpy as np
import pandas as pd
from library import config
from library import utils
from library import data_manager


class StatsEngine:
    """
    Computes, caches, and applies Factorized Multi-Moment statistical priors.
    Implements Hierarchical Spatial Context and Conditional Vectorized Subtraction.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.levels = config.GEOHASH_LEVELS
        self.bbox = config.NYC_BBOX

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, level, stat_type):
        return os.path.join(
            self.cache_dir, f"global_stats_L{level}_{stat_type}.parquet"
        )

    def _calculate_derived_moments(self, sum_col, sum_sq_col, count_col):
        """
        Calculates Mean and Standard Deviation from Sum, SumSq, and Count.
        Handles division by zero and small negative variances due to float precision.
        """
        # Calculate Mean
        mean = sum_col / count_col

        # Calculate Variance: E[X^2] - (E[X])^2
        # variance = (sum_sq / count) - (mean^2)
        # Using a slightly more stable form for vectorized ops: (sum_sq - count * mean^2) / count
        # But standard form is fine if we clip.

        term1 = sum_sq_col / count_col
        term2 = mean**2
        variance = term1 - term2

        # Clip negative variance caused by floating point errors
        variance = variance.clip(min=0)

        std = np.sqrt(variance)

        # Mask invalid stats where count is too low
        # Mean requires count >= 1, Std requires count >= 2 (technically, though biased estimator works on 1)
        # Let's return NaNs for count == 0
        mean = mean.where(count_col > 0, np.nan)
        std = std.where(count_col > 1, np.nan)

        return mean, std

    def compute_global_stats(
        self, wisdom_df: pd.DataFrame, load_cached_data: bool = True
    ) -> dict:
        """
        Computes global aggregation stats (Sum, SumSq, Count) on the Wisdom dataset.

        Args:
            wisdom_df: The filtered high-quality dataset.
            load_cached_data: Whether to load from cache.

        Returns:
            Dictionary containing stats DataFrames for each level and type.
        """
        global_stats = {}

        # Check if all needed files exist
        all_cached = True
        if load_cached_data:
            for level in self.levels:
                if not (
                    os.path.exists(self._get_cache_path(level, "route"))
                    and os.path.exists(self._get_cache_path(level, "rate"))
                ):
                    all_cached = False
                    break
        else:
            all_cached = False

        if all_cached:
            print("Loading global stats from cache...")
            for level in self.levels:
                global_stats[f"L{level}_route"] = pd.read_parquet(
                    self._get_cache_path(level, "route")
                )
                global_stats[f"L{level}_rate"] = pd.read_parquet(
                    self._get_cache_path(level, "rate")
                )
            return global_stats

        print("Computing global stats from scratch...")

        # 1. Pre-computation on Wisdom Set
        # We need Geohashes, Hour, and Fare Per Km
        # Working on a copy to avoid modifying original wisdom_df reference if needed elsewhere
        df = wisdom_df.copy()

        # Parse datetime if not already done (metadata loads as object/string usually)
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Fast parse assuming standard format from metadata
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        df["hour"] = df["pickup_datetime"].dt.hour

        # Calculate Distance and Rate
        dist = utils.calculate_haversine(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )
        # Safe division for rate
        safe_dist = np.maximum(dist, 0.001)
        df["fare_per_km"] = df["fare_amount"] / safe_dist

        # Pre-compute Geohashes for all levels
        for level in self.levels:
            df[f"pick_geo_L{level}"] = utils.encode_geohash(
                df["pickup_latitude"].values,
                df["pickup_longitude"].values,
                precision=level,
            )
            df[f"drop_geo_L{level}"] = utils.encode_geohash(
                df["dropoff_latitude"].values,
                df["dropoff_longitude"].values,
                precision=level,
            )

        # 2. Aggregation Loop
        for level in self.levels:
            print(f"  Aggregating Level {level}...")

            # --- Route Stats (Pickup -> Dropoff) ---
            # Group by [pick, drop]
            # Target: fare_amount
            route_grp = df.groupby([f"pick_geo_L{level}", f"drop_geo_L{level}"])[
                "fare_amount"
            ]
            route_stats = route_grp.agg(
                sum_fare="sum", sum_sq_fare=lambda x: np.sum(x**2), count_fare="count"
            ).reset_index()

            # Rename columns for clarity/merging
            route_stats.columns = [
                "pickup_geohash",
                "dropoff_geohash",
                "sum_val",
                "sum_sq_val",
                "count_val",
            ]

            # --- Rate Stats (Pickup + Hour) ---
            # Group by [pick, hour]
            # Target: fare_per_km
            rate_grp = df.groupby([f"pick_geo_L{level}", "hour"])["fare_per_km"]
            rate_stats = rate_grp.agg(
                sum_rate="sum", sum_sq_rate=lambda x: np.sum(x**2), count_rate="count"
            ).reset_index()

            rate_stats.columns = [
                "pickup_geohash",
                "hour",
                "sum_val",
                "sum_sq_val",
                "count_val",
            ]

            # Save to dict and cache
            global_stats[f"L{level}_route"] = route_stats
            global_stats[f"L{level}_rate"] = rate_stats

            route_stats.to_parquet(self._get_cache_path(level, "route"), index=False)
            rate_stats.to_parquet(self._get_cache_path(level, "rate"), index=False)

        return global_stats

    def enrich_data(
        self, target_df: pd.DataFrame, global_stats: dict, mode: str = "test"
    ) -> pd.DataFrame:
        """
        Enriches the target DataFrame with statistical moments.

        Args:
            target_df: The DataFrame to enrich (Learner fold, Val, or Test).
            global_stats: The dictionary of pre-computed wisdom stats.
            mode: 'train', 'val', or 'test'.
                  If 'train', performs Conditional Vectorized Subtraction.

        Returns:
            Enriched DataFrame with new mean/std features.
        """
        # Work on a copy
        df = target_df.copy()

        # Ensure base features exist
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)

        if "hour" not in df.columns:
            df["hour"] = df["pickup_datetime"].dt.hour

        # If in train mode, we need fare_per_km for the subtraction logic
        if mode == "train":
            dist = utils.calculate_haversine(
                df["pickup_latitude"].values,
                df["pickup_longitude"].values,
                df["dropoff_latitude"].values,
                df["dropoff_longitude"].values,
            )
            safe_dist = np.maximum(dist, 0.001)
            df["fare_per_km"] = df["fare_amount"] / safe_dist

            # Identify overlap with Wisdom criteria
            # We use the strict wisdom mask on the current fold
            wisdom_mask = data_manager.get_wisdom_mask(df)
            overlap_df = df[wisdom_mask].copy()

        # Iterate through levels to generate features
        for level in self.levels:
            # Generate Geohashes for this level
            p_col = f"pick_geo_L{level}"
            d_col = f"drop_geo_L{level}"

            df[p_col] = utils.encode_geohash(
                df["pickup_latitude"].values,
                df["pickup_longitude"].values,
                precision=level,
            )
            df[d_col] = utils.encode_geohash(
                df["dropoff_latitude"].values,
                df["dropoff_longitude"].values,
                precision=level,
            )

            if mode == "train":
                # Ensure overlap_df has the geohashes too
                overlap_df[p_col] = df.loc[wisdom_mask, p_col]
                overlap_df[d_col] = df.loc[wisdom_mask, d_col]

            # --- 1. Route Stats (Pickup -> Dropoff) ---
            g_route = global_stats[f"L{level}_route"]

            # Merge Global Stats onto Target
            # We use left join. Missing keys get NaNs (which is correct, implies no prior).
            merged_route = pd.merge(
                df[[p_col, d_col]],
                g_route,
                left_on=[p_col, d_col],
                right_on=["pickup_geohash", "dropoff_geohash"],
                how="left",
            )

            # Extract vectors
            curr_sum = merged_route["sum_val"].fillna(0).astype(float).values
            curr_sum_sq = merged_route["sum_sq_val"].fillna(0).astype(float).values
            curr_count = merged_route["count_val"].fillna(0).astype(float).values

            # Conditional Subtraction
            if mode == "train":
                # Compute aggregates of the overlapping rows in the current fold
                fold_grp = overlap_df.groupby([p_col, d_col])["fare_amount"]
                fold_stats = fold_grp.agg(
                    sum_f="sum", sum_sq_f=lambda x: np.sum(x**2), count_f="count"
                ).reset_index()

                # Map these fold stats to the original df rows
                # Note: We are mapping based on the keys in df.
                # Rows in df that are NOT in overlap_df will get 0 subtraction.
                # Rows in df that ARE in overlap_df will get the subtraction corresponding to their group.

                # We need to join fold_stats back to the full df to align vectors
                fold_mapped = pd.merge(
                    df[[p_col, d_col]],
                    fold_stats,
                    left_on=[p_col, d_col],
                    right_on=[p_col, d_col],
                    how="left",
                )

                sub_sum = fold_mapped["sum_f"].fillna(0).values
                sub_sum_sq = fold_mapped["sum_sq_f"].fillna(0).values
                sub_count = fold_mapped["count_f"].fillna(0).values

                # Apply Subtraction
                curr_sum -= sub_sum
                curr_sum_sq -= sub_sum_sq
                curr_count -= sub_count

            # Calculate Moments
            mean_feat, std_feat = self._calculate_derived_moments(
                curr_sum, curr_sum_sq, curr_count
            )

            df[f"mean_fare_L{level}"] = mean_feat
            df[f"std_fare_L{level}"] = std_feat

            # --- 2. Rate Stats (Pickup + Hour) ---
            g_rate = global_stats[f"L{level}_rate"]

            merged_rate = pd.merge(
                df[[p_col, "hour"]],
                g_rate,
                left_on=[p_col, "hour"],
                right_on=["pickup_geohash", "hour"],
                how="left",
            )

            curr_sum_r = merged_rate["sum_val"].fillna(0).astype(float).values
            curr_sum_sq_r = merged_rate["sum_sq_val"].fillna(0).astype(float).values
            curr_count_r = merged_rate["count_val"].fillna(0).astype(float).values

            if mode == "train":
                fold_grp_r = overlap_df.groupby([p_col, "hour"])["fare_per_km"]
                fold_stats_r = fold_grp_r.agg(
                    sum_r="sum", sum_sq_r=lambda x: np.sum(x**2), count_r="count"
                ).reset_index()

                fold_mapped_r = pd.merge(
                    df[[p_col, "hour"]],
                    fold_stats_r,
                    left_on=[p_col, "hour"],
                    right_on=[p_col, "hour"],
                    how="left",
                )

                sub_sum_r = fold_mapped_r["sum_r"].fillna(0).values
                sub_sum_sq_r = fold_mapped_r["sum_sq_r"].fillna(0).values
                sub_count_r = fold_mapped_r["count_r"].fillna(0).values

                curr_sum_r -= sub_sum_r
                curr_sum_sq_r -= sub_sum_sq_r
                curr_count_r -= sub_count_r

            mean_rate, std_rate = self._calculate_derived_moments(
                curr_sum_r, curr_sum_sq_r, curr_count_r
            )

            df[f"mean_rate_L{level}"] = mean_rate
            df[f"std_rate_L{level}"] = std_rate

            # Cleanup temporary columns to save memory
            df.drop(columns=[p_col, d_col], inplace=True)

        # Cleanup auxiliary columns
        if "fare_per_km" in df.columns:
            df.drop(columns=["fare_per_km"], inplace=True)

        return df
