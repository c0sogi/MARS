import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import compute_geohash_bins, haversine_distance


class HierarchicalStatsEngine:
    """
    Implements the Multi-Moment Hierarchical Dual-Hygiene strategy.
    Computes statistical priors (Mean, Std, Count) for taxi routes at multiple
    spatial resolutions (Geohash L5, L6, L7).
    """

    def __init__(self):
        self.levels = Config.GEOHASH_LEVELS
        self.global_stats = {}  # Stores aggregated stats for each level
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_route_keys(self, df: pd.DataFrame, level: int) -> pd.Series:
        """
        Generates a unique route identifier for each row based on pickup and dropoff coordinates.
        Route ID = Pickup_Bin_ID * 1_000_000_000 + Dropoff_Bin_ID
        """
        p_bins = compute_geohash_bins(
            df["pickup_latitude"].values, df["pickup_longitude"].values, level
        )
        d_bins = compute_geohash_bins(
            df["dropoff_latitude"].values, df["dropoff_longitude"].values, level
        )

        # Combine into a single unique int64 key
        # Max bin ID is approx 3.5e7, so 1e9 is a safe multiplier
        route_keys = p_bins.astype(np.int64) * 1_000_000_000 + d_bins.astype(np.int64)
        return pd.Series(route_keys, index=df.index, name="route_key")

    def _aggregate_stats(self, df: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
        """
        Computes Sum, SumSq, and Count of fare_amount grouped by keys.
        """
        # Create a temporary dataframe for aggregation
        tmp = pd.DataFrame({"key": keys, "fare": df["fare_amount"]})

        # Pre-calculate square for efficiency
        tmp["fare_sq"] = tmp["fare"] ** 2

        # Groupby and aggregate
        stats = tmp.groupby("key").agg(
            sum_fare=("fare", "sum"),
            sum_sq_fare=("fare_sq", "sum"),
            count=("fare", "count"),
        )
        return stats

    def fit(self, wisdom_df: pd.DataFrame, load_cached_data: bool = True):
        """
        Aggregates global statistics on the Wisdom Set.

        Args:
            wisdom_df: The strict-filtered dataset for generating priors.
            load_cached_data: Whether to load pre-computed stats from disk.
        """
        all_cached = True
        for level in self.levels:
            cache_path = os.path.join(self.cache_dir, f"global_stats_L{level}.parquet")
            if not (load_cached_data and os.path.exists(cache_path)):
                all_cached = False
                break

        if all_cached:
            print(f"Loading cached global stats from {self.cache_dir}...")
            for level in self.levels:
                cache_path = os.path.join(
                    self.cache_dir, f"global_stats_L{level}.parquet"
                )
                self.global_stats[level] = pd.read_parquet(cache_path)
            return

        print("Computing global stats on Wisdom Set...")
        for level in self.levels:
            print(f"  Aggregating Level {level}...")
            keys = self._compute_route_keys(wisdom_df, level)
            stats = self._aggregate_stats(wisdom_df, keys)

            self.global_stats[level] = stats

            # Cache
            cache_path = os.path.join(self.cache_dir, f"global_stats_L{level}.parquet")
            stats.to_parquet(cache_path)

        print("Global stats computation complete.")

    def transform_train(self, learner_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies K-Fold Vectorized Subtraction to generate features for the training set.
        Ensures no target leakage by subtracting the fold's contribution from global stats.
        """
        print("Transforming Learner Set with K-Fold Vectorized Subtraction...")
        learner_df = learner_df.copy()

        # Initialize feature columns with NaNs
        for level in self.levels:
            learner_df[f"mean_fare_L{level}"] = np.nan
            learner_df[f"std_fare_L{level}"] = np.nan
            learner_df[f"count_L{level}"] = np.nan

        # Pre-calculate Route Keys for all levels to save time inside loop
        route_keys_map = {}
        for level in self.levels:
            route_keys_map[level] = self._compute_route_keys(learner_df, level)

        # Identify which rows in Learner Set would have contributed to Wisdom Stats
        # We must only subtract rows that passed the Wisdom filters.
        # Re-applying Wisdom logic:
        dist_km = haversine_distance(
            learner_df["pickup_latitude"],
            learner_df["pickup_longitude"],
            learner_df["dropoff_latitude"],
            learner_df["dropoff_longitude"],
        )
        fare_per_km = learner_df["fare_amount"] / (dist_km + 1e-6)

        wisdom_mask = (
            (learner_df["fare_amount"] >= Config.WISDOM_MIN_FARE)
            & (learner_df["fare_amount"] <= Config.WISDOM_MAX_FARE)
            & (fare_per_km < Config.WISDOM_MAX_FARE_PER_KM)
        )

        kf = KFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED
        )

        for fold_idx, (_, val_idx) in enumerate(kf.split(learner_df)):
            # val_idx represents the current fold we are generating features for.
            # We want (Global - Current_Fold_Contribution).

            # Subset of the fold that contributes to wisdom stats
            fold_wisdom_indices = learner_df.index[val_idx][wisdom_mask.iloc[val_idx]]

            # If no rows in this fold contributed to wisdom, subtraction is 0 (skip heavy compute)
            fold_contributes = len(fold_wisdom_indices) > 0

            for level in self.levels:
                global_stats = self.global_stats[level]
                current_keys = route_keys_map[level].iloc[val_idx]

                # 1. Map Global Stats to current fold
                # We use a left join logic via map/reindex
                matched_stats = global_stats.reindex(current_keys).fillna(0)

                if fold_contributes:
                    # 2. Compute Fold Stats (only for rows that pass wisdom filter)
                    # We need stats for the specific keys in this fold
                    fold_df_subset = learner_df.loc[fold_wisdom_indices]
                    fold_keys_subset = route_keys_map[level].loc[fold_wisdom_indices]

                    fold_stats = self._aggregate_stats(fold_df_subset, fold_keys_subset)

                    # 3. Subtract Fold Stats from Global Stats
                    # Align fold_stats to the current_keys (fill non-matching with 0)
                    fold_stats_aligned = fold_stats.reindex(current_keys).fillna(0)

                    # Perform subtraction
                    oof_sum = matched_stats["sum_fare"] - fold_stats_aligned["sum_fare"]
                    oof_sum_sq = (
                        matched_stats["sum_sq_fare"] - fold_stats_aligned["sum_sq_fare"]
                    )
                    oof_count = matched_stats["count"] - fold_stats_aligned["count"]
                else:
                    # No subtraction needed
                    oof_sum = matched_stats["sum_fare"]
                    oof_sum_sq = matched_stats["sum_sq_fare"]
                    oof_count = matched_stats["count"]

                # 4. Clip to ensure numerical stability (handle floating point errors or missing keys)
                oof_count = oof_count.clip(lower=0)
                oof_sum = oof_sum.clip(lower=0)
                oof_sum_sq = oof_sum_sq.clip(lower=0)

                # 5. Compute Moments
                # Add epsilon to count to avoid division by zero
                # If count is 0, mean/std will be 0 (handled by numerator) or handled explicitly
                valid_mask = oof_count > 0

                mean_fare = np.zeros_like(oof_sum)
                std_fare = np.zeros_like(oof_sum)

                # Mean
                mean_fare[valid_mask] = oof_sum[valid_mask] / oof_count[valid_mask]

                # Variance = E[X^2] - (E[X])^2
                # Var = (SumSq / N) - (Mean)^2
                term1 = oof_sum_sq[valid_mask] / oof_count[valid_mask]
                term2 = mean_fare[valid_mask] ** 2
                variance = term1 - term2
                variance = np.maximum(variance, 0)  # Clip negative variance
                std_fare[valid_mask] = np.sqrt(variance)

                # 6. Assign back to dataframe
                learner_df.iloc[
                    val_idx, learner_df.columns.get_loc(f"mean_fare_L{level}")
                ] = mean_fare
                learner_df.iloc[
                    val_idx, learner_df.columns.get_loc(f"std_fare_L{level}")
                ] = std_fare
                learner_df.iloc[
                    val_idx, learner_df.columns.get_loc(f"count_L{level}")
                ] = oof_count

        return learner_df

    def transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps global statistics to the test/validation set (no subtraction).
        """
        print("Transforming Test/Val Set with Global Priors...")
        df = df.copy()

        for level in self.levels:
            keys = self._compute_route_keys(df, level)
            global_stats = self.global_stats[level]

            # Reindex to map stats to the dataframe rows
            # Rows with unknown keys get NaN, we fill with 0
            matched = global_stats.reindex(keys).fillna(0)

            # Extract arrays
            g_sum = matched["sum_fare"].values
            g_sum_sq = matched["sum_sq_fare"].values
            g_count = matched["count"].values

            # Compute moments
            valid_mask = g_count > 0
            mean_fare = np.zeros_like(g_sum)
            std_fare = np.zeros_like(g_sum)

            mean_fare[valid_mask] = g_sum[valid_mask] / g_count[valid_mask]

            term1 = g_sum_sq[valid_mask] / g_count[valid_mask]
            term2 = mean_fare[valid_mask] ** 2
            variance = np.maximum(term1 - term2, 0)
            std_fare[valid_mask] = np.sqrt(variance)

            # Assign
            df[f"mean_fare_L{level}"] = mean_fare
            df[f"std_fare_L{level}"] = std_fare
            df[f"count_L{level}"] = g_count

        return df
