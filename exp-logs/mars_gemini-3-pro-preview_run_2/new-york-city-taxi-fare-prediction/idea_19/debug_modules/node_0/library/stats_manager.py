import os
import gc
import numpy as np
import pandas as pd
from library.config import ProjectConfig
from library.utils import clamp_coordinates, calculate_geohash, haversine_distance


class StatsManager:
    def __init__(self):
        self.cache_dir = ProjectConfig.CACHE_DIR
        self.levels = ProjectConfig.GEOHASH_LEVELS
        self.num_folds = ProjectConfig.NUM_FOLDS

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self):
        paths = {}
        # Global stats paths
        for l in self.levels:
            paths[f"global_L{l}"] = os.path.join(
                self.cache_dir, f"global_stats_L{l}.parquet"
            )
        paths["global_L5_hour"] = os.path.join(
            self.cache_dir, "global_stats_L5_hour.parquet"
        )

        # Fold stats paths
        for l in self.levels:
            paths[f"fold_L{l}"] = os.path.join(
                self.cache_dir, f"fold_stats_L{l}.parquet"
            )

        return paths

    def _assign_folds(self, df):
        """
        Assigns folds deterministically based on the 'key' column.
        Ensures consistency between Wisdom generation and Learner training.
        """
        # hash_array returns uint64, safe for modulo
        return pd.util.hash_array(df["key"].astype(str).values) % self.num_folds

    def compute_global_moments(self, load_cached=True):
        """
        Computes or loads the Hierarchical Distributional Priors (Mean, Std, Count)
        from the Wisdom Set (Strictly Filtered Full Training Data).
        """
        paths = self._get_cache_paths()
        all_exist = all(os.path.exists(p) for p in paths.values())

        if load_cached and all_exist:
            print("Loading stats from cache...")
            stats = {}
            for k, p in paths.items():
                stats[k] = pd.read_parquet(p)
            return stats

        print("Computing global moments from scratch (Wisdom Set)...")

        # Load necessary columns from full training set
        cols = [
            "key",
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]

        # Load data (using pyarrow engine if available for speed)
        try:
            df = pd.read_parquet(ProjectConfig.TRAIN_PATH, columns=cols)
        except:
            df = pd.read_parquet(
                ProjectConfig.TRAIN_PATH, columns=cols, engine="fastparquet"
            )

        # 1. Input Sanitization (Safety First)
        df = clamp_coordinates(df, inplace=True)

        # 2. Feature Engineering for Filtering
        df["distance"] = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # 3. Wisdom Filter (Strict Hygiene)
        # Exclude noise and extreme outliers to build clean priors
        mask = (
            (df["fare_amount"] >= ProjectConfig.WISDOM_MIN_FARE)
            & (df["fare_amount"] <= ProjectConfig.WISDOM_MAX_FARE)
            & (df["distance"] > 0.2)  # Min 200m to avoid GPS noise
            & (
                (df["fare_amount"] / df["distance"])
                <= ProjectConfig.WISDOM_MAX_FARE_PER_KM
            )
        )
        wisdom = df.loc[mask].copy()

        # Clean up memory
        del df
        gc.collect()

        # 4. Pre-compute Geohashes and Time
        print("Generating Geohashes for Wisdom Set...")
        for l in self.levels:
            wisdom[f"geohash_{l}"] = calculate_geohash(
                wisdom["pickup_latitude"].values, wisdom["pickup_longitude"].values, l
            )

        wisdom["pickup_datetime"] = pd.to_datetime(wisdom["pickup_datetime"])
        wisdom["hour"] = wisdom["pickup_datetime"].dt.hour

        # Pre-compute aggregation targets
        wisdom["fare_sq"] = wisdom["fare_amount"] ** 2
        wisdom["fare_per_km"] = wisdom["fare_amount"] / wisdom["distance"]

        # 5. Assign Folds
        wisdom["fold"] = self._assign_folds(wisdom)

        stats = {}

        # 6. Aggregations
        for l in self.levels:
            g_key = f"geohash_{l}"
            print(f"Aggregating Global and Fold Stats for L{l}...")

            # Global Stats: Sum, SumSq, Count
            agg = (
                wisdom.groupby(g_key)
                .agg({"fare_amount": "sum", "fare_sq": "sum", "key": "count"})
                .rename(
                    columns={
                        "fare_amount": "sum_fare",
                        "fare_sq": "sum_sq",
                        "key": "count",
                    }
                )
            )
            # Cast to appropriate types to save space
            agg = agg.astype(
                {"sum_fare": "float32", "sum_sq": "float32", "count": "int32"}
            )
            stats[f"global_L{l}"] = agg
            agg.to_parquet(paths[f"global_L{l}"])

            # Fold Stats: Sum, SumSq, Count per (Fold, Geohash)
            f_agg = (
                wisdom.groupby(["fold", g_key])
                .agg({"fare_amount": "sum", "fare_sq": "sum", "key": "count"})
                .rename(
                    columns={
                        "fare_amount": "sum_fare",
                        "fare_sq": "sum_sq",
                        "key": "count",
                    }
                )
            )
            f_agg = f_agg.astype(
                {"sum_fare": "float32", "sum_sq": "float32", "count": "int32"}
            )
            stats[f"fold_L{l}"] = f_agg
            f_agg.to_parquet(paths[f"fold_L{l}"])

        # L5 + Hour (Mean Fare Per Km) - Global Only
        print("Aggregating Global L5 + Hour...")
        l5_h_agg = wisdom.groupby(["geohash_5", "hour"]).agg(
            {"fare_per_km": ["mean", "count"]}
        )
        l5_h_agg.columns = ["mean_fare_per_km", "count"]
        # Filter sparse buckets
        l5_h_agg = l5_h_agg[l5_h_agg["count"] >= 5]
        l5_h_agg = l5_h_agg.astype({"mean_fare_per_km": "float32", "count": "int32"})

        stats["global_L5_hour"] = l5_h_agg
        l5_h_agg.to_parquet(paths["global_L5_hour"])

        del wisdom
        gc.collect()

        return stats

    def compute_kfold_moments(self, df, stats):
        """
        Enriches the dataframe with Hierarchical Distributional Priors.

        Logic:
        - If 'fold' column exists (Training/Learner Mode):
          Performs Vectorized Subtraction: Stats = (Global - Fold_k)
          This ensures strict leakage prevention.
        - If 'fold' column missing (Inference Mode):
          Uses Global Stats directly.
        """
        df = df.copy()

        # Ensure Geohashes exist
        for l in self.levels:
            if f"geohash_{l}" not in df.columns:
                df[f"geohash_{l}"] = calculate_geohash(
                    df["pickup_latitude"].values, df["pickup_longitude"].values, l
                )

        # Ensure Hour exists
        if "hour" not in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
                df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
            df["hour"] = df["pickup_datetime"].dt.hour

        # Check if we are in Training Mode (Learner Set)
        # We detect this by presence of 'fare_amount'. If so, we assign folds to enable subtraction.
        is_train = False
        if "fare_amount" in df.columns:
            if "fold" not in df.columns:
                df["fold"] = self._assign_folds(df)
            is_train = True

        # 1. Hierarchical Moments (L5, L6, L7)
        for l in self.levels:
            g_stats = stats[f"global_L{l}"]
            f_stats = stats.get(f"fold_L{l}")

            # Merge Global Stats
            # Left join to keep all rows. Unseen geohashes get NaNs.
            df = df.merge(
                g_stats,
                left_on=f"geohash_{l}",
                right_index=True,
                how="left",
                suffixes=("", "_g"),
            )

            # Fill NaNs with 0 for calculation
            df["sum_fare"] = df["sum_fare"].fillna(0)
            df["sum_sq"] = df["sum_sq"].fillna(0)
            df["count"] = df["count"].fillna(0)

            # Vectorized Subtraction for Training
            if is_train and f_stats is not None:
                # Merge Fold Stats on (Fold, Geohash)
                df = df.merge(
                    f_stats,
                    left_on=["fold", f"geohash_{l}"],
                    right_index=True,
                    how="left",
                    suffixes=("", "_f"),
                )

                # Fill NaNs (if no data in that fold for that geohash, subtract 0)
                df["sum_fare_f"] = df["sum_fare_f"].fillna(0)
                df["sum_sq_f"] = df["sum_sq_f"].fillna(0)
                df["count_f"] = df["count_f"].fillna(0)

                # Subtract
                eff_sum = df["sum_fare"] - df["sum_fare_f"]
                eff_sq = df["sum_sq"] - df["sum_sq_f"]
                eff_count = df["count"] - df["count_f"]
            else:
                # Inference Mode: Use Global directly
                eff_sum = df["sum_fare"]
                eff_sq = df["sum_sq"]
                eff_count = df["count"]

            # Compute Moments (Mean, Std)
            mask_valid = eff_count > 0

            mean_col = f"mean_fare_L{l}"
            std_col = f"std_fare_L{l}"
            count_col = f"count_L{l}"

            # Initialize with default values (-1 for missing/sparse)
            df[mean_col] = -1.0
            df[std_col] = -1.0
            df[count_col] = eff_count

            # Calculate Mean
            df.loc[mask_valid, mean_col] = eff_sum[mask_valid] / eff_count[mask_valid]

            # Calculate Std
            # Var = E[X^2] - (E[X])^2
            mean_vals = df.loc[mask_valid, mean_col]
            sq_mean = eff_sq[mask_valid] / eff_count[mask_valid]
            var = sq_mean - (mean_vals**2)
            var = var.clip(lower=0)  # Fix float precision issues
            df.loc[mask_valid, std_col] = np.sqrt(var)

            # Cleanup temporary columns
            cols_to_drop = ["sum_fare", "sum_sq", "count"]
            if is_train:
                cols_to_drop += ["sum_fare_f", "sum_sq_f", "count_f"]

            # Drop only if they exist
            df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        # 2. Temporal Rate (L5 + Hour)
        # Using Global stats (no subtraction) for this secondary feature
        l5h_stats = stats["global_L5_hour"]
        # Rename columns in stats before merge to avoid collision
        l5h_stats_renamed = l5h_stats.rename(
            columns={
                "mean_fare_per_km": "mean_fare_per_km_L5_hour",
                "count": "count_L5_hour",
            }
        )

        df = df.merge(
            l5h_stats_renamed,
            left_on=["geohash_5", "hour"],
            right_index=True,
            how="left",
        )

        # Fill missing
        df["mean_fare_per_km_L5_hour"] = df["mean_fare_per_km_L5_hour"].fillna(-1)
        df["count_L5_hour"] = df["count_L5_hour"].fillna(0)

        return df
