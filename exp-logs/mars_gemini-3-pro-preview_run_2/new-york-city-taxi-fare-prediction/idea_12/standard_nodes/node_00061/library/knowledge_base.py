import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import clean_memory


class KnowledgeBase:
    """
    Implements K-Fold Target Encoding with Vectorized Subtraction.
    Cite {solution_lesson_node_00045}
    Cite {solution_lesson_node_00060}
    """

    def __init__(self, config: Config):
        self.config = config

    def _calculate_rate(self, df):
        epsilon = 1e-6
        return df["fare_amount"] / (df["dist_haversine"] + epsilon)

    def _get_global_stats(self, df, key_col, target_col):
        """Computes Sum and Count for the entire dataframe."""
        stats = df.groupby(key_col)[target_col].agg(["sum", "count"])
        return stats

    def _apply_stats(self, df, stats_df, prefix):
        """Merges stats back to df."""
        df = df.merge(
            stats_df.rename(
                columns={"mean": f"{prefix}_mean", "count": f"{prefix}_cnt"}
            ),
            on=stats_df.index.name,
            how="left",
        )
        return df

    def process_kfold(self, train_df, load_cached_data=True):
        """
        Performs K-Fold Target Encoding on the training set.
        Generates out-of-fold statistics for 'key_fine' and 'key_coarse'.
        """
        cache_path = self.config.get_cache_path("train_encoded.parquet")
        if load_cached_data and os.path.exists(cache_path):
            print("Loading encoded training data from cache...")
            return pd.read_parquet(cache_path)

        print("Performing K-Fold Target Encoding on Training Set...")

        # Prepare targets
        train_df["temp_rate"] = self._calculate_rate(train_df)

        # Global Stats (needed for test set later, but also for subtraction)
        # We compute them here but return the enriched df.
        # To save memory, we process one key type at a time or handle carefully.

        # Assign folds
        train_df["fold"] = np.random.randint(0, self.config.N_FOLDS, size=len(train_df))

        # Initialize columns
        for col in ["fine_fare", "fine_rate", "coarse_fare", "coarse_rate"]:
            train_df[col] = np.nan

        # --- Helper for Vectorized Subtraction ---
        def encode_key(df, key_col, value_col, out_col_mean):
            print(f"  Encoding {key_col} -> {out_col_mean}...")

            # 1. Global Sums
            global_stats = df.groupby(key_col)[value_col].agg(["sum", "count"])

            # 2. Iterate Folds
            for f in range(self.config.N_FOLDS):
                # Identify fold rows
                fold_mask = df["fold"] == f
                fold_indices = df.index[fold_mask]

                # Compute stats for this fold
                fold_data = df.loc[fold_mask]
                fold_stats = fold_data.groupby(key_col)[value_col].agg(["sum", "count"])

                # Align with global stats (subset to keys present in fold)
                # We need global stats for keys in this fold
                current_global = global_stats.loc[fold_stats.index]

                # Subtract to get "Rest" stats
                rest_sum = current_global["sum"] - fold_stats["sum"]
                rest_count = current_global["count"] - fold_stats["count"]

                # Avoid div by zero
                rest_mean = rest_sum / rest_count.replace(0, np.nan)

                # Map back to fold rows
                # Using merge on subset is cleaner than map for large data
                rest_df = pd.DataFrame({out_col_mean: rest_mean})

                # Update original dataframe
                # We use a temporary merge
                merged = fold_data[[key_col]].merge(rest_df, on=key_col, how="left")

                # Assign values (using numpy array assignment for speed/safety)
                train_df.loc[fold_mask, out_col_mean] = merged[out_col_mean].values

            clean_memory()

        # Encode Fine Fare
        encode_key(train_df, "key_fine", "fare_amount", "fine_fare")
        # Encode Fine Rate
        encode_key(train_df, "key_fine", "temp_rate", "fine_rate")
        # Encode Coarse Fare
        encode_key(train_df, "key_coarse", "fare_amount", "coarse_fare")
        # Encode Coarse Rate
        encode_key(train_df, "key_coarse", "temp_rate", "coarse_rate")

        # Global Means for fallback
        global_mean_fare = train_df["fare_amount"].mean()
        global_mean_rate = train_df["temp_rate"].mean()

        # Smart Fallback
        train_df["smart_fare"] = (
            train_df["fine_fare"]
            .fillna(train_df["coarse_fare"])
            .fillna(global_mean_fare)
        )
        train_df["smart_rate"] = (
            train_df["fine_rate"]
            .fillna(train_df["coarse_rate"])
            .fillna(global_mean_rate)
        )

        # Cleanup
        train_df.drop(columns=["temp_rate", "fold"], inplace=True)

        print("Saving encoded training data...")
        train_df.to_parquet(cache_path, index=False)
        return train_df

    def process_test(self, test_df, train_df):
        """
        Enriches test/validation data using global stats from the full training set.
        """
        print("Enriching Test/Val data with Global Priors...")

        # Compute global stats from full train
        train_df["temp_rate"] = self._calculate_rate(train_df)

        def get_stats(key_col, val_col, name):
            return train_df.groupby(key_col)[val_col].mean().rename(name)

        # Fine
        fine_fare = get_stats("key_fine", "fare_amount", "fine_fare")
        fine_rate = get_stats("key_fine", "temp_rate", "fine_rate")

        # Coarse
        coarse_fare = get_stats("key_coarse", "fare_amount", "coarse_fare")
        coarse_rate = get_stats("key_coarse", "temp_rate", "coarse_rate")

        global_mean_fare = train_df["fare_amount"].mean()
        global_mean_rate = train_df["temp_rate"].mean()

        train_df.drop(columns=["temp_rate"], inplace=True)

        # Merge
        test_df = test_df.merge(fine_fare, on="key_fine", how="left")
        test_df = test_df.merge(fine_rate, on="key_fine", how="left")
        test_df = test_df.merge(coarse_fare, on="key_coarse", how="left")
        test_df = test_df.merge(coarse_rate, on="key_coarse", how="left")

        # Fallback
        test_df["smart_fare"] = (
            test_df["fine_fare"].fillna(test_df["coarse_fare"]).fillna(global_mean_fare)
        )
        test_df["smart_rate"] = (
            test_df["fine_rate"].fillna(test_df["coarse_rate"]).fillna(global_mean_rate)
        )

        return test_df
