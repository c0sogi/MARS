import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import clean_memory


class KnowledgeBase:
    """
    Implements the Disjoint Knowledge Decoupling strategy.

    Responsibilities:
    1. Compute statistical priors (Mean Fare, Mean Rate) from the strictly filtered
       Background Knowledge Base at multiple resolutions (Fine, Coarse, Temporal).
    2. Enrich Foreground (Train) and Test datasets by joining these priors
       and implementing fallback logic (Fine -> Coarse -> Global).
    """

    def __init__(self, config: Config):
        self.config = config

    def _calculate_rate(self, df):
        """
        Helper to calculate Fare Per Km safely.
        Assumes df has 'fare_amount' and 'dist_haversine'.
        """
        # Avoid division by zero.
        # Note: Strict filter in DataLoader ensures Fare <= 10 * Dist.
        # If Dist is 0, Fare must be 0. 0/0 -> NaN, which we can fill with 0.
        # Adding a small epsilon to distance to be safe.
        epsilon = 1e-6
        return df["fare_amount"] / (df["dist_haversine"] + epsilon)

    def compute_priors(self, background_df, load_cached_data=True):
        """
        Computes aggregated statistics from the Background Knowledge Base.

        Args:
            background_df: DataFrame containing the Background set.
                           Must be pre-featurized (have key_fine, key_coarse, key_temporal).
            load_cached_data: Whether to load from disk if available.

        Returns:
            Dictionary containing DataFrames/Values for:
            - 'fine': Stats by key_fine
            - 'coarse': Stats by key_coarse
            - 'temporal': Stats by key_temporal
            - 'global': Global mean fare and rate
        """
        # Cache paths
        cache_fine = self.config.get_cache_path("priors_fine.parquet")
        cache_coarse = self.config.get_cache_path("priors_coarse.parquet")
        cache_temporal = self.config.get_cache_path("priors_temporal.parquet")
        cache_global = self.config.get_cache_path("priors_global.npy")

        # 1. Try Loading Cache
        if (
            load_cached_data
            and os.path.exists(cache_fine)
            and os.path.exists(cache_coarse)
            and os.path.exists(cache_temporal)
            and os.path.exists(cache_global)
        ):

            print("Loading Knowledge Base priors from cache...")
            try:
                priors = {
                    "fine": pd.read_parquet(cache_fine),
                    "coarse": pd.read_parquet(cache_coarse),
                    "temporal": pd.read_parquet(cache_temporal),
                    "global": np.load(cache_global, allow_pickle=True).item(),
                }
                return priors
            except Exception as e:
                print(f"Failed to load priors cache: {e}. Re-computing.")

        # 2. Compute from Scratch
        if background_df is None:
            raise ValueError("background_df cannot be None if cache is missing.")

        print("Computing Knowledge Base priors from Background set...")

        # Calculate Rate (Fare/Km) for aggregation
        # We work on a copy or assign temporarily to avoid modifying original if strictly needed,
        # but adding a column is usually fine.
        background_df["temp_rate"] = self._calculate_rate(background_df)

        # --- Global Stats ---
        global_mean_fare = background_df["fare_amount"].mean()
        global_mean_rate = background_df["temp_rate"].mean()

        global_stats = {
            "mean_fare": float(global_mean_fare),
            "mean_rate": float(global_mean_rate),
        }

        # --- Aggregation Helper ---
        def aggregate(key_col):
            # Group by key and calculate stats
            # We use float32 to save memory
            stats = (
                background_df.groupby(key_col)
                .agg(
                    mean_fare=("fare_amount", "mean"),
                    mean_rate=("temp_rate", "mean"),
                    count=("fare_amount", "count"),
                )
                .reset_index()
            )

            stats["mean_fare"] = stats["mean_fare"].astype(np.float32)
            stats["mean_rate"] = stats["mean_rate"].astype(np.float32)
            stats["count"] = stats["count"].astype(np.int32)
            return stats

        # --- Fine-Grained Stats ---
        print("  Aggregating Fine-Grained Route stats...")
        fine_stats = aggregate("key_fine")

        # --- Coarse-Grained Stats ---
        print("  Aggregating Coarse-Grained Route stats...")
        coarse_stats = aggregate("key_coarse")

        # --- Temporal-Spatial Stats ---
        print("  Aggregating Temporal-Spatial stats...")
        temporal_stats = aggregate("key_temporal")

        # Cleanup temp column
        background_df.drop(columns=["temp_rate"], inplace=True)
        clean_memory()

        # 3. Save to Cache
        print("Saving priors to cache...")
        self.config.setup_dirs()

        fine_stats.to_parquet(cache_fine, index=False)
        coarse_stats.to_parquet(cache_coarse, index=False)
        temporal_stats.to_parquet(cache_temporal, index=False)
        np.save(cache_global, global_stats)

        priors = {
            "fine": fine_stats,
            "coarse": coarse_stats,
            "temporal": temporal_stats,
            "global": global_stats,
        }

        return priors

    def enrich_dataset(self, df, priors):
        """
        Enriches a dataset (Foreground Train, Val, or Test) with the computed priors.
        Performs Left Joins and implements the 'Smart Fallback' logic.

        Args:
            df: Target DataFrame to enrich. Must have spatial keys.
            priors: Dictionary returned by compute_priors().

        Returns:
            Enriched DataFrame with new columns.
        """
        print(f"Enriching dataset with {len(df)} rows...")

        # Unpack priors
        fine_stats = priors["fine"]
        coarse_stats = priors["coarse"]
        temporal_stats = priors["temporal"]
        global_stats = priors["global"]

        # Rename columns to avoid collisions and clarify source
        # We do this on copies or renaming during merge would be cleaner,
        # but let's just assume standard names from compute_priors: mean_fare, mean_rate, count

        # 1. Join Fine Stats
        df = df.merge(
            fine_stats.rename(
                columns={
                    "mean_fare": "fine_fare",
                    "mean_rate": "fine_rate",
                    "count": "fine_cnt",
                }
            ),
            on="key_fine",
            how="left",
        )

        # 2. Join Coarse Stats
        df = df.merge(
            coarse_stats.rename(
                columns={
                    "mean_fare": "coarse_fare",
                    "mean_rate": "coarse_rate",
                    "count": "coarse_cnt",
                }
            ),
            on="key_coarse",
            how="left",
        )

        # 3. Join Temporal Stats
        df = df.merge(
            temporal_stats.rename(
                columns={
                    "mean_fare": "temporal_fare",
                    "mean_rate": "temporal_rate",
                    "count": "temporal_cnt",
                }
            ),
            on="key_temporal",
            how="left",
        )

        # 4. Handle Missing Values / Smart Fallback
        # Counts: Fill NaN with 0
        for col in ["fine_cnt", "coarse_cnt", "temporal_cnt"]:
            df[col] = df[col].fillna(0).astype(np.int32)

        # Smart Fallback for Fare
        # Logic: Try Fine -> Try Coarse -> Fallback to Global
        # We create a specific feature for this to guide the tree
        df["smart_fare"] = df["fine_fare"]
        df["smart_fare"] = df["smart_fare"].fillna(df["coarse_fare"])
        df["smart_fare"] = df["smart_fare"].fillna(global_stats["mean_fare"])

        # Smart Fallback for Rate
        df["smart_rate"] = df["fine_rate"]
        df["smart_rate"] = df["smart_rate"].fillna(df["coarse_rate"])
        df["smart_rate"] = df["smart_rate"].fillna(global_stats["mean_rate"])

        # We also leave the individual columns (fine_fare, etc.) with NaNs
        # because XGBoost can handle them and might learn confidence based on them.
        # However, filling them with -1 or global mean is also an option.
        # Given XGBoost, leaving as NaN (or whatever merge produced) is usually best
        # so the split direction handles "missing data".

        # Optimization: Downcast floats
        cols_to_float32 = [
            "fine_fare",
            "fine_rate",
            "coarse_fare",
            "coarse_rate",
            "temporal_fare",
            "temporal_rate",
            "smart_fare",
            "smart_rate",
        ]
        for col in cols_to_float32:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        clean_memory()
        return df
