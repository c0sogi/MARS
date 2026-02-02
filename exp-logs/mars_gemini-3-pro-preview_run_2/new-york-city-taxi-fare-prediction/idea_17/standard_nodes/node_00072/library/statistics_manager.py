import pandas as pd
import numpy as np
import os
from library.config import (
    TRAIN_DATA_PATH,
    GLOBAL_STATS_PATH,
    WISDOM_CRITERIA,
    NYC_BOUNDING_BOX,
    GRID_PRECISION,
)
from library.spatial_ops import (
    clamp_coordinates,
    get_spatial_grid_id,
    haversine_distance,
)


class RouteStatCalculator:
    """
    Implements Variance-Aware Dual-Hygiene Gradient Boosting strategy.
    Calculates global route statistics (Mean, Variance) from the full dataset
    and applies them to training data using Vectorized Subtraction (LOO)
    to prevent leakage.
    """

    def __init__(self):
        self.universal_mean = None
        self.universal_std = None

    def aggregate_global_stats(self, load_cached_data=True):
        """
        Aggregates sum_fare, sum_fare_sq, and count for every spatial grid
        using the full training set, filtered by WISDOM_CRITERIA.
        """
        # 1. Cache Check
        if load_cached_data and os.path.exists(GLOBAL_STATS_PATH):
            print(f"Loading global stats from {GLOBAL_STATS_PATH}...")
            stats_df = pd.read_parquet(GLOBAL_STATS_PATH)

            # Re-calculate universal priors from loaded stats
            total_sum = stats_df["sum_fare"].sum()
            total_sq = stats_df["sum_fare_sq"].sum()
            total_count = stats_df["count"].sum()
            self.universal_mean = total_sum / total_count
            var = (total_sq / total_count) - (self.universal_mean**2)
            self.universal_std = np.sqrt(np.maximum(0, var))

            return stats_df

        print("Computing global stats from scratch...")

        # 2. Load Data
        # Using pyarrow engine for speed on large files
        df = pd.read_parquet(TRAIN_DATA_PATH, engine="pyarrow")

        # 3. Spatial Hygiene
        df = clamp_coordinates(df)

        # 4. Filter by WISDOM_CRITERIA
        # Calculate distance for filtering
        dist_km = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Criteria Masks
        mask_min_fare = df["fare_amount"] >= WISDOM_CRITERIA["min_fare"]
        mask_max_fare = df["fare_amount"] <= WISDOM_CRITERIA["max_fare"]
        mask_min_dist = dist_km >= WISDOM_CRITERIA["min_dist_km"]

        # Fare per KM check
        # Avoid division by zero by using the mask_min_dist
        fare_per_km = np.zeros_like(dist_km)
        valid_dist = dist_km > 0.001
        fare_per_km[valid_dist] = (
            df.loc[valid_dist, "fare_amount"] / dist_km[valid_dist]
        )
        mask_rate = fare_per_km <= WISDOM_CRITERIA["max_fare_per_km"]

        # Combined Wisdom Mask
        is_wisdom = mask_min_fare & mask_max_fare & mask_min_dist & mask_rate
        wisdom_df = df[is_wisdom].copy()

        # 5. Generate Grid IDs
        wisdom_df["grid_id"] = get_spatial_grid_id(wisdom_df, precision=GRID_PRECISION)

        # 6. Aggregation
        wisdom_df["fare_sq"] = wisdom_df["fare_amount"] ** 2

        stats_df = (
            wisdom_df.groupby("grid_id")
            .agg(
                sum_fare=("fare_amount", "sum"),
                sum_fare_sq=("fare_sq", "sum"),
                count=("fare_amount", "count"),
            )
            .reset_index()
        )

        # 7. Calculate Universal Priors (before saving)
        total_sum = stats_df["sum_fare"].sum()
        total_sq = stats_df["sum_fare_sq"].sum()
        total_count = stats_df["count"].sum()

        self.universal_mean = total_sum / total_count
        var = (total_sq / total_count) - (self.universal_mean**2)
        self.universal_std = np.sqrt(np.maximum(0, var))

        print(f"Global Stats Computed. Rows: {len(stats_df)}")
        print(
            f"Universal Mean: {self.universal_mean}, Universal Std: {self.universal_std}"
        )

        # 8. Save to Cache
        # Ensure directory exists
        os.makedirs(os.path.dirname(GLOBAL_STATS_PATH), exist_ok=True)
        stats_df.to_parquet(GLOBAL_STATS_PATH)

        return stats_df

    def retrieve_and_subtract_priors(self, df, global_stats):
        """
        Applies global stats to a training/validation subsample using K-Fold Vectorized Subtraction.
        Cite solution_lesson_node_00071: Prefer K-Fold Target Encoding over LOO to prevent leakage.
        Cite solution_lesson_node_00045: Use Vectorized Subtraction for efficiency.
        """
        df = df.copy()

        # Ensure coordinates are clamped for consistent grid generation
        df = clamp_coordinates(df)

        # Generate Grid IDs
        df["grid_id"] = get_spatial_grid_id(df, precision=GRID_PRECISION)

        # Assign Folds (K=5)
        # We use a deterministic seed for reproducibility of the fold assignment
        np.random.seed(42)
        n_splits = 5
        df["fold"] = np.random.randint(0, n_splits, size=len(df))

        # Determine if rows satisfy WISDOM_CRITERIA (to know if they contributed to global stats)
        dist_km = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        mask_min_fare = df["fare_amount"] >= WISDOM_CRITERIA["min_fare"]
        mask_max_fare = df["fare_amount"] <= WISDOM_CRITERIA["max_fare"]
        mask_min_dist = dist_km >= WISDOM_CRITERIA["min_dist_km"]

        fare_per_km = np.zeros_like(dist_km)
        valid_dist = dist_km > 0.001
        fare_per_km[valid_dist] = (
            df.loc[valid_dist, "fare_amount"] / dist_km[valid_dist]
        )
        mask_rate = fare_per_km <= WISDOM_CRITERIA["max_fare_per_km"]

        is_wisdom = (mask_min_fare & mask_max_fare & mask_min_dist & mask_rate).astype(
            int
        )

        # Prepare Fold Stats
        # We only sum contributions from rows that are 'wisdom'
        df["w_fare"] = df["fare_amount"] * is_wisdom
        df["w_fare_sq"] = (df["fare_amount"] ** 2) * is_wisdom
        df["w_count"] = is_wisdom

        # Aggregate stats per (grid_id, fold)
        fold_stats = (
            df.groupby(["grid_id", "fold"])
            .agg(
                fold_sum_fare=("w_fare", "sum"),
                fold_sum_sq=("w_fare_sq", "sum"),
                fold_count=("w_count", "sum"),
            )
            .reset_index()
        )

        # Merge Global Stats to the main dataframe
        # Use left join to keep all training rows
        merged = df.merge(global_stats, on="grid_id", how="left")

        # Merge Fold Stats to the dataframe
        merged = merged.merge(fold_stats, on=["grid_id", "fold"], how="left")

        # Fill missing stats
        for col in [
            "sum_fare",
            "sum_fare_sq",
            "count",
            "fold_sum_fare",
            "fold_sum_sq",
            "fold_count",
        ]:
            merged[col] = merged[col].fillna(0)

        # Vectorized Subtraction (Global - Fold)
        # This gives us the stats from "everything else" (Rest of World)
        adj_sum_fare = merged["sum_fare"] - merged["fold_sum_fare"]
        adj_sum_sq = merged["sum_fare_sq"] - merged["fold_sum_sq"]
        adj_count = merged["count"] - merged["fold_count"]

        # Compute Statistics
        mean_vals = np.full(len(merged), np.nan)
        std_vals = np.full(len(merged), np.nan)

        valid_mask = adj_count > 0

        if valid_mask.any():
            # Mean
            mean_vals[valid_mask] = adj_sum_fare[valid_mask] / adj_count[valid_mask]

            # Variance -> Std
            term1 = adj_sum_sq[valid_mask] / adj_count[valid_mask]
            term2 = mean_vals[valid_mask] ** 2
            var = term1 - term2
            std_vals[valid_mask] = np.sqrt(np.maximum(0, var))

        # Assign to DataFrame
        df["route_mean_fare"] = mean_vals
        df["route_std_fare"] = std_vals

        # Impute Missing Values with Universal Priors
        if self.universal_mean is None:
            total_sum = global_stats["sum_fare"].sum()
            total_count = global_stats["count"].sum()
            total_sq = global_stats["sum_fare_sq"].sum()
            self.universal_mean = total_sum / total_count
            var = (total_sq / total_count) - (self.universal_mean**2)
            self.universal_std = np.sqrt(np.maximum(0, var))

        df["route_mean_fare"] = df["route_mean_fare"].fillna(self.universal_mean)
        df["route_std_fare"] = df["route_std_fare"].fillna(self.universal_std)

        # Clean up temporary columns
        df.drop(columns=["fold", "w_fare", "w_fare_sq", "w_count"], inplace=True)

        return df

    def retrieve_priors_test(self, df, global_stats):
        """
        Applies global stats to the test set.
        No subtraction is performed.
        """
        df = df.copy()
        df = clamp_coordinates(df)
        df["grid_id"] = get_spatial_grid_id(df, precision=GRID_PRECISION)

        merged = df.merge(global_stats, on="grid_id", how="left")

        # Initialize
        mean_vals = np.full(len(merged), np.nan)
        std_vals = np.full(len(merged), np.nan)

        # Valid mask (where we found a match in global stats)
        valid_mask = merged["count"].notna() & (merged["count"] > 0)

        if valid_mask.any():
            mean_vals[valid_mask] = (
                merged.loc[valid_mask, "sum_fare"] / merged.loc[valid_mask, "count"]
            )

            term1 = (
                merged.loc[valid_mask, "sum_fare_sq"] / merged.loc[valid_mask, "count"]
            )
            term2 = mean_vals[valid_mask] ** 2
            var = term1 - term2
            std_vals[valid_mask] = np.sqrt(np.maximum(0, var))

        df["route_mean_fare"] = mean_vals
        df["route_std_fare"] = std_vals

        # Impute
        if self.universal_mean is None:
            total_sum = global_stats["sum_fare"].sum()
            total_count = global_stats["count"].sum()
            total_sq = global_stats["sum_fare_sq"].sum()
            self.universal_mean = total_sum / total_count
            var = (total_sq / total_count) - (self.universal_mean**2)
            self.universal_std = np.sqrt(np.maximum(0, var))

        df["route_mean_fare"] = df["route_mean_fare"].fillna(self.universal_mean)
        df["route_std_fare"] = df["route_std_fare"].fillna(self.universal_std)

        return df
