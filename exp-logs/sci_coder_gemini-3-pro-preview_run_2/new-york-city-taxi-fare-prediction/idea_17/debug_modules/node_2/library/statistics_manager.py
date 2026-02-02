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
        Applies global stats to a training/validation subsample.
        Performs Vectorized Subtraction (LOO) for rows that contributed to the global stats.
        """
        df = df.copy()

        # Ensure coordinates are clamped for consistent grid generation
        df = clamp_coordinates(df)

        # Generate Grid IDs
        df["grid_id"] = get_spatial_grid_id(df, precision=GRID_PRECISION)

        # Determine if rows satisfy WISDOM_CRITERIA (to know if we subtract)
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

        # Merge Global Stats
        # Use left join to keep all training rows
        merged = df.merge(global_stats, on="grid_id", how="left")

        # Fill missing stats with 0 for calculation (will be handled by Universal fallback later if count=0)
        merged["sum_fare"] = merged["sum_fare"].fillna(0)
        merged["sum_fare_sq"] = merged["sum_fare_sq"].fillna(0)
        merged["count"] = merged["count"].fillna(0)

        # Vectorized Subtraction (Leave-One-Out)
        # If is_wisdom is 1, we subtract this row's contribution
        # If is_wisdom is 0, we subtract 0
        # Cite debug_lesson_5: Use .values to prevent index alignment issues between merged df (new index) and mask (old index)
        is_wisdom_values = is_wisdom.values
        adj_sum_fare = merged["sum_fare"] - (merged["fare_amount"] * is_wisdom_values)
        adj_sum_sq = merged["sum_fare_sq"] - (
            (merged["fare_amount"] ** 2) * is_wisdom_values
        )
        adj_count = merged["count"] - is_wisdom_values

        # Compute Statistics
        # Handle division by zero or low counts
        # We require at least 1 sample to compute mean, and preferably more for std,
        # but technically N=1 has std=0 (or undefined sample std).
        # Here we use population-style variance formula logic for feature stability.

        # Initialize with NaNs
        mean_vals = np.full(len(merged), np.nan)
        std_vals = np.full(len(merged), np.nan)

        valid_mask = adj_count > 0

        if valid_mask.any():
            # Mean
            mean_vals[valid_mask] = adj_sum_fare[valid_mask] / adj_count[valid_mask]

            # Variance -> Std
            # Var = E[X^2] - (E[X])^2
            term1 = adj_sum_sq[valid_mask] / adj_count[valid_mask]
            term2 = mean_vals[valid_mask] ** 2
            var = term1 - term2
            # Clip negative variance due to floating point errors
            std_vals[valid_mask] = np.sqrt(np.maximum(0, var))

        # Assign to DataFrame
        df["route_mean_fare"] = mean_vals
        df["route_std_fare"] = std_vals

        # Impute Missing Values with Universal Priors
        # If self.universal_mean is not set (e.g. if aggregate wasn't called on this instance),
        # we calculate it from the passed global_stats
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
