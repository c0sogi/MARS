import os
import numpy as np
import pandas as pd
from library.config import Config
from library.data_factory import DataFactory
from library.utils import haversine_distance, clamp_coordinates


class KnowledgeBase:
    """
    Manages the Global Knowledge Base (Stage 1).
    Constructs and caches hierarchical statistical priors from the full dataset.
    These priors (Fine Grid, Coarse Grid, Physics Rate) serve as the base_margin
    for the residual learning model.
    """

    def __init__(self):
        self.fine_stats = None
        self.coarse_stats = None
        self.global_rate = None

    def build(self, load_cached_data=True):
        """
        Builds or loads the hierarchical stats.

        Args:
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (fine_stats, coarse_stats, global_rate)
                - fine_stats: DataFrame indexed by (lat, lon) at fine resolution.
                - coarse_stats: DataFrame indexed by (lat, lon) at coarse resolution.
                - global_rate: Float representing global $/km.
        """
        stats_path = Config.GLOBAL_STATS_CACHE_PATH
        # Sidecar file for the scalar global rate
        rate_path = stats_path.replace(".parquet", "_rate.npy")

        # 1. Try Loading from Cache
        if (
            load_cached_data
            and os.path.exists(stats_path)
            and os.path.exists(rate_path)
        ):
            print(f"Loading Global Knowledge Base from {stats_path}...")
            try:
                self.fine_stats = pd.read_parquet(stats_path)
                self.global_rate = float(np.load(rate_path))

                # Derive coarse stats from fine stats to save disk space and ensure consistency
                print("Deriving Coarse Grid stats from Fine Grid stats...")
                self.coarse_stats = self._aggregate_coarse_from_fine(self.fine_stats)

                return self.fine_stats, self.coarse_stats, self.global_rate
            except Exception as e:
                print(f"Error loading cache: {e}. Rebuilding from scratch...")

        # 2. Build from Scratch
        print("Building Global Knowledge Base from scratch...")

        # Load full clean data
        # DataFactory.load_clean_full_train_data() applies the physics filter (Stage 1 logic)
        df = DataFactory.load_clean_full_train_data()

        # Clamp coordinates to ensure grid consistency with training/inference
        # This prevents generating grid cells outside the valid NYC bounding box.
        df = clamp_coordinates(df)

        # A. Calculate Global Rate (Physics Fallback)
        print("Calculating Global Fare Rate (Physics Prior)...")

        # Filter NaNs to prevent propagation (Fix for 0.00 rate bug)
        valid_mask = (
            df["pickup_latitude"].notna()
            & df["pickup_longitude"].notna()
            & df["dropoff_latitude"].notna()
            & df["dropoff_longitude"].notna()
        )
        df_valid = df[valid_mask]

        dists = haversine_distance(
            df_valid["pickup_latitude"].values,
            df_valid["pickup_longitude"].values,
            df_valid["dropoff_latitude"].values,
            df_valid["dropoff_longitude"].values,
        )
        # Avoid zero division
        total_dist = np.sum(dists)
        total_fare = df_valid["fare_amount"].sum()

        if total_dist > 1e-6:
            self.global_rate = total_fare / total_dist
        else:
            self.global_rate = 0.0

        print(f"Global Rate calculated: {self.global_rate:.4f} $/km")

        # B. Calculate Fine Grid Stats
        print(
            f"Aggregating Fine Grid Stats (Resolution: {Config.GRID_RES_FINE} decimals)..."
        )

        # Create grid keys based on Pickup Location (Spatial Smoothing)
        # We round to the defined resolution
        lat_col = "grid_lat"
        lon_col = "grid_lon"

        # Use a copy or assign to new columns to avoid SettingWithCopy if df is a view
        df[lat_col] = df["pickup_latitude"].round(Config.GRID_RES_FINE)
        df[lon_col] = df["pickup_longitude"].round(Config.GRID_RES_FINE)

        # GroupBy: Compute Sum and Count
        # We need these to calculate means later and to perform vectorized subtraction
        grp = df.groupby([lat_col, lon_col])["fare_amount"].agg(["sum", "count"])
        grp.columns = ["fare_sum", "fare_count"]

        self.fine_stats = grp

        # C. Save to Cache
        print(f"Saving Fine Stats to {stats_path}...")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        self.fine_stats.to_parquet(stats_path)

        print(f"Saving Global Rate to {rate_path}...")
        np.save(rate_path, np.array(self.global_rate))

        # D. Derive Coarse Stats
        print(
            f"Deriving Coarse Grid stats (Resolution: {Config.GRID_RES_COARSE} decimals)..."
        )
        self.coarse_stats = self._aggregate_coarse_from_fine(self.fine_stats)

        return self.fine_stats, self.coarse_stats, self.global_rate

    def _aggregate_coarse_from_fine(self, fine_df):
        """
        Aggregates fine grid statistics up to the coarse grid level.
        This is more efficient than re-aggregating the full raw dataset.

        Args:
            fine_df: DataFrame indexed by (lat_fine, lon_fine) with cols (fare_sum, fare_count)

        Returns:
            DataFrame indexed by (lat_coarse, lon_coarse) with cols (fare_sum, fare_count)
        """
        # Reset index to access lat/lon columns
        # The index names are usually preserved in parquet, but we access by position to be safe
        df = fine_df.reset_index()

        # Identify coordinate columns (first two columns after reset)
        lat_col_name = df.columns[0]
        lon_col_name = df.columns[1]

        # Round to Coarse Resolution
        coarse_lat = df[lat_col_name].round(Config.GRID_RES_COARSE)
        coarse_lon = df[lon_col_name].round(Config.GRID_RES_COARSE)

        # Group by coarse coords and sum the statistics
        # Sum of sums is the total sum; Sum of counts is the total count.
        grp = df.groupby([coarse_lat, coarse_lon])[["fare_sum", "fare_count"]].sum()

        return grp
