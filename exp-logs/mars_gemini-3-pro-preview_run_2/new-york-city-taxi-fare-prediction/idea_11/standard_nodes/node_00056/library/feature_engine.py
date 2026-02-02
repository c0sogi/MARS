import numpy as np
import pandas as pd
from library.config import Config
from library.utils import haversine_distance, manhattan_distance, clamp_coordinates


class MarginCalculator:
    """
    Calculates the base_margin (prediction prior) for the taxi fare model.
    Implements the Hierarchical Prior logic: Fine Grid -> Coarse Grid -> Physics Model.
    Handles 'Vectorized Subtraction' for training data to prevent leakage.
    """

    def __init__(self, fine_stats, coarse_stats, global_rate):
        """
        Args:
            fine_stats (pd.DataFrame): Aggregated stats at fine resolution.
            coarse_stats (pd.DataFrame): Aggregated stats at coarse resolution.
            global_rate (float): Global fare per km rate.
        """
        self.fine_stats = fine_stats
        self.coarse_stats = coarse_stats
        self.global_rate = global_rate

    def calculate_margin(self, df, is_training=False):
        """
        Computes the base margin for the given dataframe.

        Args:
            df (pd.DataFrame): Input data.
            is_training (bool): If True, performs vectorized subtraction of the current
                                batch from the global stats to prevent leakage.

        Returns:
            np.array: The calculated base_margin values.
        """
        # Validation
        if is_training and "fare_amount" not in df.columns:
            raise ValueError(
                "is_training=True requires 'fare_amount' column in input dataframe."
            )

        # --- Level 3: Physics Model (Fallback) ---
        # Calculate Haversine distance for the physics model
        dist = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )
        # Prior = Global Rate * Distance
        # We use the global rate derived from the full dataset
        base_margin = self.global_rate * dist

        # --- Helper for Grid Lookup ---
        def get_grid_stats(stats_df, lat_vals, lon_vals):
            # Create temp df for merging
            # Note: We use the values directly to avoid index alignment issues
            tmp = pd.DataFrame({"lat": lat_vals, "lon": lon_vals})

            # Reset index of stats to access lat/lon as columns
            # We assume stats_df is indexed by [lat, lon]
            stats_reset = stats_df.reset_index()

            # Merge
            # stats_reset columns: [lat_col, lon_col, fare_sum, fare_count]
            # We assume the first two columns are the keys (lat, lon)
            key_lat = stats_reset.columns[0]
            key_lon = stats_reset.columns[1]

            merged = pd.merge(
                tmp,
                stats_reset,
                left_on=["lat", "lon"],
                right_on=[key_lat, key_lon],
                how="left",
            )

            return merged["fare_sum"].values, merged["fare_count"].values

        # --- Level 2: Coarse Grid ---
        # Round coordinates to coarse resolution
        coarse_lat = df["pickup_latitude"].round(Config.GRID_RES_COARSE).values
        coarse_lon = df["pickup_longitude"].round(Config.GRID_RES_COARSE).values

        c_sum, c_count = get_grid_stats(self.coarse_stats, coarse_lat, coarse_lon)

        # Handle NaNs (cells not in stats)
        c_sum = np.nan_to_num(c_sum)
        c_count = np.nan_to_num(c_count)

        # Vectorized Subtraction (Leakage Prevention)
        # If the current batch is part of the global stats (Training set),
        # we must subtract its contribution to avoid memorization.
        if is_training:
            c_sum -= df["fare_amount"].values
            c_count -= 1

        # Compute Average
        # Mask where count is sufficient
        mask_coarse = c_count >= Config.PRIOR_COUNT_THRESHOLD

        # Safe division
        c_avg = np.divide(c_sum, c_count, out=np.zeros_like(c_sum), where=c_count > 0)

        # Apply Coarse Prior where valid (overrides Physics)
        base_margin[mask_coarse] = c_avg[mask_coarse]

        # --- Level 1: Fine Grid ---
        # Round coordinates to fine resolution
        fine_lat = df["pickup_latitude"].round(Config.GRID_RES_FINE).values
        fine_lon = df["pickup_longitude"].round(Config.GRID_RES_FINE).values

        f_sum, f_count = get_grid_stats(self.fine_stats, fine_lat, fine_lon)

        f_sum = np.nan_to_num(f_sum)
        f_count = np.nan_to_num(f_count)

        if is_training:
            f_sum -= df["fare_amount"].values
            f_count -= 1

        mask_fine = f_count >= Config.PRIOR_COUNT_THRESHOLD
        f_avg = np.divide(f_sum, f_count, out=np.zeros_like(f_sum), where=f_count > 0)

        # Apply Fine Prior where valid (overrides Coarse and Physics)
        base_margin[mask_fine] = f_avg[mask_fine]

        return base_margin


class FeatureEngineer:
    """
    Orchestrates feature engineering and base margin calculation.
    """

    def __init__(self, margin_calculator):
        self.margin_calculator = margin_calculator

    def process(self, df, is_training=False):
        """
        Generates features and adds base_margin/residual.

        Args:
            df (pd.DataFrame): Input dataframe.
            is_training (bool): Flag for training mode (affects margin calc and residual).

        Returns:
            pd.DataFrame: Processed dataframe with new features.
        """
        # Work on a copy
        df = df.copy()

        # 1. Clamp Coordinates (Safety)
        # Prevents linear extrapolation risks outside the city
        df = clamp_coordinates(df)

        # 2. Datetime Features
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Optimization: Check first row for ' UTC' suffix to avoid full string scan if possible
            if df["pickup_datetime"].dtype == "object":
                # Optimistic check
                if len(df) > 0 and str(df["pickup_datetime"].iloc[0]).endswith(" UTC"):
                    df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, -4)

            df["pickup_datetime"] = pd.to_datetime(
                df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
            )

        df["hour"] = df["pickup_datetime"].dt.hour
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
        df["year"] = df["pickup_datetime"].dt.year
        df["month"] = df["pickup_datetime"].dt.month

        # 3. Physical/Spatial Features
        # Haversine Distance
        df["dist_haversine"] = haversine_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Manhattan Distance
        df["dist_manhattan"] = manhattan_distance(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["dropoff_longitude"].values,
        )

        # Coordinate Deltas
        df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
        df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]

        # Rotated Coordinates (45 degrees)
        # Helps tree models split diagonal roads
        df["pickup_rot_1"] = df["pickup_latitude"] + df["pickup_longitude"]
        df["pickup_rot_2"] = df["pickup_latitude"] - df["pickup_longitude"]
        df["dropoff_rot_1"] = df["dropoff_latitude"] + df["dropoff_longitude"]
        df["dropoff_rot_2"] = df["dropoff_latitude"] - df["dropoff_longitude"]

        # 4. Prior Feature Construction
        # Cite solution_lesson_node_00055: Supply priors as input features ("Soft Integration")
        # rather than enforcing them as base margins.
        mean_fare_estimate = self.margin_calculator.calculate_margin(
            df, is_training=is_training
        )
        df["mean_fare_estimate"] = mean_fare_estimate

        # 5. Residual Calculation -> REMOVED
        # We predict 'fare_amount' directly using 'mean_fare_estimate' as a strong feature.

        return df
