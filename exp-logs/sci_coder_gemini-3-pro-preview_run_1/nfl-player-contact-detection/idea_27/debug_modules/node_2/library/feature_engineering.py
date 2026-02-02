import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import CacheManager


class FeatureEngineer:
    """
    Implements Multi-Scale Vector-Aligned Feature Engineering.
    Handles Relaxed Quadratic Gating and Collision-Aligned Basis transformations.
    """

    def __init__(self):
        self.cache_manager = CacheManager()
        self.working_dir = Config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def process_features(self, df, split="train", load_cached_data=True):
        """
        Main entry point for feature engineering.

        Args:
            df (pd.DataFrame): Merged metadata and tracking data.
            split (str): Dataset split ('train', 'val', 'test').
            load_cached_data (bool): Whether to load from cache.

        Returns:
            pd.DataFrame: DataFrame with generated features.
        """
        # Define cache filename
        cache_file = f"features_{split}_full.parquet"

        # 1. Check Cache
        if load_cached_data and self.cache_manager.exists(cache_file):
            # print(f"Loading features for {split} from cache...")
            return self.cache_manager.load_parquet(cache_file)

        # print(f"Computing features for {split}...")

        # 2. Preprocessing & Sorting
        # Ensure data is sorted for temporal operations
        # Construct a unique pair ID for grouping
        df["pair_id"] = (
            df["game_play"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        )

        df = df.sort_values(by=["pair_id", "step"]).reset_index(drop=True)

        # 3. Apply Relaxed Quadratic Gating (Train/Val only)
        # We filter out pairs that never get close enough to interact
        if split in ["train", "val"]:
            df = self.apply_quadratic_gating(df)

        # 4. Compute Collision-Aligned Features
        df = self.compute_collision_features(df)

        # 5. Filter Columns
        # Keep only the features defined in Config plus identifiers and target
        keep_cols = ["contact_id", "game_play", "step", "pair_id"] + Config.FEATURE_COLS
        if "contact" in df.columns:
            keep_cols.append("contact")

        # Ensure all columns exist (fill missing with 0 if any calculation failed)
        for col in Config.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df = df[keep_cols].copy()

        # 6. Save to Cache
        self.cache_manager.save_parquet(df, cache_file)

        return df

    def apply_quadratic_gating(self, df):
        """
        Filters player pairs based on a relaxed distance threshold.
        Keeps pairs where min(distance) < threshold OR pairs involving Ground.
        """
        # Calculate raw Euclidean distance
        # Handle Ground: If x_position_p2 is NaN, it's a Ground interaction
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Assign to dataframe temporarily
        df["temp_dist"] = dist

        # Identify Ground interactions (where P2 coordinates are NaN)
        is_ground = df["x_position_p2"].isna()

        # For Ground, set distance to 0.0 effectively for the gating check (always keep)
        df.loc[is_ground, "temp_dist"] = 0.0

        # Group by pair and find minimum distance
        # transform('min') broadcasts the min value to all rows of the group
        min_dists = df.groupby("pair_id")["temp_dist"].transform("min")

        # Filter
        mask = min_dists < Config.GATING_THRESHOLD

        # Drop temp column
        df = df.drop(columns=["temp_dist"])

        return df[mask].reset_index(drop=True)

    def compute_collision_features(self, df):
        """
        Computes vector-aligned features and multi-scale temporal aggregates.
        """
        # --- 1. Vector Basis Construction ---

        # Convert Polar (Speed/Direction) to Cartesian (Vx, Vy)
        # Assumption: NFL Standard 0=Y (North), 90=X (East), Clockwise?
        # Or Standard Math 0=X, 90=Y?
        # Standard conversion: Vx = S * sin(deg), Vy = S * cos(deg) usually works for relative magnitude
        # regardless of reference as long as it's consistent.

        # Player 1
        rad_p1 = np.deg2rad(df["direction_p1"].fillna(0))
        df["vx_p1"] = df["speed_p1"] * np.sin(rad_p1)
        df["vy_p1"] = df["speed_p1"] * np.cos(rad_p1)
        df["ax_p1"] = df["acceleration_p1"] * np.sin(
            rad_p1
        )  # Project accel along motion
        df["ay_p1"] = df["acceleration_p1"] * np.cos(rad_p1)

        # Player 2
        rad_p2 = np.deg2rad(df["direction_p2"].fillna(0))
        df["vx_p2"] = df["speed_p2"] * np.sin(rad_p2)
        df["vy_p2"] = df["speed_p2"] * np.cos(rad_p2)
        df["ax_p2"] = df["acceleration_p2"] * np.sin(rad_p2)
        df["ay_p2"] = df["acceleration_p2"] * np.cos(rad_p2)

        # Handle Ground (P2 is NaN) -> Fill with 0 for vector calcs
        for col in [
            "vx_p2",
            "vy_p2",
            "ax_p2",
            "ay_p2",
            "x_position_p2",
            "y_position_p2",
        ]:
            df[col] = df[col].fillna(0.0)

        # Relative Vectors (P1 - P2)
        rx = df["x_position_p1"] - df["x_position_p2"]
        ry = df["y_position_p1"] - df["y_position_p2"]
        distance = np.sqrt(rx**2 + ry**2)

        # Avoid division by zero
        distance_safe = distance.replace(0, 1e-6)

        # Unit Vector r (Collision Axis: P2 -> P1)
        # If P2 is origin (Ground), this is vector from origin to P1
        ux = rx / distance_safe
        uy = ry / distance_safe

        # Unit Vector t (Tangent Axis: Orthogonal to r)
        # Rotate 90 degrees: (-y, x)
        tx = -uy
        ty = ux

        # Relative Velocity & Acceleration
        rvx = df["vx_p1"] - df["vx_p2"]
        rvy = df["vy_p1"] - df["vy_p2"]
        rax = df["ax_p1"] - df["ax_p2"]
        ray = df["ay_p1"] - df["ay_p2"]

        # --- 2. Projections (Collision-Aligned Basis) ---

        # Radial Components (Impact)
        df["v_r"] = rvx * ux + rvy * uy
        df["a_r"] = rax * ux + ray * uy

        # Tangential Components (Shear/Pass-by)
        df["v_t"] = rvx * tx + rvy * ty
        df["a_t"] = rax * tx + ray * ty

        # --- 3. Ground Handling ---

        # Identify Ground Rows
        # We check if the original x_position_p2 was NaN (before we filled it with 0)
        # But we filled it in place. We can check nfl_player_id_2.
        # 'G' is usually in metadata, but merged df might have NaNs in p2 cols.
        # Let's rely on the sentinel value logic.

        # If nfl_player_id_2 is 'G' or NaN (if merge failed for G)
        is_ground = (df["nfl_player_id_2"] == "G") | (df["nfl_player_id_2"].isna())

        # Set Distance Sentinel
        df.loc[is_ground, "distance"] = Config.SENTINEL_VALUE
        df.loc[~is_ground, "distance"] = distance[~is_ground]

        # For Ground, override projections with raw P1 physics
        # Impact with ground is defined by P1's own speed/accel
        df.loc[is_ground, "v_r"] = df.loc[is_ground, "speed_p1"]
        df.loc[is_ground, "v_t"] = 0.0
        df.loc[is_ground, "a_r"] = df.loc[is_ground, "acceleration_p1"]
        df.loc[is_ground, "a_t"] = 0.0

        # --- 4. Multi-Scale Temporal Processing ---

        # We need to perform rolling operations per pair
        # Data is already sorted by pair_id, step

        # Define GroupBy object
        grp = df.groupby("pair_id")

        # Macro Scale: Approach Trajectory (Trend)
        # Window: +/- MACRO_WINDOW (e.g., 10 steps)
        # We use center=True to look ahead and behind
        df["v_r_macro_mean"] = grp["v_r"].transform(
            lambda x: x.rolling(
                window=2 * Config.MACRO_WINDOW + 1, min_periods=1, center=True
            ).mean()
        )
        df["v_t_macro_mean"] = grp["v_t"].transform(
            lambda x: x.rolling(
                window=2 * Config.MACRO_WINDOW + 1, min_periods=1, center=True
            ).mean()
        )

        # Micro Scale: Impact Impulse (Shock)
        # Jerk: First difference of acceleration
        df["a_r_jerk"] = grp["a_r"].diff().fillna(0)

        # Energy: RMS of acceleration over small window
        # Window: +/- MICRO_WINDOW (e.g., 2 steps)
        def rolling_rms(x):
            return np.sqrt(
                (x**2)
                .rolling(window=2 * Config.MICRO_WINDOW + 1, min_periods=1, center=True)
                .mean()
            )

        df["a_r_energy"] = grp["a_r"].transform(rolling_rms)

        # Fill any remaining NaNs from rolling ops
        cols_to_fill = ["v_r_macro_mean", "v_t_macro_mean", "a_r_energy"]
        df[cols_to_fill] = df[cols_to_fill].fillna(0)

        return df
