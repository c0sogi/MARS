import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage
from library.feature_shared import (
    project_vector_to_body_frame,
    create_temporal_lags,
)
from library.data_manager import DataManager


class StreamBFeatureGenerator:
    """
    Implements the Hybrid-Coordinate Kinematic Pipeline for Player-Ground impacts.
    Generates features for Stream B:
    - Field-Centric Anchor (Raw Position, Velocity, Acceleration, Jerk)
    - Ego-Centric Augmentation (Surge/Sway projections)
    - Physics Derivatives (Pose-Motion Alignment)
    - Exponential Temporal Lags
    """

    def __init__(self):
        self.config = Config
        self.data_manager = DataManager()
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def generate_features(self, mode="train", load_cached_data=True):
        """
        Generates or loads Stream B features.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to try loading from cache.

        Returns:
            pd.DataFrame: Feature matrix for Stream B.
        """
        cache_file = os.path.join(self.working_dir, f"features_stream_b_{mode}.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"[Stream B] Loading cached features from {cache_file}...")
            return pd.read_parquet(cache_file)

        print(f"[Stream B] Generating features for {mode}...")

        # 1. Load Merged Data
        df = self.data_manager.load_dataset(mode=mode, load_cached_data=True)

        # 2. Filter for Player-Ground Interactions Only
        # Stream B is strictly for P-G.
        print(
            f"[Stream B] Filtering for Player-Ground interactions (Total rows before: {len(df)})..."
        )
        df = df[df["nfl_player_id_2"] == "G"].copy()
        print(f"[Stream B] Rows after filtering: {len(df)}")

        if len(df) == 0:
            print("[Stream B] Warning: No Player-Ground interactions found.")
            return pd.DataFrame()

        # 3. Kinematic Derivation (Derivatives & Components)
        print("[Stream B] Computing Kinematic Derivatives...")

        # Ensure data is sorted for time-based differentiation
        df.sort_values(by=["game_play", "nfl_player_id_1", "step"], inplace=True)

        # Convert Speed/Direction to Cartesian Velocity (Field Frame)
        # Direction: 0=North(Y), 90=East(X)
        rad_dir = np.radians(df["direction_p1"].fillna(0))
        df["v_x_p1"] = df["speed_p1"] * np.sin(rad_dir)
        df["v_y_p1"] = df["speed_p1"] * np.cos(rad_dir)

        # Calculate Acceleration Vector (Derivative of Velocity)
        # Group by play/player to prevent boundary bleeding
        # Time delta is 0.1s
        dt = 0.1

        # Helper for grouped diff
        def calc_diff(series):
            return series.diff().fillna(0) / dt

        # Group object
        grouped = df.groupby(["game_play", "nfl_player_id_1"])

        df["a_x_p1_calc"] = grouped["v_x_p1"].transform(calc_diff)
        df["a_y_p1_calc"] = grouped["v_y_p1"].transform(calc_diff)

        # Pose-Motion Alignment (Cos similarity between Orientation and Direction)
        # Orientation: 0=North(Y), Increases Clockwise
        # Direction: 0=North(Y), Increases Clockwise
        # Alignment = cos(orientation - direction)
        rad_orient = np.radians(df["orientation_p1"].fillna(0))
        df["pose_motion_alignment_p1"] = np.cos(rad_orient - rad_dir)

        # Cyclic Encoding for Angles
        for col in ["direction_p1", "orientation_p1"]:
            rad = np.radians(df[col].fillna(0))
            df[f"{col}_sin"] = np.sin(rad)
            df[f"{col}_cos"] = np.cos(rad)

        # 4. Ego-Centric Augmentation (Projection)
        print("[Stream B] Projecting vectors to Body Frame (Ego-Centric)...")

        # Project Velocity
        # Note: We use the components derived from speed/direction
        v_proj = project_vector_to_body_frame(
            df, "v_x_p1", "v_y_p1", "orientation_p1", "v"
        )

        # Project Acceleration
        # We use the calculated acceleration vector (a_x_p1_calc) rather than raw magnitude
        # to ensure correct vector projection relative to body
        a_proj = project_vector_to_body_frame(
            df, "a_x_p1_calc", "a_y_p1_calc", "orientation_p1", "a"
        )

        # Concatenate projections
        df = pd.concat([df, v_proj, a_proj], axis=1)

        # 5. Temporal Lags
        print("[Stream B] Applying Temporal Lags...")

        # Define features to lag
        # Field-Centric Anchor
        # Removed jerk_mag_p1 to avoid noise amplification (Cite Lesson 00032)
        raw_feats = [
            "x_position_p1",
            "y_position_p1",
            "speed_p1",
            "acceleration_p1",
            "sa_p1",
            "direction_p1_sin",
            "direction_p1_cos",
            "orientation_p1_sin",
            "orientation_p1_cos",
            "pose_motion_alignment_p1",
        ]

        # Ego-Centric Augmentation
        # Removed jerk projections (Cite Lesson 00062)
        ego_feats = ["v_surge", "v_sway", "a_surge", "a_sway"]

        features_to_lag = raw_feats + ego_feats

        # Lags from config
        lags = self.config.FEATURE_CONFIG["tracking_lags"]

        # Group columns for lag generation
        group_cols = ["game_play", "nfl_player_id_1"]

        df_lags = create_temporal_lags(df, group_cols, features_to_lag, lags)
        df = pd.concat([df, df_lags], axis=1)

        # 6. Cleanup and Save
        # Drop intermediate calculation columns
        drop_cols = [
            "v_x_p1",
            "v_y_p1",
            "a_x_p1_calc",
            "a_y_p1_calc",
        ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # Keep only necessary columns: Identifiers, Target, Base Features, Lag Features
        # The dataframe currently has all original columns + new features.
        # We rely on the model trainer to select X and y, but we should ensure
        # we don't carry unnecessary heavy columns if any.

        df = reduce_mem_usage(df)

        print(f"[Stream B] Saving {len(df)} rows to {cache_file}...")
        df.to_parquet(cache_file, index=False)

        return df
