import os
import gc
import numpy as np
import pandas as pd
from library.config import Config


class FeatureEngineer:
    """
    Implements the Entity-First data processing pipeline for the SSE-RVN model.
    Handles tracking lag generation, helmet max-pooling, and hybrid ground imputation.
    """

    def __init__(self):
        self.input_dir = Config.INPUT_DIR
        self.metadata_dir = Config.METADATA_DIR
        self.working_dir = Config.WORKING_DIR
        self.window_size = Config.WINDOW_SIZE  # 11 frames
        self.half_window = self.window_size // 2

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_cache_path(self, name):
        return os.path.join(self.working_dir, f"{name}.parquet")

    def process_tracking_data(
        self, df_tracking, load_cached=True, cache_name="processed_tracking"
    ):
        """
        Generates rolling window features for tracking data.
        """
        cache_path = self._get_cache_path(cache_name)
        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached tracking data from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Processing tracking data (generating lags)...")
        # Ensure sorted
        df_tracking = df_tracking.sort_values(
            ["game_play", "nfl_player_id", "step"]
        ).reset_index(drop=True)

        # Features to lag
        feature_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
        ]

        # Output dataframe initialized with keys
        df_out = df_tracking[["game_play", "nfl_player_id", "step"]].copy()

        # Generate lags: t-5 to t+5
        # We group by game_play and nfl_player_id to ensure lags don't cross boundaries
        grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

        for lag in range(-self.half_window, self.half_window + 1):
            suffix = f"_lag_{lag}"
            for col in feature_cols:
                # shift(-lag) because if lag is -5 (past), we want shift(5)
                # Actually, standard notation: lag -5 means t-5.
                # df.shift(1) gives t-1. So df.shift(5) gives t-5.
                # df.shift(-1) gives t+1.
                # So we use shift(-lag).
                # Example: lag=-5 -> shift(5) -> gets value from 5 steps ago.
                shifted = grouped[col].shift(-lag)
                df_out[f"{col}{suffix}"] = shifted

        # Fill NaNs resulting from shifts (edges of play) with nearest valid observation or 0
        # Forward fill then backward fill within groups is safer for trajectories
        # However, simple fillna(0) is often used in this comp for simplicity,
        # but ffill/bfill is better for physics.
        # Given the strict resource limits, we'll use a global fillna(0) for edge cases
        # after checking if groups preserve alignment.
        # Ideally, we drop rows with NaNs if they are outside valid play, but we need to keep all steps.
        df_out = df_out.fillna(0)

        # Save to cache
        print(f"Saving processed tracking data to {cache_path}")
        df_out.to_parquet(cache_path, index=False)
        return df_out

    def process_helmet_data(
        self, df_helmets, load_cached=True, cache_name="processed_helmets"
    ):
        """
        Applies Max-Pooling strategy to select best helmet view per step.
        """
        cache_path = self._get_cache_path(cache_name)
        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached helmet data from {cache_path}")
            return pd.read_parquet(cache_path)

        print("Processing helmet data (max-pooling views)...")

        # Map frame to step
        # Formula: frame = round((step * 0.1 + 5.0) * 59.94)
        # Inverting: step = round((frame / 59.94 - 5.0) * 10)
        # But we need to map the helmet frames to the steps we have in labels.
        # Instead of calculating step for every helmet frame, we'll calculate the target frame for every step
        # and merge. However, df_helmets is large.
        # Let's calculate 'step' for the helmet data to facilitate grouping.

        # Calculate approximate step for each helmet frame
        # We use a vectorized operation
        # Snap is at 5s.
        df_helmets["step"] = (
            ((df_helmets["frame"] / 59.94 - 5.0) * 10).round().astype(int)
        )

        # Calculate Box Area
        df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

        # Max Pooling: Select row with max area for each (game_play, step, nfl_player_id)
        # Sort by area descending and drop duplicates
        df_best = df_helmets.sort_values("area", ascending=False).drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"]
        )

        # Keep relevant columns
        keep_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "left",
            "top",
            "width",
            "height",
            "view",
        ]
        df_out = df_best[keep_cols].copy()

        # Save to cache
        print(f"Saving processed helmet data to {cache_path}")
        df_out.to_parquet(cache_path, index=False)
        return df_out

    def merge_and_impute(self, df_labels, df_tracking, df_helmets):
        """
        Merges labels with tracking/helmets and handles Ground imputation.
        """
        print("Merging data and imputing Ground features...")

        # ---------------------------------------------------------
        # 1. Merge Tracking for Player 1
        # ---------------------------------------------------------
        # Ensure nfl_player_id is correct type
        df_labels["nfl_player_id_1"] = pd.to_numeric(
            df_labels["nfl_player_id_1"], errors="coerce"
        )

        df_merged = df_labels.merge(
            df_tracking.add_suffix("_1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1"],
            how="left",
        )

        # ---------------------------------------------------------
        # 2. Merge Tracking for Player 2 (Handling 'G')
        # ---------------------------------------------------------
        # We first merge normally for non-G players
        # Convert 'G' to NaN or a dummy for merging
        df_merged["nfl_player_id_2_int"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        df_merged = df_merged.merge(
            df_tracking.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2_int"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
            suffixes=("", "_tracking"),
        )

        # ---------------------------------------------------------
        # 3. Ground Imputation
        # ---------------------------------------------------------
        # Identify Ground rows
        is_ground = df_merged["nfl_player_id_2"] == "G"

        # Columns that have lags
        lag_cols = [
            c
            for c in df_tracking.columns
            if c not in ["game_play", "nfl_player_id", "step"]
        ]

        # For Ground rows:
        # P2 Position = P1 Position (Distance = 0)
        # P2 Velocity/Accel = 0

        for col in lag_cols:
            p1_col = f"{col}_1"
            p2_col = f"{col}_2"

            if "position" in col:
                # Copy P1 position to P2
                df_merged.loc[is_ground, p2_col] = df_merged.loc[is_ground, p1_col]
            else:
                # Set velocity, accel, orientation, direction to 0
                df_merged.loc[is_ground, p2_col] = 0.0

        # ---------------------------------------------------------
        # 4. Feature Engineering (Relative Stats)
        # ---------------------------------------------------------
        # We calculate these for the center frame (lag 0) and potentially others if needed.
        # Typically, distance at lag 0 is the strongest signal.

        # Distance
        dx = df_merged["x_position_lag_0_1"] - df_merged["x_position_lag_0_2"]
        dy = df_merged["y_position_lag_0_1"] - df_merged["y_position_lag_0_2"]
        dist = np.sqrt(dx**2 + dy**2)

        df_merged["distance"] = dist
        df_merged["log_distance"] = np.log1p(dist)

        # Relative Speed (Scalar difference)
        df_merged["relative_speed"] = (
            df_merged["speed_lag_0_1"] - df_merged["speed_lag_0_2"]
        )

        # Relative Angle (Shortest Arc)
        # orientation is 0-360
        diff = (
            df_merged["orientation_lag_0_1"] - df_merged["orientation_lag_0_2"]
        ).abs() % 360
        df_merged["relative_orientation"] = np.minimum(diff, 360 - diff)

        diff_dir = (
            df_merged["direction_lag_0_1"] - df_merged["direction_lag_0_2"]
        ).abs() % 360
        df_merged["relative_direction"] = np.minimum(diff_dir, 360 - diff_dir)

        # ---------------------------------------------------------
        # 5. Merge Helmets
        # ---------------------------------------------------------
        # Merge P1 Helmets
        df_merged = df_merged.merge(
            df_helmets.add_suffix("_1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_1", "step_1", "nfl_player_id_1"],
            how="left",
        )

        # Merge P2 Helmets (Only if not Ground)
        df_merged = df_merged.merge(
            df_helmets.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2_int"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
            suffixes=("", "_helmet"),
        )

        # Fill missing helmet data with 0 (e.g. if no helmet found or Ground)
        visual_cols = ["left", "top", "width", "height"]
        for c in visual_cols:
            df_merged[f"{c}_1"] = df_merged[f"{c}_1"].fillna(0)
            df_merged[f"{c}_2"] = df_merged[f"{c}_2"].fillna(0)

        # ---------------------------------------------------------
        # 6. Explicit Clamping (Physical Constraints)
        # ---------------------------------------------------------
        # Clamp derivatives and relative stats to [-50, 50]
        clamp_cols = ["relative_speed", "acceleration_lag_0_1", "acceleration_lag_0_2"]
        # Also clamp all lag speeds/accels
        for lag in range(-self.half_window, self.half_window + 1):
            clamp_cols.append(f"speed_lag_{lag}_1")
            clamp_cols.append(f"speed_lag_{lag}_2")
            clamp_cols.append(f"acceleration_lag_{lag}_1")
            clamp_cols.append(f"acceleration_lag_{lag}_2")

        for col in clamp_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].clip(Config.CLAMP_MIN, Config.CLAMP_MAX)

        # Cleanup
        drop_cols = [
            c
            for c in df_merged.columns
            if "_1" in c and ("game_play" in c or "step" in c)
        ]
        drop_cols += [
            c
            for c in df_merged.columns
            if "_2" in c and ("game_play" in c or "step" in c)
        ]
        drop_cols += ["nfl_player_id_2_int"]
        df_merged = df_merged.drop(columns=drop_cols, errors="ignore")

        return df_merged

    def generate_features(self, split="train", load_cached=True, debug=False):
        """
        Main entry point to generate features for train or test sets.
        """
        output_path = os.path.join(self.working_dir, f"{split}_features.parquet")

        if load_cached and os.path.exists(output_path):
            print(f"Loading final {split} features from {output_path}")
            return pd.read_parquet(output_path)

        print(f"Generating {split} features from scratch...")

        # 1. Load Metadata / Labels
        meta_file = "train.csv" if split == "train" else "test.csv"
        df_meta = pd.read_csv(os.path.join(self.metadata_dir, meta_file))

        if split == "train" and Config.DEBUG:
            print(f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} rows...")
            df_meta = df_meta.sample(
                min(len(df_meta), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            )

        # 2. Load and Process Tracking
        tracking_file = (
            f"{split}_player_tracking.csv"
            if split == "test"
            else "train_player_tracking.csv"
        )
        df_tracking_raw = pd.read_csv(os.path.join(self.input_dir, tracking_file))

        # Filter tracking to relevant game_plays to save memory
        relevant_gps = df_meta["game_play"].unique()
        df_tracking_raw = df_tracking_raw[
            df_tracking_raw["game_play"].isin(relevant_gps)
        ].copy()

        df_tracking_proc = self.process_tracking_data(
            df_tracking_raw, load_cached=load_cached, cache_name=f"tracking_{split}"
        )
        del df_tracking_raw
        gc.collect()

        # 3. Load and Process Helmets
        helmet_file = (
            f"{split}_baseline_helmets.csv"
            if split == "test"
            else "train_baseline_helmets.csv"
        )
        df_helmets_raw = pd.read_csv(os.path.join(self.input_dir, helmet_file))
        df_helmets_raw = df_helmets_raw[
            df_helmets_raw["game_play"].isin(relevant_gps)
        ].copy()

        df_helmets_proc = self.process_helmet_data(
            df_helmets_raw, load_cached=load_cached, cache_name=f"helmets_{split}"
        )
        del df_helmets_raw
        gc.collect()

        # 4. Merge
        df_final = self.merge_and_impute(df_meta, df_tracking_proc, df_helmets_proc)

        # 5. Save
        print(f"Saving final {split} features to {output_path}")
        df_final.to_parquet(output_path, index=False)

        return df_final
