import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.data_loader import DataLoader


class FeatureEngineer:
    """
    Implements the Entity-First feature engineering pipeline for the TD-SRN solution.
    Handles kinematic window generation, visual feature aggregation, and ground imputation.
    """

    def __init__(self):
        pass

    def create_windowed_features(self, df_tracking: pd.DataFrame) -> pd.DataFrame:
        """
        Generates time-distributed features (lags/leads) for tracking data.
        Converts angles to sin/cos for numerical stability.

        Args:
            df_tracking: Raw tracking dataframe.

        Returns:
            DataFrame with flattened windowed features per player/step.
        """
        # Ensure sorting for correct shifting
        df_tracking = df_tracking.sort_values(["game_play", "nfl_player_id", "step"])

        # 1. Angular Continuity (Explicit Numerical Stability)
        # Convert degrees to radians and then to sin/cos components
        for col in ["direction", "orientation"]:
            rads = np.deg2rad(df_tracking[col].fillna(0))
            df_tracking[f"{col}_sin"] = np.sin(rads)
            df_tracking[f"{col}_cos"] = np.cos(rads)

        # Select features to window
        # We exclude raw angles in favor of sin/cos
        base_features = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "sa",
            "direction_sin",
            "direction_cos",
            "orientation_sin",
            "orientation_cos",
        ]

        # 2. Window Generation (Time-Distributed Input)
        # We generate lags from t-WINDOW_HALF to t+WINDOW_HALF
        dfs = []
        grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

        for lag in range(-Config.WINDOW_HALF, Config.WINDOW_HALF + 1):
            # lag < 0: Future (shift negative), lag > 0: Past (shift positive)
            # We name columns to indicate relative time step
            suffix = f"_t{lag:+d}" if lag != 0 else ""

            # Shift the features
            shifted = grouped[base_features].shift(lag)
            shifted.columns = [f"{c}{suffix}" for c in base_features]
            dfs.append(shifted)

        # Concatenate all windowed features
        df_features = pd.concat(dfs, axis=1)

        # Restore keys
        df_features["game_play"] = df_tracking["game_play"]
        df_features["nfl_player_id"] = df_tracking["nfl_player_id"]
        df_features["step"] = df_tracking["step"]

        return df_features

    def process_visual_features(self, df_helmets: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates helmet boxes using Max-Pooling Selection Strategy.
        Maps 60Hz video frames to 10Hz tracking steps.
        """
        # Map frame to step (59.94Hz video, 10Hz tracking, Snap at frame 300 = step 0)
        # step = round((frame - 300) / 5.994)
        df_helmets["step"] = ((df_helmets["frame"] - 300) / 5.994).round().astype(int)

        # Calculate Box Area for Max-Pooling
        df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

        # Max-Pooling: Select the box with largest area for each (game_play, step, player)
        # This handles multiple views (Sideline/Endzone) by picking the "best" one
        df_best = df_helmets.sort_values("area", ascending=False).drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"]
        )

        # Select features
        keep_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "left",
            "top",
            "width",
            "height",
        ]
        return df_best[keep_cols]

    def generate_features(
        self,
        split: str = "train",
        load_cached_data: bool = True,
        debug_sample: int = None,
    ) -> pd.DataFrame:
        """
        Main pipeline execution.
        Loads data, computes features, merges pairs, imputes ground, and caches results.
        """
        # 0. Cache Check
        cache_path = None
        if split == "train":
            cache_path = Config.CACHE_TRAIN_FEATURES
        elif split == "validation":
            cache_path = Config.CACHE_VAL_FEATURES
        elif split == "test":
            cache_path = Config.CACHE_TEST_FEATURES

        if load_cached_data and cache_path and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        # 1. Load Raw Data
        df_meta = DataLoader.load_metadata(split)
        if debug_sample:
            df_meta = df_meta.head(debug_sample)

        # Load only relevant tracking/helmets to save memory
        game_plays = df_meta["game_play"].unique().tolist()
        df_tracking = DataLoader.load_tracking_data(split, game_plays=game_plays)
        df_helmets = DataLoader.load_helmets_data(split, game_plays=game_plays)

        # 2. Entity-First Engineering
        df_track_proc = self.create_windowed_features(df_tracking)
        df_vis_proc = self.process_visual_features(df_helmets)

        # 3. Merge Player 1 Data
        # Ensure ID consistency
        df_meta["nfl_player_id_1"] = pd.to_numeric(
            df_meta["nfl_player_id_1"], errors="coerce"
        )

        # Merge Tracking
        df_merged = df_meta.merge(
            df_track_proc,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        # Rename P1 columns
        track_cols = [
            c
            for c in df_track_proc.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_dict_p1 = {c: f"p1_{c}" for c in track_cols}
        df_merged = df_merged.rename(columns=rename_dict_p1)

        # Merge Visuals
        df_merged = df_merged.merge(
            df_vis_proc,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        vis_cols = ["left", "top", "width", "height"]
        rename_dict_vis_p1 = {c: f"p1_vis_{c}" for c in vis_cols}
        df_merged = df_merged.rename(columns=rename_dict_vis_p1)

        # 4. Merge Player 2 Data (Handles 'G' later)
        df_merged["nfl_player_id_2_num"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )

        # Merge Tracking
        df_merged = df_merged.merge(
            df_track_proc,
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        rename_dict_p2 = {c: f"p2_{c}" for c in track_cols}
        df_merged = df_merged.rename(columns=rename_dict_p2)

        # Merge Visuals
        df_merged = df_merged.merge(
            df_vis_proc,
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        rename_dict_vis_p2 = {c: f"p2_vis_{c}" for c in vis_cols}
        df_merged = df_merged.rename(columns=rename_dict_vis_p2)

        # 5. Impute Ground Interactions
        # Identify Ground rows
        is_ground = df_merged["nfl_player_id_2"] == "G"

        # Logic: P2 Position = P1 Position (Relative Dist=0), P2 Velocity = 0

        # Impute Positions (x, y)
        for base in ["x_position", "y_position"]:
            # Find all P2 columns for this base feature (including lags)
            p2_cols = [c for c in df_merged.columns if c.startswith(f"p2_{base}")]
            for p2_c in p2_cols:
                p1_c = p2_c.replace("p2_", "p1_")
                if p1_c in df_merged.columns:
                    df_merged.loc[is_ground, p2_c] = df_merged.loc[is_ground, p1_c]

        # Impute Motion (speed, accel, sin, cos) -> Set to 0
        motion_bases = [
            "speed",
            "acceleration",
            "sa",
            "direction_sin",
            "direction_cos",
            "orientation_sin",
            "orientation_cos",
        ]
        for base in motion_bases:
            p2_cols = [c for c in df_merged.columns if c.startswith(f"p2_{base}")]
            for p2_c in p2_cols:
                df_merged.loc[is_ground, p2_c] = 0.0

        # Impute Visuals (P2 Visuals = 0 for Ground)
        for c in vis_cols:
            df_merged.loc[is_ground, f"p2_vis_{c}"] = 0.0

        # 6. Compute Pairwise Features (Distance)
        # We compute log1p distance for all time steps to aid the model
        for lag in range(-Config.WINDOW_HALF, Config.WINDOW_HALF + 1):
            suffix = f"_t{lag:+d}" if lag != 0 else ""

            x1 = df_merged[f"p1_x_position{suffix}"]
            y1 = df_merged[f"p1_y_position{suffix}"]
            x2 = df_merged[f"p2_x_position{suffix}"]
            y2 = df_merged[f"p2_y_position{suffix}"]

            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if Config.USE_LOG_DISTANCE:
                dist = np.log1p(dist)

            df_merged[f"dist{suffix}"] = dist

        # 7. Explicit Numerical Stability (Clamping)
        # Clamp derived/motion features to prevent outliers.
        # Do NOT clamp raw positions (x, y) as they exceed 50.

        num_cols = df_merged.select_dtypes(include=[np.number]).columns
        # Exclude IDs, step, contact, and raw positions
        exclude_keywords = [
            "game_play",
            "step",
            "contact",
            "id",
            "x_position",
            "y_position",
        ]
        cols_to_clamp = [
            c for c in num_cols if not any(k in c for k in exclude_keywords)
        ]

        df_merged[cols_to_clamp] = df_merged[cols_to_clamp].clip(
            Config.CLAMP_MIN, Config.CLAMP_MAX
        )

        # Fill remaining NaNs (e.g. missing tracking for players) with 0
        # This is safe after clamping
        df_merged = df_merged.fillna(0)

        # 8. Save to Cache
        if cache_path:
            Config.setup_directories()
            # Use pyarrow for speed
            df_merged.to_parquet(cache_path, index=False)

        return df_merged
