import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config


class FeatureEngineering:
    """
    Handles the 'Entity-First' data processing pipeline for LRP-Net.
    """

    def __init__(self):
        self.scaler = None
        self.feature_cols = []

    def _get_window_cols(self, col_name):
        """Generates column names for the temporal window."""
        cols = []
        for t in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_t{t}" if t < 0 else f"_t+{t}" if t > 0 else ""
            cols.append(f"{col_name}{suffix}")
        return cols

    def process_tracking(self, df_tracking):
        """
        Generates time-series features (lags, windows) on the tracking dataframe.
        """
        # Sort for windowing
        df_tracking = df_tracking.sort_values(
            ["game_play", "nfl_player_id", "step"]
        ).reset_index(drop=True)

        # Group by game_play and player to ensure boundaries are respected
        grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

        # Base columns that identify the row
        base_cols = ["game_play", "nfl_player_id", "step"]
        df_processed = df_tracking[base_cols].copy()

        for col in Config.KINEMATIC_COLS:
            # Create shifts for window t-WINDOW to t+WINDOW
            for t in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
                shift_col_name = (
                    f"{col}_t{t}" if t < 0 else f"{col}_t+{t}" if t > 0 else col
                )
                # shift(-t): t=-1 (past) -> shift(1); t=1 (future) -> shift(-1)
                shifted = grouped[col].shift(-t)
                df_processed[shift_col_name] = shifted

        # Fill missing window values (start/end of play)
        # groupby().ffill() drops the grouping keys, so we must restore them
        filled = df_processed.groupby(["game_play", "nfl_player_id"]).ffill().bfill()
        filled["game_play"] = df_processed["game_play"]
        filled["nfl_player_id"] = df_processed["nfl_player_id"]
        df_processed = filled

        # Fill any remaining NaNs (e.g. track shorter than window) with 0
        df_processed = df_processed.fillna(0)

        return df_processed

    def process_visuals(self, df_helmets):
        """
        Applies Max-Pooling Selection Strategy for helmet boxes.
        """
        # Calculate box area
        df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

        # Sort by area descending so the first item per group is the largest
        df_helmets = df_helmets.sort_values("area", ascending=False)

        # Drop duplicates keeping the first (largest area)
        cols_to_keep = ["game_play", "nfl_player_id", "frame"] + Config.VISUAL_COLS
        df_unique = df_helmets.drop_duplicates(
            subset=["game_play", "nfl_player_id", "frame"], keep="first"
        )

        return df_unique[cols_to_keep]

    def merge_and_impute(self, df_meta, df_tracking_proc, df_vis_proc):
        """
        Combines datasets and handles Ground imputation.
        """
        # Ensure types match for merging
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_tracking_proc["nfl_player_id"] = df_tracking_proc["nfl_player_id"].astype(
            str
        )

        # Identify tracking columns (excluding keys)
        track_cols = [
            c
            for c in df_tracking_proc.columns
            if c not in ["game_play", "nfl_player_id", "step"]
        ]

        # --- Merge Player 1 ---
        p1_rename = {c: f"{c}_p1" for c in track_cols}
        merged = (
            df_meta.merge(
                df_tracking_proc,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .drop(columns=["nfl_player_id"])
            .rename(columns=p1_rename)
        )

        # --- Merge Player 2 ---
        # Handle Ground ID "G" temporarily as string
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        p2_rename = {c: f"{c}_p2" for c in track_cols}

        merged = (
            merged.merge(
                df_tracking_proc,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .drop(columns=["nfl_player_id"])
            .rename(columns=p2_rename)
        )

        # --- Ground Imputation ---
        is_ground = merged["nfl_player_id_2"] == "G"

        # For Ground rows, P2 pos = P1 pos (Distance=0), P2 vel/accel = 0
        for t in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_t{t}" if t < 0 else f"_t+{t}" if t > 0 else ""

            # Position columns: Copy P1 to P2
            for axis in ["x_position", "y_position"]:
                col_p1 = f"{axis}{suffix}_p1"
                col_p2 = f"{axis}{suffix}_p2"
                merged.loc[is_ground, col_p2] = merged.loc[is_ground, col_p1]

            # Derivative columns: Set P2 to 0
            for feat in ["speed", "acceleration", "orientation", "direction", "sa"]:
                col_p2 = f"{feat}{suffix}_p2"
                merged.loc[is_ground, col_p2] = 0.0

        # Fill remaining NaNs (missing tracking) with 0
        merged = merged.fillna(0)

        # --- Derived Features (Time t) ---
        # Log Distance
        dx = merged["x_position_p1"] - merged["x_position_p2"]
        dy = merged["y_position_p1"] - merged["y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)
        merged["log_distance"] = np.log1p(dist)

        # Relative Speed (Simple Diff)
        merged["speed_diff"] = merged["speed_p1"] - merged["speed_p2"]

        # Relative Angle (Shortest Arc)
        ori_diff = (merged["orientation_p1"] - merged["orientation_p2"]).abs()
        merged["orientation_diff"] = np.minimum(ori_diff, 360 - ori_diff)

        # --- Merge Visuals ---
        # Map step to frame approximation: frame = 300 + step * 6
        merged["frame_approx"] = (300 + merged["step"] * 6).astype(int)

        # P1 Visuals
        vis_cols = Config.VISUAL_COLS
        p1_vis_rename = {c: f"{c}_p1" for c in vis_cols}
        merged = merged.merge(
            df_vis_proc,
            left_on=["game_play", "nfl_player_id_1", "frame_approx"],
            right_on=["game_play", "nfl_player_id", "frame"],
            how="left",
        ).drop(columns=["nfl_player_id", "frame"])
        merged = merged.rename(columns=p1_vis_rename)

        # P2 Visuals
        p2_vis_rename = {c: f"{c}_p2" for c in vis_cols}
        merged = merged.merge(
            df_vis_proc,
            left_on=["game_play", "nfl_player_id_2", "frame_approx"],
            right_on=["game_play", "nfl_player_id", "frame"],
            how="left",
        ).drop(columns=["nfl_player_id", "frame", "frame_approx"])
        merged = merged.rename(columns=p2_vis_rename)

        # Impute Visuals (0 if missing or Ground)
        for c in vis_cols:
            merged[f"{c}_p1"] = merged[f"{c}_p1"].fillna(0)
            merged[f"{c}_p2"] = merged[f"{c}_p2"].fillna(0)

        return merged

    def preprocess(self, df, fit=False):
        """
        Applies explicit clamping and StandardScaler.
        """
        # Construct feature list
        kin_cols = []
        for c in Config.KINEMATIC_COLS:
            kin_cols.extend(self._get_window_cols(f"{c}_p1"))
            kin_cols.extend(self._get_window_cols(f"{c}_p2"))

        derived_cols = ["log_distance", "speed_diff", "orientation_diff"]
        vis_cols = [f"{c}_p1" for c in Config.VISUAL_COLS] + [
            f"{c}_p2" for c in Config.VISUAL_COLS
        ]

        feature_list = kin_cols + derived_cols + vis_cols
        self.feature_cols = feature_list

        X = df[feature_list].copy()

        # Explicit Clamping for physical derivatives to prevent outliers
        # We clamp speed, acceleration, sa, and speed_diff
        clamp_targets = [
            c for c in X.columns if any(x in c for x in ["speed", "acceleration", "sa"])
        ]
        X[clamp_targets] = X[clamp_targets].clip(Config.CLAMP_MIN, Config.CLAMP_MAX)

        # Convert to float32 for memory efficiency
        X = X.astype(np.float32)

        # Standardization
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            joblib.dump(self.scaler, Config.SCALER_PATH)
        else:
            if self.scaler is None:
                if os.path.exists(Config.SCALER_PATH):
                    self.scaler = joblib.load(Config.SCALER_PATH)
                else:
                    # Fallback: fit on current data (only if scaler missing in inference)
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X)
                    return X_scaled, (
                        df[Config.TARGET_COL].values
                        if Config.TARGET_COL in df
                        else None
                    )

            X_scaled = self.scaler.transform(X)

        y = (
            df[Config.TARGET_COL].values.astype(np.float32)
            if Config.TARGET_COL in df
            else None
        )

        return X_scaled, y

    def load_and_process_data(self, split="train", debug=False, load_cached_data=True):
        """
        Orchestrates data loading, processing, and caching.
        """
        Config.setup_directories()

        cache_X = os.path.join(Config.WORKING_DIR, f"{split}_X.npy")
        cache_y = os.path.join(Config.WORKING_DIR, f"{split}_y.npy")
        cache_meta = os.path.join(Config.WORKING_DIR, f"{split}_meta.parquet")

        # Check Cache
        if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_meta):
            print(f"Loading cached {split} data...")
            X = np.load(cache_X)
            y = np.load(cache_y) if os.path.exists(cache_y) else None
            meta = pd.read_parquet(cache_meta)
            return X, y, meta

        print(f"Processing {split} data from scratch...")

        # 1. Load Metadata
        if split == "train":
            df_meta = pd.read_csv(Config.META_TRAIN)
            tracking_path = Config.TRAIN_TRACKING
            helmets_path = Config.TRAIN_HELMETS
        elif split == "validation":
            df_meta = pd.read_csv(Config.META_VAL)
            tracking_path = Config.TRAIN_TRACKING
            helmets_path = Config.TRAIN_HELMETS
        else:
            df_meta = pd.read_csv(Config.META_TEST)
            tracking_path = Config.TEST_TRACKING
            helmets_path = Config.TEST_HELMETS

        if debug:
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Load Raw Data (Filtered by relevant plays)
        relevant_plays = df_meta["game_play"].unique()

        df_tracking = pd.read_csv(tracking_path)
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        df_helmets = pd.read_csv(helmets_path)
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # 3. Process
        df_track_proc = self.process_tracking(df_tracking)
        df_vis_proc = self.process_visuals(df_helmets)
        df_merged = self.merge_and_impute(df_meta, df_track_proc, df_vis_proc)

        # 4. Preprocess
        fit_scaler = split == "train"
        X, y = self.preprocess(df_merged, fit=fit_scaler)

        # 5. Cache
        np.save(cache_X, X)
        if y is not None:
            np.save(cache_y, y)

        # Save lightweight metadata for inference/validation mapping
        meta_cols = ["contact_id", "game_play", "step"]
        if "contact" in df_merged.columns:
            meta_cols.append("contact")
        df_merged[meta_cols].to_parquet(cache_meta)

        return X, y, df_merged[meta_cols]
