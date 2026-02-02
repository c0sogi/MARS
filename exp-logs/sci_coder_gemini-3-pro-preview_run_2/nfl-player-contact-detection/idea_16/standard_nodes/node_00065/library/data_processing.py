import os
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class DataProcessor:
    """
    Handles data loading, ego-centric feature engineering, caching, and scaling
    for the Ego-Centric Deep Cross Network (EC-DCN).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        seed_everything(Config.SEED)

    def _load_tracking_data(self, path, game_plays=None):
        """
        Loads tracking data and filters by relevant game_plays to save memory.
        Creates shifted lag features for the temporal window.
        """
        df = pd.read_csv(path, usecols=Config.TRACKING_COLS)

        if game_plays is not None:
            df = df[df["game_play"].isin(game_plays)].copy()

        # Sort for shifting
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Fill NaNs in orientation/direction with 0
        df[["orientation", "direction"]] = df[["orientation", "direction"]].fillna(0)

        # Create wide-format tracking data with lags
        # We group by player and shift
        # Since we need t-W to t+W, we create columns for each lag

        # Pivot is too slow/memory intensive for full lags.
        # Instead, we will perform multiple merges or efficient shifting.
        # Given the memory constraint (220GB), we can create a dictionary of DataFrames for each lag.

        tracking_lags = {}
        grouped = df.groupby(["game_play", "nfl_player_id"])

        # Features to shift
        shift_cols = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
        ]

        for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            # shift(-lag) because if lag is -5 (past), we want the value from 5 steps ago.
            # pandas shift(k) shifts data down by k.
            # If we want data at t-5, we look at the row 5 steps up? No.
            # If we are at index t, we want value from index t-5.
            # df.shift(5) puts value from t-5 into row t. Correct.
            shifted = grouped[shift_cols].shift(lag)

            # Rename columns
            shifted.columns = [f"{c}_lag{lag}" for c in shift_cols]
            tracking_lags[lag] = shifted

        # Concatenate all lags horizontally
        # Note: The index is preserved, so this aligns perfectly
        df_wide = pd.concat(
            [df[["game_play", "step", "nfl_player_id"]]] + list(tracking_lags.values()),
            axis=1,
        )

        # Drop rows where window is incomplete (NaNs due to shift)
        # For training, we can drop. For test, we might need to fill?
        # The prompt implies strict window. We usually drop incomplete windows or fill.
        # Given the large dataset, dropping edges is safer for data quality.
        # df_wide = df_wide.dropna()

        # FIX: Impute missing edge values instead of dropping to prevent sample mismatch
        # This ensures all labeled timestamps (even at start/end of play) have features.
        # We group by player to avoid bleeding data across different tracks.
        meta_cols = ["game_play", "step", "nfl_player_id"]
        feature_cols = [c for c in df_wide.columns if c not in meta_cols]

        # Forward and Backward fill within each player's track to handle window edges
        # Groupby preserves index, so assignment aligns correctly
        filled = (
            df_wide.groupby(["game_play", "nfl_player_id"])[feature_cols]
            .ffill()
            .bfill()
        )
        df_wide[feature_cols] = filled

        # Fill any remaining NaNs (e.g. tracks shorter than window) with 0
        df_wide = df_wide.fillna(0)

        return df_wide

    def _compute_features(self, df_merged):
        """
        Vectorized computation of Field-Centric features.
        Replaces noisy ego-centric transformations with robust relative kinematics.
        Cite Lesson 00064: Robustness of Invariant Features vs. Noisy Reference Frames.
        """
        feature_arrays = []

        for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
            suffix = f"_lag{lag}"

            # 1. Extract Raw Attributes
            s1 = df_merged[f"speed{suffix}_1"]
            a1 = df_merged[f"acceleration{suffix}_1"]
            s2 = df_merged[f"speed{suffix}_2"]
            a2 = df_merged[f"acceleration{suffix}_2"]

            # 2. Compute Velocity Vectors (0 deg = North/Y, 90 deg = East/X)
            # vx = speed * sin(dir), vy = speed * cos(dir)
            dir1 = np.deg2rad(df_merged[f"direction{suffix}_1"])
            vx1 = s1 * np.sin(dir1)
            vy1 = s1 * np.cos(dir1)

            dir2 = np.deg2rad(df_merged[f"direction{suffix}_2"])
            vx2 = s2 * np.sin(dir2)
            vy2 = s2 * np.cos(dir2)

            # 3. Relative Position
            dx = df_merged[f"x_position{suffix}_2"] - df_merged[f"x_position{suffix}_1"]
            dy = df_merged[f"y_position{suffix}_2"] - df_merged[f"y_position{suffix}_1"]

            # 4. Geometric Invariants
            dist = np.sqrt(dx**2 + dy**2)
            log_dist = np.log1p(dist)  # Cite Lesson 00005

            # 5. Closing Speed
            # -(v_rel . p_rel) / |p_rel|
            dvx = vx2 - vx1
            dvy = vy2 - vy1
            dot = dvx * dx + dvy * dy
            # Cite Lesson 00007: Clamped denominator for stability
            closing_speed = -(dot) / np.maximum(dist, 1e-6)

            # 6. Meta
            is_ground = df_merged["is_ground"]

            # Stack features
            # Order must match Config.STEP_FEATURES
            step_feats = np.stack(
                [
                    dx.values,
                    dy.values,
                    vx1.values,
                    vy1.values,
                    vx2.values,
                    vy2.values,
                    log_dist.values,
                    closing_speed.values,
                    s1.values,
                    a1.values,
                    s2.values,
                    a2.values,
                    is_ground.values,
                ],
                axis=1,
            )

            feature_arrays.append(step_feats)

        # Concatenate all steps to form the wide vector
        # Shape: (N_samples, Input_Dim)
        X = np.hstack(feature_arrays)
        return X.astype(np.float32)

    def _process_split(self, meta_df, tracking_df, is_train=True):
        """
        Merges metadata with tracking and computes features.
        Handles Player-Player and Player-Ground cases separately.
        """
        # Prepare Metadata
        # Ensure nfl_player_id_1 is int
        meta_df["nfl_player_id_1"] = pd.to_numeric(
            meta_df["nfl_player_id_1"], errors="coerce"
        ).astype(int)

        # Split into Ground and Player-Player
        is_ground = meta_df["nfl_player_id_2"] == "G"
        df_pg = meta_df[is_ground].copy()
        df_pp = meta_df[~is_ground].copy()

        # --- Process Player-Player ---
        if not df_pp.empty:
            df_pp["nfl_player_id_2"] = pd.to_numeric(df_pp["nfl_player_id_2"]).astype(
                int
            )

            # Merge P1
            df_pp = df_pp.merge(
                tracking_df,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="inner",
                suffixes=("", "_drop"),
            ).drop(columns=["nfl_player_id"])

            # Merge P2
            df_pp = df_pp.merge(
                tracking_df,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="inner",
                suffixes=("_1", "_2"),
            ).drop(columns=["nfl_player_id"])

            df_pp["is_ground"] = 0.0

        # --- Process Player-Ground ---
        if not df_pg.empty:
            # Merge P1 only
            df_pg = df_pg.merge(
                tracking_df,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="inner",
            ).drop(columns=["nfl_player_id"])

            # Rename P1 columns to _1 suffix
            # tracking_df columns are raw names. We need to map them to _1
            # The merge kept them as is. Let's rename.
            cols_map = {
                c: f"{c}_1"
                for c in tracking_df.columns
                if c not in ["game_play", "step"]
            }
            df_pg = df_pg.rename(columns=cols_map)

            # Impute P2 columns
            # Pos_2 = Pos_1 (so dist=0)
            # Vel/Accel_2 = 0
            # Orientation_2 = 0
            for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
                suffix = f"_lag{lag}"
                df_pg[f"x_position{suffix}_2"] = df_pg[f"x_position{suffix}_1"]
                df_pg[f"y_position{suffix}_2"] = df_pg[f"y_position{suffix}_1"]
                df_pg[f"speed{suffix}_2"] = 0.0
                df_pg[f"acceleration{suffix}_2"] = 0.0
                df_pg[f"orientation{suffix}_2"] = 0.0
                df_pg[f"direction{suffix}_2"] = 0.0

            df_pg["is_ground"] = 1.0

        # Combine
        df_combined = pd.concat([df_pp, df_pg], axis=0, ignore_index=True)

        # Compute Features
        X = self._compute_features(df_combined)

        # Extract Targets/IDs
        if is_train:
            y = df_combined["contact"].values.astype(np.float32)
            return X, y
        else:
            ids = df_combined["contact_id"].values
            return X, ids

    def get_train_val_datasets(self, load_cached=True):
        """
        Main entry point for training data.
        Returns: X_train, y_train, X_val, y_val
        """
        # 1. Check Cache
        if (
            load_cached
            and os.path.exists(Config.CACHE_TRAIN_FEATURES)
            and os.path.exists(Config.CACHE_VAL_FEATURES)
        ):
            print("Loading cached training features...")
            train_data = pd.read_parquet(Config.CACHE_TRAIN_FEATURES)
            val_data = pd.read_parquet(Config.CACHE_VAL_FEATURES)

            # Load Scaler
            if os.path.exists(Config.SCALER_PATH):
                self.scaler = joblib.load(Config.SCALER_PATH)

            # Split X, y
            y_train = train_data["target"].values
            X_train = train_data.drop(columns=["target"]).values

            y_val = val_data["target"].values
            X_val = val_data.drop(columns=["target"]).values

            return X_train, y_train, X_val, y_val

        print("Processing training data from scratch...")

        # 2. Load Metadata
        df_train_meta = pd.read_csv(Config.TRAIN_LABELS_PATH)
        df_val_meta = pd.read_csv(Config.VAL_LABELS_PATH)

        # 3. Load Tracking (Only for relevant game_plays)
        train_gps = df_train_meta["game_play"].unique()
        val_gps = df_val_meta["game_play"].unique()
        all_gps = np.concatenate([train_gps, val_gps])

        print("Loading and shifting tracking data...")
        df_tracking = self._load_tracking_data(Config.TRAIN_TRACKING_PATH, all_gps)

        # 4. Process Splits
        print("Generating Train features...")
        X_train, y_train = self._process_split(
            df_train_meta, df_tracking, is_train=True
        )

        print("Generating Validation features...")
        X_val, y_val = self._process_split(df_val_meta, df_tracking, is_train=True)

        # 5. Scale Data
        print("Fitting Scaler...")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)

        # 6. Save Cache
        print("Saving cache...")
        # Save Scaler
        joblib.dump(self.scaler, Config.SCALER_PATH)

        # Save Parquet (Convert to DataFrame for easy IO)
        # Using float32 to save space
        col_names = [f"f_{i}" for i in range(X_train.shape[1])]

        df_train_save = pd.DataFrame(X_train, columns=col_names)
        df_train_save["target"] = y_train
        df_train_save.to_parquet(Config.CACHE_TRAIN_FEATURES)

        df_val_save = pd.DataFrame(X_val, columns=col_names)
        df_val_save["target"] = y_val
        df_val_save.to_parquet(Config.CACHE_VAL_FEATURES)

        return X_train, y_train, X_val, y_val

    def get_test_dataset(self, load_cached=True):
        """
        Main entry point for test data.
        Returns: X_test, ids
        """
        if load_cached and os.path.exists(Config.CACHE_TEST_FEATURES):
            print("Loading cached test features...")
            test_data = pd.read_parquet(Config.CACHE_TEST_FEATURES)
            ids = test_data["contact_id"].values
            X_test = test_data.drop(columns=["contact_id"]).values
            return X_test, ids

        print("Processing test data from scratch...")

        # Load Metadata
        df_test_meta = pd.read_csv(Config.TEST_META_PATH)

        # Load Tracking
        print("Loading test tracking...")
        df_tracking = self._load_tracking_data(
            Config.TEST_TRACKING_PATH, df_test_meta["game_play"].unique()
        )

        # Process
        print("Generating Test features...")
        X_test, ids = self._process_split(df_test_meta, df_tracking, is_train=False)

        # Scale
        # Load scaler if not in memory
        if not hasattr(self.scaler, "mean_"):
            if os.path.exists(Config.SCALER_PATH):
                self.scaler = joblib.load(Config.SCALER_PATH)
            else:
                raise ValueError("Scaler not found! Train the model first.")

        X_test = self.scaler.transform(X_test)

        # Save Cache
        col_names = [f"f_{i}" for i in range(X_test.shape[1])]
        df_test_save = pd.DataFrame(X_test, columns=col_names)
        df_test_save["contact_id"] = ids
        df_test_save.to_parquet(Config.CACHE_TEST_FEATURES)

        return X_test, ids
