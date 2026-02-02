import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    load_parquet,
    save_parquet,
    reduce_mem_usage,
    vectorized_distance,
    vectorized_closing_speed,
)


class FeatureEngineer:
    """
    Implements the Kinematically-Projected Mining Ensemble (KP-ME) feature engineering pipeline.
    Handles data loading, physics derivation, temporal windowing, kinematic gating, and caching.
    """

    def __init__(self):
        self.tracking_cols = Config.TRACKING_COLS
        self.window_size = Config.WINDOW_SIZE
        self.gating_distance = Config.GATING_DISTANCE
        self.gating_ttc = Config.GATING_TTC
        self.ground_sentinel = Config.GROUND_DISTANCE_SENTINEL

    def _load_tracking(self, path):
        """
        Loads and preprocesses tracking data.
        Computes derived physics (components, jerk, angular jerk) and creates
        a wide format dataframe containing features for the +/- window_size.
        """
        # Load raw data
        df = pd.read_csv(path)

        # Filter to necessary columns
        df = df[self.tracking_cols].copy()

        # Sort to ensure correct shifting
        df = df.sort_values(["game_play", "nfl_player_id", "step"]).reset_index(
            drop=True
        )

        # ---------------------------------------------------------
        # 1. Physics Derivatives
        # ---------------------------------------------------------
        # Convert direction to radians for component calculation
        df["dir_rad"] = np.deg2rad(df["direction"])

        # Velocity components
        df["vx"] = df["speed"] * np.sin(df["dir_rad"])
        df["vy"] = df["speed"] * np.cos(df["dir_rad"])

        # Group by player to compute temporal derivatives
        grp = df.groupby(["game_play", "nfl_player_id"])

        # Jerk: Derivative of Acceleration
        df["jerk"] = grp["acceleration"].diff().fillna(0)

        # Angular Velocity & Jerk
        # Approx Angular Velocity = diff(orientation)
        df["ang_vel"] = grp["orientation"].diff().fillna(0)
        # Angular Jerk = diff(ang_vel)
        df["ang_jerk"] = grp["ang_vel"].diff().fillna(0)

        # ---------------------------------------------------------
        # 2. Temporal Window Flattening (Wide Format)
        # ---------------------------------------------------------
        # Features to retain and shift
        features_to_shift = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "vx",
            "vy",
            "jerk",
            "ang_jerk",
        ]

        # Base keys for alignment check
        keys = df[["game_play", "nfl_player_id", "step"]]

        # Start with current step (lag 0)
        final_wide = keys.copy()

        # We iterate through lags from -10 to +10
        lags = range(-self.window_size, self.window_size + 1)

        for k in lags:
            if k == 0:
                # No shift
                shifted = df[features_to_shift].copy()
                shifted.columns = [f"{c}_{k}" for c in shifted.columns]
                final_wide = pd.concat([final_wide, shifted], axis=1)
            else:
                # Shift logic:
                # We want data at step+k.
                # df.shift(-k) moves the row at i+k to i.
                shift_amount = -k

                shifted = df[features_to_shift].shift(shift_amount)

                # Verify alignment (game_play and player must match)
                shifted_keys = keys.shift(shift_amount)
                mask = (keys["game_play"] == shifted_keys["game_play"]) & (
                    keys["nfl_player_id"] == shifted_keys["nfl_player_id"]
                )

                # Mask invalid shifts (boundaries of plays)
                shifted[~mask] = np.nan

                shifted.columns = [f"{c}_{k}" for c in shifted.columns]
                final_wide = pd.concat([final_wide, shifted], axis=1)

        # Clean up
        del df
        gc.collect()

        return reduce_mem_usage(final_wide)

    def _process_dataset(self, metadata_path, tracking_path, apply_gating=True):
        """
        Core processing logic: Merge -> Interact -> Gate -> Clean.
        """
        # 1. Load Metadata
        df_meta = pd.read_csv(metadata_path)

        # 2. Load Wide Tracking Data
        df_track = self._load_tracking(tracking_path)

        # 3. Merge Player 1
        df_merged = df_meta.merge(
            df_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # Rename P1 columns
        track_feat_cols = [
            c
            for c in df_track.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_p1 = {c: f"{c}_p1" for c in track_feat_cols}
        df_merged = df_merged.rename(columns=rename_p1)
        df_merged = df_merged.drop(columns=["nfl_player_id"])

        # 4. Merge Player 2 (Handle Ground vs Player)
        mask_ground = df_merged["nfl_player_id_2"] == "G"
        df_ground = df_merged[mask_ground].copy()
        df_players = df_merged[~mask_ground].copy()

        # 4a. Player-Player Merge
        if not df_players.empty:
            df_players["nfl_player_id_2"] = df_players["nfl_player_id_2"].astype(int)
            df_players = df_players.merge(
                df_track,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            rename_p2 = {c: f"{c}_p2" for c in track_feat_cols}
            df_players = df_players.rename(columns=rename_p2)
            df_players = df_players.drop(columns=["nfl_player_id"])

        # 4b. Player-Ground Handling
        # Fill P2 features with 0.0 for Ground rows
        if not df_ground.empty:
            for c in track_feat_cols:
                df_ground[f"{c}_p2"] = 0.0

        # Recombine
        df_full = pd.concat([df_players, df_ground], axis=0).sort_index()
        del df_players, df_ground, df_track
        gc.collect()

        # 5. Compute Interaction Features
        lags = range(-self.window_size, self.window_size + 1)

        for k in lags:
            s1 = f"_{k}_p1"
            s2 = f"_{k}_p2"

            # Extract vectors
            x1, y1 = df_full[f"x_position{s1}"], df_full[f"y_position{s1}"]
            x2, y2 = df_full[f"x_position{s2}"], df_full[f"y_position{s2}"]
            vx1, vy1 = df_full[f"vx{s1}"], df_full[f"vy{s1}"]
            vx2, vy2 = df_full[f"vx{s2}"], df_full[f"vy{s2}"]

            # Distance
            dist_col = f"distance_{k}"
            df_full[dist_col] = vectorized_distance(x1, y1, x2, y2)

            # Apply Sentinel for Ground
            is_ground = df_full["nfl_player_id_2"] == "G"
            df_full.loc[is_ground, dist_col] = self.ground_sentinel

            # Speed Diff
            df_full[f"speed_diff_{k}"] = np.abs(
                df_full[f"speed{s1}"] - df_full[f"speed{s2}"]
            )

            # Interaction Primitives at t=0
            if k == 0:
                # Closing Speed
                cs = vectorized_closing_speed(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
                df_full["closing_speed"] = cs

                # Time-To-Collision (TTC)
                # Avoid div/0. TTC is valid only if closing_speed > 0
                df_full["ttc"] = df_full[dist_col] / (cs + 1e-6)

                # Kinetic Energy Proxy (Relative Speed Squared)
                rel_vx = vx1 - vx2
                rel_vy = vy1 - vy2
                df_full["ke_proxy"] = rel_vx**2 + rel_vy**2

        # 6. Kinematic Reachability Gating
        if apply_gating:
            print(f"Applying Kinematic Gating (Pre-filter rows: {len(df_full)})...")

            dist0 = df_full["distance_0"]
            ttc = df_full["ttc"]
            is_ground = df_full["nfl_player_id_2"] == "G"

            # Keep if:
            # 1. Is Ground Interaction (Always keep)
            # 2. Distance < Threshold (Close proximity)
            # 3. TTC < Threshold AND TTC > 0 (High velocity approach)

            cond_dist = dist0 < self.gating_distance
            cond_ttc = (ttc > 0) & (ttc < self.gating_ttc)

            keep_mask = is_ground | cond_dist | cond_ttc

            df_full = df_full[keep_mask].reset_index(drop=True)
            print(f"Gating Complete. Rows remaining: {len(df_full)}")

        # 7. Cleanup & Invariance
        # Drop absolute coordinates
        cols_to_drop = [
            c for c in df_full.columns if "x_position" in c or "y_position" in c
        ]
        df_full = df_full.drop(columns=cols_to_drop)

        return reduce_mem_usage(df_full)

    def create_train_features(self, load_cached_data=True):
        """
        Generates or loads features for the training set.
        Applies Kinematic Gating.
        """
        if load_cached_data and os.path.exists(Config.CACHE_TRAIN_FEATURES):
            print("Loading cached train features...")
            return load_parquet(Config.CACHE_TRAIN_FEATURES)

        print("Generating train features...")
        df = self._process_dataset(
            Config.TRAIN_METADATA_PATH, Config.TRAIN_TRACKING_PATH, apply_gating=True
        )
        save_parquet(df, Config.CACHE_TRAIN_FEATURES)
        return df

    def create_val_features(self, load_cached_data=True):
        """
        Generates or loads features for the validation set.
        Applies Kinematic Gating to match training distribution for mining.
        """
        if load_cached_data and os.path.exists(Config.CACHE_VAL_FEATURES):
            print("Loading cached val features...")
            return load_parquet(Config.CACHE_VAL_FEATURES)

        print("Generating val features...")
        # Val metadata is a subset of train labels, so it uses train tracking
        df = self._process_dataset(
            Config.VAL_METADATA_PATH, Config.TRAIN_TRACKING_PATH, apply_gating=True
        )
        save_parquet(df, Config.CACHE_VAL_FEATURES)
        return df

    def create_test_features(self, load_cached_data=True):
        """
        Generates or loads features for the test set.
        Does NOT apply Gating (inference on all candidates).
        """
        if load_cached_data and os.path.exists(Config.CACHE_TEST_FEATURES):
            print("Loading cached test features...")
            return load_parquet(Config.CACHE_TEST_FEATURES)

        print("Generating test features...")
        df = self._process_dataset(
            Config.TEST_METADATA_PATH, Config.TEST_TRACKING_PATH, apply_gating=False
        )
        save_parquet(df, Config.CACHE_TEST_FEATURES)
        return df
