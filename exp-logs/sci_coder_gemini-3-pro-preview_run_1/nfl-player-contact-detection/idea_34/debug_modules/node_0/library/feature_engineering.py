import pandas as pd
import numpy as np
import os
import logging
import gc
from library import config, utils


class ReferenceAnchoredFeatures:
    def __init__(self):
        self.window_size = config.WINDOW_SIZE
        self.gating_threshold = config.GATING_THRESHOLD
        self.sentinel_value = config.GROUND_DISTANCE_SENTINEL
        self.working_dir = config.WORKING_DIR
        self.lags = range(-self.window_size, self.window_size + 1)

    def _load_tracking(self, is_test=False):
        path = config.TEST_TRACKING_PATH if is_test else config.TRAIN_TRACKING_PATH
        logging.info(f"Loading tracking data from {path}")
        df_trk = pd.read_csv(path)

        # Standardize columns and types
        df_trk["nfl_player_id"] = df_trk["nfl_player_id"].astype(str)
        df_trk["game_play"] = df_trk["game_play"].astype(str)
        df_trk["step"] = df_trk["step"].astype(int)

        # Convert angles to radians and unit vectors
        # NFL Tracking: 0 is North (Y-axis), increasing Clockwise.
        # x = sin(theta), y = cos(theta)
        for col in ["orientation", "direction"]:
            rads = np.deg2rad(df_trk[col].fillna(0))
            df_trk[f"{col}_x"] = np.sin(rads)
            df_trk[f"{col}_y"] = np.cos(rads)

        return df_trk

    def _get_basis_vectors(self, df, mode="pp"):
        """
        Calculates basis vectors at lag 0 (t=0).
        df must contain P1 and P2 (if pp) coordinates/velocities at lag 0.
        """
        # P-P Basis: Unit vector from P2 to P1
        if mode == "pp":
            dx = df["x_position_p1_lag0"] - df["x_position_p2_lag0"]
            dy = df["y_position_p1_lag0"] - df["y_position_p2_lag0"]
            norm = np.sqrt(dx**2 + dy**2)

            # Handle zero distance (rare) - fallback to Y-axis
            mask = norm < 1e-6
            norm = np.where(mask, 1.0, norm)
            dx = np.where(mask, 0.0, dx)
            dy = np.where(mask, 1.0, dy)

            u_long_x = dx / norm
            u_long_y = dy / norm

        # P-G Basis: Unit vector of P1 velocity
        else:
            # Use velocity vectors derived from speed and direction
            vx = df["speed_p1_lag0"] * df["direction_x_p1_lag0"]
            vy = df["speed_p1_lag0"] * df["direction_y_p1_lag0"]
            norm = np.sqrt(vx**2 + vy**2)

            # Handle zero speed - fallback to orientation or Y-axis
            mask = norm < 1e-6

            # Try orientation if speed is 0
            fallback_x = df["orientation_x_p1_lag0"]
            fallback_y = df["orientation_y_p1_lag0"]

            # If orientation is also missing/zero, use Y-axis
            mask2 = (fallback_x**2 + fallback_y**2) < 1e-6

            u_long_x = np.where(mask, fallback_x, vx / np.where(mask, 1.0, norm))
            u_long_y = np.where(mask, fallback_y, vy / np.where(mask, 1.0, norm))

            u_long_x = np.where(mask & mask2, 0.0, u_long_x)
            u_long_y = np.where(mask & mask2, 1.0, u_long_y)

        # Transverse vector: Rotate +90 degrees (x, y) -> (-y, x)
        u_trans_x = -u_long_y
        u_trans_y = u_long_x

        return u_long_x, u_long_y, u_trans_x, u_trans_y

    def _project(self, x, y, u_long_x, u_long_y, u_trans_x, u_trans_y):
        long = x * u_long_x + y * u_long_y
        trans = x * u_trans_x + y * u_trans_y
        return long, trans

    def generate_features(self, df_meta, split="train", load_cached_data=True):
        cache_path = os.path.join(self.working_dir, f"features_{split}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            logging.info(f"Loading cached features from {cache_path}")
            return pd.read_parquet(cache_path)

        logging.info(f"Generating features for {split} set...")

        # Load Tracking
        is_test = split == "test"
        df_trk = self._load_tracking(is_test)

        # Prepare Metadata
        df_meta = df_meta.copy()
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_meta["step"] = df_meta["step"].astype(int)

        # Split into Player-Player and Player-Ground
        mask_g = df_meta["nfl_player_id_2"] == "G"
        df_pg = df_meta[mask_g].copy()
        df_pp = df_meta[~mask_g].copy()

        # Define columns needed from tracking
        cols_needed = [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation_x",
            "orientation_y",
            "direction_x",
            "direction_y",
        ]
        trk_full = df_trk[["game_play", "step", "nfl_player_id"] + cols_needed]

        # ==========================================
        # 1. Player-Player Processing (with Gating)
        # ==========================================
        if not df_pp.empty:
            logging.info("Processing Player-Player interactions...")

            # --- Gating Phase ---
            # Check min distance over the window to filter candidates
            df_pp["min_dist"] = np.inf
            pos_cols = ["x_position", "y_position"]
            trk_pos = df_trk[["game_play", "step", "nfl_player_id"] + pos_cols]

            for lag in self.lags:
                df_pp["step_lag"] = df_pp["step"] + lag

                # Merge P1 Position
                temp = df_pp[["game_play", "step_lag", "nfl_player_id_1"]].merge(
                    trk_pos,
                    left_on=["game_play", "step_lag", "nfl_player_id_1"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                )
                x1, y1 = temp["x_position"], temp["y_position"]

                # Merge P2 Position
                temp = df_pp[["game_play", "step_lag", "nfl_player_id_2"]].merge(
                    trk_pos,
                    left_on=["game_play", "step_lag", "nfl_player_id_2"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                )
                x2, y2 = temp["x_position"], temp["y_position"]

                d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                df_pp["min_dist"] = np.minimum(df_pp["min_dist"], d.fillna(np.inf))

            # Apply Gating
            logging.info(f"PP Rows before gating: {len(df_pp)}")
            df_pp = df_pp[df_pp["min_dist"] < self.gating_threshold].copy()
            logging.info(f"PP Rows after gating: {len(df_pp)}")
            df_pp.drop(columns=["min_dist", "step_lag"], inplace=True, errors="ignore")

            # --- Feature Phase ---

            # Helper to merge specific lag
            def merge_lag(df, lag, suffix_lag_str):
                df["step_lag"] = df["step"] + lag
                # P1
                df = df.merge(
                    trk_full,
                    left_on=["game_play", "step_lag", "nfl_player_id_1"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                    suffixes=("", "_tmp"),
                ).drop(
                    columns=["nfl_player_id", "step_lag", "game_play_tmp", "step_tmp"],
                    errors="ignore",
                )
                rename_p1 = {c: f"{c}_p1_{suffix_lag_str}" for c in cols_needed}
                df.rename(columns=rename_p1, inplace=True)

                # P2
                df = df.merge(
                    trk_full,
                    left_on=["game_play", "step_lag", "nfl_player_id_2"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                    suffixes=("", "_tmp"),
                ).drop(
                    columns=["nfl_player_id", "step_lag", "game_play_tmp", "step_tmp"],
                    errors="ignore",
                )
                rename_p2 = {c: f"{c}_p2_{suffix_lag_str}" for c in cols_needed}
                df.rename(columns=rename_p2, inplace=True)
                return df

            # Merge Lag 0 first to get basis
            df_pp = merge_lag(df_pp, 0, "lag0")
            ux, uy, vx, vy = self._get_basis_vectors(df_pp, mode="pp")

            # Iterate all lags and project
            for lag in self.lags:
                lag_str = f"lag{lag}"
                if lag != 0:
                    df_pp = merge_lag(df_pp, lag, lag_str)

                # Project P1 Velocity
                v1_x = df_pp[f"speed_p1_{lag_str}"] * df_pp[f"direction_x_p1_{lag_str}"]
                v1_y = df_pp[f"speed_p1_{lag_str}"] * df_pp[f"direction_y_p1_{lag_str}"]
                df_pp[f"p1_v_long_{lag_str}"], df_pp[f"p1_v_trans_{lag_str}"] = (
                    self._project(v1_x, v1_y, ux, uy, vx, vy)
                )

                # Project P1 Orientation
                o1_x = df_pp[f"orientation_x_p1_{lag_str}"]
                o1_y = df_pp[f"orientation_y_p1_{lag_str}"]
                df_pp[f"p1_o_long_{lag_str}"], df_pp[f"p1_o_trans_{lag_str}"] = (
                    self._project(o1_x, o1_y, ux, uy, vx, vy)
                )

                # Project P2 Velocity
                v2_x = df_pp[f"speed_p2_{lag_str}"] * df_pp[f"direction_x_p2_{lag_str}"]
                v2_y = df_pp[f"speed_p2_{lag_str}"] * df_pp[f"direction_y_p2_{lag_str}"]
                df_pp[f"p2_v_long_{lag_str}"], df_pp[f"p2_v_trans_{lag_str}"] = (
                    self._project(v2_x, v2_y, ux, uy, vx, vy)
                )

                # Project P2 Orientation
                o2_x = df_pp[f"orientation_x_p2_{lag_str}"]
                o2_y = df_pp[f"orientation_y_p2_{lag_str}"]
                df_pp[f"p2_o_long_{lag_str}"], df_pp[f"p2_o_trans_{lag_str}"] = (
                    self._project(o2_x, o2_y, ux, uy, vx, vy)
                )

                # Distance
                dx = (
                    df_pp[f"x_position_p1_{lag_str}"]
                    - df_pp[f"x_position_p2_{lag_str}"]
                )
                dy = (
                    df_pp[f"y_position_p1_{lag_str}"]
                    - df_pp[f"y_position_p2_{lag_str}"]
                )
                df_pp[f"dist_{lag_str}"] = np.sqrt(dx**2 + dy**2)

                # Scalars
                df_pp[f"accel_p1_{lag_str}"] = df_pp[f"acceleration_p1_{lag_str}"]
                df_pp[f"accel_p2_{lag_str}"] = df_pp[f"acceleration_p2_{lag_str}"]

            # Compute Acceleration from Velocity Diffs (Finite Difference)
            for lag in self.lags:
                lag_str = f"lag{lag}"
                prev_lag = lag - 1
                prev_str = f"lag{prev_lag}"

                if prev_lag >= -self.window_size:
                    df_pp[f"p1_a_long_{lag_str}"] = (
                        df_pp[f"p1_v_long_{lag_str}"] - df_pp[f"p1_v_long_{prev_str}"]
                    ) / 0.1
                    df_pp[f"p1_a_trans_{lag_str}"] = (
                        df_pp[f"p1_v_trans_{lag_str}"] - df_pp[f"p1_v_trans_{prev_str}"]
                    ) / 0.1
                    df_pp[f"p2_a_long_{lag_str}"] = (
                        df_pp[f"p2_v_long_{lag_str}"] - df_pp[f"p2_v_long_{prev_str}"]
                    ) / 0.1
                    df_pp[f"p2_a_trans_{lag_str}"] = (
                        df_pp[f"p2_v_trans_{lag_str}"] - df_pp[f"p2_v_trans_{prev_str}"]
                    ) / 0.1
                else:
                    for c in ["p1_a_long", "p1_a_trans", "p2_a_long", "p2_a_trans"]:
                        df_pp[f"{c}_{lag_str}"] = 0.0

        # ==========================================
        # 2. Player-Ground Processing
        # ==========================================
        if not df_pg.empty:
            logging.info("Processing Player-Ground interactions...")

            def merge_lag_p1(df, lag, suffix_lag_str):
                df["step_lag"] = df["step"] + lag
                df = df.merge(
                    trk_full,
                    left_on=["game_play", "step_lag", "nfl_player_id_1"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                    suffixes=("", "_tmp"),
                ).drop(
                    columns=["nfl_player_id", "step_lag", "game_play_tmp", "step_tmp"],
                    errors="ignore",
                )
                rename_p1 = {c: f"{c}_p1_{suffix_lag_str}" for c in cols_needed}
                df.rename(columns=rename_p1, inplace=True)
                return df

            # Merge Lag 0 for Basis
            df_pg = merge_lag_p1(df_pg, 0, "lag0")
            ux, uy, vx, vy = self._get_basis_vectors(df_pg, mode="pg")

            for lag in self.lags:
                lag_str = f"lag{lag}"
                if lag != 0:
                    df_pg = merge_lag_p1(df_pg, lag, lag_str)

                # Project P1
                v1_x = df_pg[f"speed_p1_{lag_str}"] * df_pg[f"direction_x_p1_{lag_str}"]
                v1_y = df_pg[f"speed_p1_{lag_str}"] * df_pg[f"direction_y_p1_{lag_str}"]
                df_pg[f"p1_v_long_{lag_str}"], df_pg[f"p1_v_trans_{lag_str}"] = (
                    self._project(v1_x, v1_y, ux, uy, vx, vy)
                )

                o1_x = df_pg[f"orientation_x_p1_{lag_str}"]
                o1_y = df_pg[f"orientation_y_p1_{lag_str}"]
                df_pg[f"p1_o_long_{lag_str}"], df_pg[f"p1_o_trans_{lag_str}"] = (
                    self._project(o1_x, o1_y, ux, uy, vx, vy)
                )

                # P2 Features (Ground) -> Sentinel/Zero
                df_pg[f"dist_{lag_str}"] = self.sentinel_value
                for feat in [
                    "p2_v_long",
                    "p2_v_trans",
                    "p2_o_long",
                    "p2_o_trans",
                    "speed_p2",
                    "accel_p2",
                ]:
                    df_pg[f"{feat}_{lag_str}"] = 0.0

                df_pg[f"accel_p1_{lag_str}"] = df_pg[f"acceleration_p1_{lag_str}"]

            # Calc Accel P1 via diff
            for lag in self.lags:
                lag_str = f"lag{lag}"
                prev_lag = lag - 1
                prev_str = f"lag{prev_lag}"

                if prev_lag >= -self.window_size:
                    df_pg[f"p1_a_long_{lag_str}"] = (
                        df_pg[f"p1_v_long_{lag_str}"] - df_pg[f"p1_v_long_{prev_str}"]
                    ) / 0.1
                    df_pg[f"p1_a_trans_{lag_str}"] = (
                        df_pg[f"p1_v_trans_{lag_str}"] - df_pg[f"p1_v_trans_{prev_str}"]
                    ) / 0.1
                else:
                    df_pg[f"p1_a_long_{lag_str}"] = 0.0
                    df_pg[f"p1_a_trans_{lag_str}"] = 0.0

                df_pg[f"p2_a_long_{lag_str}"] = 0.0
                df_pg[f"p2_a_trans_{lag_str}"] = 0.0

        # ==========================================
        # 3. Combine and Finalize
        # ==========================================
        df_final = pd.concat([df_pp, df_pg], axis=0, ignore_index=True)

        # Select Columns
        keep_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        if "contact" in df_final.columns:
            keep_cols.append("contact")

        feature_cols = config.FEATURE_COLUMNS
        # Ensure all feature cols exist (fill NaN with 0)
        for c in feature_cols:
            if c not in df_final.columns:
                df_final[c] = 0.0

        df_final = df_final[keep_cols + feature_cols]
        df_final = df_final.fillna(0.0)

        # Reduce Memory
        df_final = utils.reduce_mem_usage(df_final)

        # Save Cache
        logging.info(f"Saving features to {cache_path}")
        df_final.to_parquet(cache_path)

        return df_final
