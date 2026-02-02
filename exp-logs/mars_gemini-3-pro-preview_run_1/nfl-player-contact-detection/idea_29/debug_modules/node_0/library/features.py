import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import (
    get_file_hash,
    calculate_quadratic_min_distance,
    project_vector,
)


class FeatureGenerator:
    """
    Implements the Dual-Basis Kinematic-Spectral Anchored-Ensemble feature engineering pipeline.

    Key Components:
    1. Tracking Pre-processing: Computes instantaneous kinematic derivatives (Velocity, Accel, Jerk).
    2. Relaxed Quadratic Gating: Filters P-P pairs based on predicted minimum distance.
    3. Dual-Basis Projection:
       - P-P: Projects onto Collision Axis (Radial/Tangential).
       - P-G: Projects onto Motion Axis (Longitudinal/Lateral).
    """

    def __init__(self):
        self.tracking_cache = {}

    def load_tracking_data(self, path):
        """
        Loads tracking data with type optimization and pre-calculates Jerk (Spectral proxy).
        """
        if path in self.tracking_cache:
            return self.tracking_cache[path]

        print(f"Loading tracking data from {path}...")
        df_track = pd.read_csv(path)

        # Type optimization to save memory
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype("int32")
        df_track["step"] = df_track["step"].astype("int16")

        # Ensure sorted for lag calculations
        df_track = df_track.sort_values(["game_play", "nfl_player_id", "step"])

        # Group by player to ensure boundaries are respected
        g = df_track.groupby(["game_play", "nfl_player_id"])

        # --- Kinematic Derivative Calculation ---
        # We calculate finite differences to derive consistent vectors for V, A, and J (Jerk).
        # Time delta is 0.1s.
        dt = 0.1

        # 1. Velocity (vx, vy) from Position
        df_track["x_prev"] = g["x_position"].shift(1).fillna(df_track["x_position"])
        df_track["y_prev"] = g["y_position"].shift(1).fillna(df_track["y_position"])

        df_track["vx"] = (df_track["x_position"] - df_track["x_prev"]) / dt
        df_track["vy"] = (df_track["y_position"] - df_track["y_prev"]) / dt

        # 2. Acceleration (ax, ay) from Velocity
        df_track["vx_prev"] = g["vx"].shift(1).fillna(df_track["vx"])
        df_track["vy_prev"] = g["vy"].shift(1).fillna(df_track["vy"])

        df_track["ax"] = (df_track["vx"] - df_track["vx_prev"]) / dt
        df_track["ay"] = (df_track["vy"] - df_track["vy_prev"]) / dt

        # 3. Jerk (jx, jy) from Acceleration -> Proxy for "Spectral Shock"
        df_track["ax_prev"] = g["ax"].shift(1).fillna(df_track["ax"])
        df_track["ay_prev"] = g["ay"].shift(1).fillna(df_track["ay"])

        df_track["jx"] = (df_track["ax"] - df_track["ax_prev"]) / dt
        df_track["jy"] = (df_track["ay"] - df_track["ay_prev"]) / dt

        # Cleanup temporary columns
        drop_cols = ["x_prev", "y_prev", "vx_prev", "vy_prev", "ax_prev", "ay_prev"]
        df_track.drop(columns=drop_cols, inplace=True)

        self.tracking_cache[path] = df_track
        return df_track

    def generate_features(
        self, metadata_path, tracking_path, output_path, load_cached_data=True
    ):
        """
        Main pipeline execution: Load -> Merge -> Gate -> Feature Engineer -> Save.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(output_path):
            print(f"Loading cached features from {output_path}...")
            return pd.read_parquet(output_path)

        print(f"Generating features for {metadata_path}...")

        # 2. Load Data
        df_meta = pd.read_csv(metadata_path)
        df_track = self.load_tracking_data(tracking_path)

        # 3. Split into Player-Player and Player-Ground
        # P-G is identified by nfl_player_id_2 == "G"
        mask_pg = df_meta["nfl_player_id_2"] == "G"
        df_pg = df_meta[mask_pg].copy()
        df_pp = df_meta[~mask_pg].copy()

        # Ensure types for merge
        if not df_pp.empty:
            df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

        # ---------------------------------------------------------
        # 4. Process Player-Player (P-P)
        # ---------------------------------------------------------
        features_pp = pd.DataFrame()
        if not df_pp.empty:
            # Merge P1
            df_pp = (
                df_pp.merge(
                    df_track[
                        [
                            "game_play",
                            "step",
                            "nfl_player_id",
                            "x_position",
                            "y_position",
                            "vx",
                            "vy",
                            "ax",
                            "ay",
                            "jx",
                            "jy",
                        ]
                    ],
                    left_on=["game_play", "step", "nfl_player_id_1"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                )
                .rename(
                    columns={
                        "x_position": "x1",
                        "y_position": "y1",
                        "vx": "vx1",
                        "vy": "vy1",
                        "ax": "ax1",
                        "ay": "ay1",
                        "jx": "jx1",
                        "jy": "jy1",
                    }
                )
                .drop(columns=["nfl_player_id"])
            )

            # Merge P2
            df_pp = (
                df_pp.merge(
                    df_track[
                        [
                            "game_play",
                            "step",
                            "nfl_player_id",
                            "x_position",
                            "y_position",
                            "vx",
                            "vy",
                            "ax",
                            "ay",
                            "jx",
                            "jy",
                        ]
                    ],
                    left_on=["game_play", "step", "nfl_player_id_2"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                )
                .rename(
                    columns={
                        "x_position": "x2",
                        "y_position": "y2",
                        "vx": "vx2",
                        "vy": "vy2",
                        "ax": "ax2",
                        "ay": "ay2",
                        "jx": "jx2",
                        "jy": "jy2",
                    }
                )
                .drop(columns=["nfl_player_id"])
            )

            # Fill NaNs (missing tracking data) with 0.0
            feat_cols = [
                "x1",
                "y1",
                "vx1",
                "vy1",
                "ax1",
                "ay1",
                "jx1",
                "jy1",
                "x2",
                "y2",
                "vx2",
                "vy2",
                "ax2",
                "ay2",
                "jx2",
                "jy2",
            ]
            df_pp[feat_cols] = df_pp[feat_cols].fillna(0.0)

            # --- Gating (Relaxed Quadratic) ---
            min_dists = calculate_quadratic_min_distance(
                df_pp["x1"].values,
                df_pp["y1"].values,
                df_pp["vx1"].values,
                df_pp["vy1"].values,
                df_pp["ax1"].values,
                df_pp["ay1"].values,
                df_pp["x2"].values,
                df_pp["y2"].values,
                df_pp["vx2"].values,
                df_pp["vy2"].values,
                df_pp["ax2"].values,
                df_pp["ay2"].values,
                time_window=1.5,
            )

            df_pp["min_dist_pred"] = min_dists
            df_pp["gating_active"] = (
                df_pp["min_dist_pred"] < Config.GATING_THRESHOLD
            ).astype(int)

            # --- Dual-Basis Feature Engineering (Case A: P-P) ---
            # Basis: Collision Axis (P2 -> P1)
            dx = df_pp["x1"] - df_pp["x2"]
            dy = df_pp["y1"] - df_pp["y2"]
            dist = np.sqrt(dx**2 + dy**2)

            dist_safe = dist.replace(0, 1e-6)
            ux = dx / dist_safe
            uy = dy / dist_safe

            # Perpendicular Basis (Rotate 90 deg)
            ux_perp = -uy
            uy_perp = ux

            # Relative Vectors
            rvx = df_pp["vx1"] - df_pp["vx2"]
            rvy = df_pp["vy1"] - df_pp["vy2"]
            rax = df_pp["ax1"] - df_pp["ax2"]
            ray = df_pp["ay1"] - df_pp["ay2"]
            rjx = df_pp["jx1"] - df_pp["jx2"]
            rjy = df_pp["jy1"] - df_pp["jy2"]

            # Projections
            # Component 1 (Radial/Impact)
            df_pp["v_comp1"] = project_vector(rvx, rvy, ux, uy)
            df_pp["a_comp1"] = project_vector(rax, ray, ux, uy)
            df_pp["j_comp1"] = project_vector(rjx, rjy, ux, uy)

            # Component 2 (Tangential/Shear)
            df_pp["v_comp2"] = project_vector(rvx, rvy, ux_perp, uy_perp)
            df_pp["a_comp2"] = project_vector(rax, ray, ux_perp, uy_perp)
            df_pp["j_comp2"] = project_vector(rjx, rjy, ux_perp, uy_perp)

            df_pp["distance"] = dist
            features_pp = df_pp

        # ---------------------------------------------------------
        # 5. Process Player-Ground (P-G)
        # ---------------------------------------------------------
        features_pg = pd.DataFrame()
        if not df_pg.empty:
            # Merge P1 only
            df_pg = (
                df_pg.merge(
                    df_track[
                        [
                            "game_play",
                            "step",
                            "nfl_player_id",
                            "x_position",
                            "y_position",
                            "vx",
                            "vy",
                            "ax",
                            "ay",
                            "jx",
                            "jy",
                        ]
                    ],
                    left_on=["game_play", "step", "nfl_player_id_1"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                )
                .rename(
                    columns={
                        "x_position": "x1",
                        "y_position": "y1",
                        "vx": "vx1",
                        "vy": "vy1",
                        "ax": "ax1",
                        "ay": "ay1",
                        "jx": "jx1",
                        "jy": "jy1",
                    }
                )
                .drop(columns=["nfl_player_id"])
            )

            pg_feat_cols = ["x1", "y1", "vx1", "vy1", "ax1", "ay1", "jx1", "jy1"]
            df_pg[pg_feat_cols] = df_pg[pg_feat_cols].fillna(0.0)

            # --- Gating ---
            # P-G is always active
            df_pg["min_dist_pred"] = -1.0
            df_pg["gating_active"] = 1

            # --- Dual-Basis Feature Engineering (Case B: P-G) ---
            # Basis: Motion Axis (Velocity of P1)
            v_mag = np.sqrt(df_pg["vx1"] ** 2 + df_pg["vy1"] ** 2)
            v_mag_safe = v_mag.replace(0, 1e-6)

            ux = df_pg["vx1"] / v_mag_safe
            uy = df_pg["vy1"] / v_mag_safe

            # Perpendicular
            ux_perp = -uy
            uy_perp = ux

            # Projections (Absolute vectors of P1)
            # Component 1 (Longitudinal)
            df_pg["v_comp1"] = v_mag
            df_pg["a_comp1"] = project_vector(df_pg["ax1"], df_pg["ay1"], ux, uy)
            df_pg["j_comp1"] = project_vector(df_pg["jx1"], df_pg["jy1"], ux, uy)

            # Component 2 (Lateral)
            df_pg["v_comp2"] = 0.0
            df_pg["a_comp2"] = project_vector(
                df_pg["ax1"], df_pg["ay1"], ux_perp, uy_perp
            )
            df_pg["j_comp2"] = project_vector(
                df_pg["jx1"], df_pg["jy1"], ux_perp, uy_perp
            )

            # Sentinel Distance
            df_pg["distance"] = Config.DISTANCE_SENTINEL
            features_pg = df_pg

        # ---------------------------------------------------------
        # 6. Combine and Finalize
        # ---------------------------------------------------------
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "gating_active",
        ]
        if "contact" in df_meta.columns:
            meta_cols.append("contact")

        feature_cols = [
            "distance",
            "v_comp1",
            "v_comp2",
            "a_comp1",
            "a_comp2",
            "j_comp1",
            "j_comp2",
            "min_dist_pred",
        ]
        final_cols = meta_cols + feature_cols

        df_final = pd.concat([features_pp, features_pg], axis=0, ignore_index=True)
        df_final = df_final[final_cols]

        print(f"Saving {len(df_final)} rows to {output_path}...")
        df_final.to_parquet(output_path, index=False)

        del df_pp, df_pg, df_track, df_meta
        gc.collect()

        return df_final

    def process_train(self, load_cached_data=True):
        return self.generate_features(
            Config.TRAIN_METADATA_PATH,
            Config.TRAIN_TRACKING_PATH,
            Config.CACHE_TRAIN_FEATURES,
            load_cached_data,
        )

    def process_val(self, load_cached_data=True):
        return self.generate_features(
            Config.VAL_METADATA_PATH,
            Config.TRAIN_TRACKING_PATH,  # Val uses train tracking file
            Config.CACHE_VAL_FEATURES,
            load_cached_data,
        )

    def process_test(self, load_cached_data=True):
        return self.generate_features(
            Config.TEST_METADATA_PATH,
            Config.TEST_TRACKING_PATH,
            Config.CACHE_TEST_FEATURES,
            load_cached_data,
        )
