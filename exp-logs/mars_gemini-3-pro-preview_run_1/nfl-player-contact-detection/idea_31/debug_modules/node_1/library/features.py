import pandas as pd
import numpy as np
import os
import gc
import logging
from library.config import PathConfig, FeatureConfig, GatingConfig
from library.utils import (
    load_dataframe,
    save_dataframe,
    reduce_mem_usage,
    setup_logging,
)

# Initialize logging
setup_logging()


class FeatureEngineer:
    """
    Implements Time-Domain Vector-Aligned Feature Engineering with Relaxed Quadratic Gating.
    """

    def __init__(self, mode="train"):
        """
        Args:
            mode (str): 'train', 'val', or 'test'. Determines data source and gating logic.
        """
        self.mode = mode
        self.is_train = mode in ["train", "val"]

    def _get_paths(self):
        """Returns metadata and tracking paths based on mode."""
        if self.mode == "train":
            return (
                PathConfig.TRAIN_METADATA,
                PathConfig.TRAIN_TRACKING,
                PathConfig.CACHE_TRAIN_FEATURES,
            )
        elif self.mode == "val":
            return (
                PathConfig.VAL_METADATA,
                PathConfig.TRAIN_TRACKING,
                PathConfig.CACHE_VAL_FEATURES,
            )
        elif self.mode == "test":
            return (
                PathConfig.TEST_METADATA,
                PathConfig.TEST_TRACKING,
                PathConfig.CACHE_TEST_FEATURES,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def load_data(self):
        """Loads and preprocesses metadata and tracking data."""
        meta_path, track_path, _ = self._get_paths()

        logging.info(f"Loading metadata from {meta_path}...")
        df_meta = pd.read_csv(meta_path)

        logging.info(f"Loading tracking data from {track_path}...")
        df_track = pd.read_csv(track_path)
        df_track = reduce_mem_usage(df_track)

        # Standardize tracking columns for convenience
        # NFL Data: dir=0 is North (Y), 90 is East (X)
        rename_dict = {
            "x_position": "x",
            "y_position": "y",
            "speed": "s",
            "acceleration": "a",
            "direction": "dir",
            "orientation": "o",
            "sa": "sa",  # Signed acceleration if available
        }
        df_track = df_track.rename(columns=rename_dict)

        # Ensure necessary columns exist
        req_cols = ["game_play", "step", "nfl_player_id", "x", "y", "s", "a", "dir"]
        for c in req_cols:
            if c not in df_track.columns:
                # If 'sa' is missing, we might need to handle it, but it's in the description.
                # 'sa' is not strictly required if we use 'a', but good for signed projection.
                pass

        return df_meta, df_track

    def apply_gating(self, df_meta, df_track):
        """
        Stage 0: Relaxed Quadratic Reachability Gating.
        Filters pairs where the minimum projected distance over the window > Threshold.
        """
        logging.info("Applying Relaxed Quadratic Reachability Gating...")

        # Separate Ground interactions (always keep)
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_ground = df_meta[mask_ground].copy()
        df_players = df_meta[~mask_ground].copy()

        if df_players.empty:
            return df_meta

        # Prepare for merge
        # We need kinematic state at t=0
        df_players["nfl_player_id_2"] = (
            df_players["nfl_player_id_2"].astype(float).astype(int)
        )

        # Helper to get vectors
        def get_kinematics(df_in, suffix):
            # Merge
            df_m = df_in.merge(
                df_track[
                    ["game_play", "step", "nfl_player_id", "x", "y", "s", "a", "dir"]
                ],
                left_on=["game_play", "step", f"nfl_player_id_{suffix}"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )

            # Convert polar to cartesian
            # NFL: 0=Y, 90=X -> x=sin(rad), y=cos(rad)
            rad = np.radians(df_m["dir"].fillna(0))
            s = df_m["s"].fillna(0)
            a = df_m["a"].fillna(0)

            vx = s * np.sin(rad)
            vy = s * np.cos(rad)

            # Approx acceleration vector using direction of motion
            ax = a * np.sin(rad)
            ay = a * np.cos(rad)

            return (
                df_m["x"].values,
                df_m["y"].values,
                vx.values,
                vy.values,
                ax.values,
                ay.values,
            )

        # Get P1 State
        x1, y1, vx1, vy1, ax1, ay1 = get_kinematics(df_players, "1")
        # Get P2 State
        x2, y2, vx2, vy2, ax2, ay2 = get_kinematics(df_players, "2")

        # Relative State
        rx = x1 - x2
        ry = y1 - y2
        rvx = vx1 - vx2
        rvy = vy1 - vy2
        rax = ax1 - ax2
        ray = ay1 - ay2

        # Quadratic Projection
        # Evaluate distance on a grid of time steps
        # Window: +/- 1.0s (10 steps)
        t_grid = (
            np.arange(-GatingConfig.PROJECTION_STEPS, GatingConfig.PROJECTION_STEPS + 1)
            * 0.1
        )

        # Vectorized grid search for min distance
        # Shape: (N_samples, N_timesteps)
        # We expand dimensions for broadcasting
        rx_t = rx[:, None] + rvx[:, None] * t_grid + 0.5 * rax[:, None] * (t_grid**2)
        ry_t = ry[:, None] + rvy[:, None] * t_grid + 0.5 * ray[:, None] * (t_grid**2)

        dist_sq_t = rx_t**2 + ry_t**2
        min_dist = np.sqrt(np.min(dist_sq_t, axis=1))

        # Filter
        keep_mask = min_dist < GatingConfig.REACHABILITY_THRESHOLD
        df_survivors = df_players[keep_mask]

        logging.info(
            f"Gating reduced {len(df_players)} pairs to {len(df_survivors)} "
            f"({len(df_survivors)/len(df_players):.2%})"
        )

        # Recombine
        df_final = pd.concat([df_survivors, df_ground], axis=0).reset_index(drop=True)
        return df_final

    def generate_features(self, df_meta, df_track):
        """
        Stage 1: Time-Domain Vector-Aligned Feature Engineering.
        """
        logging.info("Generating features with Vector Alignment...")

        # 1. Establish Basis at t=0
        # We need to handle Ground vs Players differently for basis

        # Helper to merge a specific lag
        def merge_at_lag(df_base, lag):
            # Create join key
            df_base["step_join"] = df_base["step"] + lag

            # Merge P1
            df_res = df_base.merge(
                df_track[
                    [
                        "game_play",
                        "step",
                        "nfl_player_id",
                        "x",
                        "y",
                        "s",
                        "a",
                        "dir",
                        "sa",
                    ]
                ],
                left_on=["game_play", "step_join", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p1"),
            ).drop(
                columns=["nfl_player_id", "step_join", "step_p1"]
            )  # keep original step

            # Rename P1 columns
            p1_cols = {c: f"{c}1_lag{lag}" for c in ["x", "y", "s", "a", "dir", "sa"]}
            df_res = df_res.rename(columns=p1_cols)

            # Prepare P2 merge
            # Handle 'G' by converting to numeric, G becomes NaN
            df_res["p2_join_id"] = pd.to_numeric(
                df_res["nfl_player_id_2"], errors="coerce"
            )

            # Merge P2
            df_res["step_join"] = df_res["step"] + lag
            df_res = df_res.merge(
                df_track[
                    [
                        "game_play",
                        "step",
                        "nfl_player_id",
                        "x",
                        "y",
                        "s",
                        "a",
                        "dir",
                        "sa",
                    ]
                ],
                left_on=["game_play", "step_join", "p2_join_id"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p2"),
            ).drop(columns=["nfl_player_id", "step_join", "p2_join_id", "step_p2"])

            # Rename P2 columns
            p2_cols = {c: f"{c}2_lag{lag}" for c in ["x", "y", "s", "a", "dir", "sa"]}
            df_res = df_res.rename(columns=p2_cols)

            # Fill NaNs for P2 (Ground or missing) with 0
            # This ensures vector subtraction works (P1 - 0 = P1)
            for col in p2_cols.values():
                df_res[col] = df_res[col].fillna(0)

            return df_res

        # --- Step A: Get t=0 data for Basis ---
        df_curr = merge_at_lag(df_meta, 0)

        # Calculate Relative Position at t=0
        rx = df_curr["x1_lag0"] - df_curr["x2_lag0"]
        ry = df_curr["y1_lag0"] - df_curr["y2_lag0"]
        dist = np.sqrt(rx**2 + ry**2)

        # Define Basis Vectors
        # Collision Axis (u): Unit vector from P2 to P1
        # Tangent Axis (t): Orthogonal (-uy, ux)

        # Handle Ground (P2='G')
        # For Ground, we align basis with Global X/Y to preserve absolute P1 dynamics
        mask_g = df_curr["nfl_player_id_2"] == "G"

        # Avoid div/0
        dist_safe = dist.copy()
        dist_safe[dist_safe < 1e-6] = 1e-6

        ux = rx / dist_safe
        uy = ry / dist_safe

        # Override for Ground: u = (1,0) -> X-axis
        ux[mask_g] = 1.0
        uy[mask_g] = 0.0

        # Tangent
        tx = -uy
        ty = ux

        # Initialize Output DataFrame
        out_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        if "contact" in df_curr.columns:
            out_cols.append("contact")

        df_out = df_curr[out_cols].copy()

        # Set Sentinel Distance for Ground
        final_dist = dist.copy()
        final_dist[mask_g] = FeatureConfig.GROUND_DISTANCE_SENTINEL
        df_out["dist_0"] = final_dist.astype(np.float32)

        # --- Step B: Loop Lags and Project ---
        lags = range(-FeatureConfig.WINDOW_SIZE, FeatureConfig.WINDOW_SIZE + 1)

        for lag in lags:
            # We already have lag 0 in df_curr, but to keep loop clean we can re-merge or optimize
            # Optimization: Use df_curr for lag 0
            if lag == 0:
                df_lag = df_curr
            else:
                df_lag = merge_at_lag(
                    df_meta[
                        ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]
                    ],
                    lag,
                )

            # 1. Calculate Cartesian Vectors
            def get_cartesian(s_col, dir_col, a_col, sa_col):
                rad = np.radians(df_lag[dir_col].fillna(0))
                s = df_lag[s_col].fillna(0)
                vx = s * np.sin(rad)
                vy = s * np.cos(rad)

                # Acceleration
                # Use 'sa' (signed accel) if available, else 'a' (magnitude)
                # If 'sa' exists, it's signed along direction.
                # If 'sa' is NaN, fallback to 'a'
                acc_mag = df_lag[sa_col].fillna(df_lag[a_col].fillna(0))
                ax = acc_mag * np.sin(rad)
                ay = acc_mag * np.cos(rad)
                return vx, vy, ax, ay

            vx1, vy1, ax1, ay1 = get_cartesian(
                f"s1_lag{lag}", f"dir1_lag{lag}", f"a1_lag{lag}", f"sa1_lag{lag}"
            )
            vx2, vy2, ax2, ay2 = get_cartesian(
                f"s2_lag{lag}", f"dir2_lag{lag}", f"a2_lag{lag}", f"sa2_lag{lag}"
            )

            # Relative Vectors
            rvx = vx1 - vx2
            rvy = vy1 - vy2
            rax = ax1 - ax2
            ray = ay1 - ay2

            # 2. Project onto Fixed Basis (ux, uy, tx, ty from t=0)
            # Radial (Collision) Component
            v_rad = rvx * ux + rvy * uy
            a_rad = rax * ux + ray * uy

            # Tangent (Shear) Component
            v_tan = rvx * tx + rvy * ty
            a_tan = rax * tx + ray * ty

            # 3. Store
            df_out[f"v_rad_{lag}"] = v_rad.astype(np.float32)
            df_out[f"v_tan_{lag}"] = v_tan.astype(np.float32)
            df_out[f"a_rad_{lag}"] = a_rad.astype(np.float32)
            df_out[f"a_tan_{lag}"] = a_tan.astype(np.float32)

            # Distance at lag k (optional, but good for trajectory)
            dx = df_lag[f"x1_lag{lag}"] - df_lag[f"x2_lag{lag}"]
            dy = df_lag[f"y1_lag{lag}"] - df_lag[f"y2_lag{lag}"]
            d_k = np.sqrt(dx**2 + dy**2)
            d_k[mask_g] = FeatureConfig.GROUND_DISTANCE_SENTINEL
            df_out[f"dist_{lag}"] = d_k.astype(np.float32)

        # --- Step C: Physics Primitives ---

        # Jerk: (a_rad_1 - a_rad_-1) / 0.2s (Centered difference at t=0)
        if FeatureConfig.USE_JERK:
            jerk = (df_out["a_rad_1"] - df_out["a_rad_-1"]) / 0.2
            df_out["jerk_rad_0"] = jerk.astype(np.float32)

        # TTC: dist_0 / -v_rad_0 (if closing)
        if FeatureConfig.USE_TTC:
            closing_speed = -df_out["v_rad_0"]
            dist = df_out["dist_0"]
            # Default TTC to 10.0s
            ttc = pd.Series(10.0, index=df_out.index)

            valid_ttc = (closing_speed > 0.1) & (dist > 0)
            ttc[valid_ttc] = dist[valid_ttc] / closing_speed[valid_ttc]

            # Cap TTC
            ttc = ttc.clip(upper=10.0)

            # Sentinel for Ground? Ground doesn't have TTC in same sense, but P1 has speed towards ground?
            # With our basis, v_rad for ground is Vx.
            # Let's leave it as is; tree models will handle the distribution.
            df_out["ttc_0"] = ttc.astype(np.float32)

        return reduce_mem_usage(df_out)

    def process(self, load_cached=True):
        """
        Main execution method.
        """
        _, _, cache_path = self._get_paths()

        # 1. Check Cache
        if load_cached and os.path.exists(cache_path):
            logging.info(f"Loading cached features from {cache_path}")
            return load_dataframe(cache_path)

        # 2. Load Data
        df_meta, df_track = self.load_data()

        # 3. Apply Gating (Train/Val only)
        if self.is_train:
            df_meta = self.apply_gating(df_meta, df_track)

        # 4. Generate Features
        df_features = self.generate_features(df_meta, df_track)

        # 5. Save Cache
        save_dataframe(df_features, cache_path)

        # Cleanup
        del df_meta, df_track
        gc.collect()

        return df_features


def create_features(mode="train", load_cached=True):
    """
    Wrapper function to instantiate and run the FeatureEngineer.
    """
    engineer = FeatureEngineer(mode=mode)
    return engineer.process(load_cached=load_cached)
