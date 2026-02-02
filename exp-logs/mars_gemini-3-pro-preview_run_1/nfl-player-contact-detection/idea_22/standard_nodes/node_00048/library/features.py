import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import memory_reduction, Timer


class FeatureEngineer:
    """
    Implements the Vector-Decomposed Physics feature engineering pipeline.
    Handles data loading, preprocessing, vector decomposition, and relaxed quadratic gating.
    """

    def __init__(self, metadata_path, tracking_path):
        self.metadata_path = metadata_path
        self.tracking_path = tracking_path
        # Cache directory from Config
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_raw_data(self):
        """Loads metadata and tracking data."""
        with Timer("Load Raw Data"):
            df_meta = pd.read_csv(self.metadata_path)
            df_track = pd.read_csv(self.tracking_path)

            # Ensure correct types for merge keys
            df_meta["game_play"] = df_meta["game_play"].astype(str)
            df_meta["step"] = df_meta["step"].astype(int)
            df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(int)

            # Handle 'G' in player 2 by keeping as object temporarily or converting valid ones
            # We will handle the split logic in merge

            df_track["game_play"] = df_track["game_play"].astype(str)
            df_track["step"] = df_track["step"].astype(int)
            df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(int)

        return df_meta, df_track

    def _preprocess_tracking(self, df_track):
        """
        Preprocesses tracking data:
        1. Sorts by play, player, step.
        2. Calculates 'jerk' (transient energy proxy).
        3. Converts angles to radians.
        """
        with Timer("Preprocess Tracking"):
            # Sort for temporal calculations
            df_track = df_track.sort_values(["game_play", "nfl_player_id", "step"])

            # Calculate Jerk (Transient Energy Proxy)
            # Group by game_play and player to ensure boundaries
            # We use a simple diff as a high-pass filter proxy
            df_track["jerk"] = (
                df_track.groupby(["game_play", "nfl_player_id"])["acceleration"]
                .diff()
                .fillna(0)
                .abs()
            )

            # Convert angles to radians for vector math
            # Orientation: 0 to 360, 0 is along Y axis?
            # Standard NFL tracking: 0 is Y-axis (short axis), increasing clockwise?
            # We just need consistency. Let's convert to standard math radians (0=East, CCW) if needed,
            # or just use as is for relative diffs.
            # For vector decomposition, we need (x,y) components.
            # Tracking data usually provides direction/orientation in degrees.
            # We will assume standard conversion: rad = deg * pi / 180.
            # Note: We rely on x_position/y_position for relative vectors,
            # but for player heading we use direction.
            df_track["direction_rad"] = np.deg2rad(df_track["direction"])
            df_track["orientation_rad"] = np.deg2rad(df_track["orientation"])

            # Calculate velocity components (if not explicitly trusted from speed/direction)
            # vx = speed * sin(direction) # NFL convention usually 0=Y, 90=X
            # Let's rely on x_position diffs implicitly via relative physics or use speed/direction.
            # Using speed/direction is safer for instantaneous velocity.
            # NFL: 0 is along Y (short axis), 90 is along X (long axis).
            # vx = speed * sin(theta)
            # vy = speed * cos(theta)
            df_track["v_x"] = df_track["speed"] * np.sin(df_track["direction_rad"])
            df_track["v_y"] = df_track["speed"] * np.cos(df_track["direction_rad"])

            # Acceleration components
            # We don't have accel direction, usually assume aligned with motion or use x/y double diffs.
            # Tracking data has 'x_position', 'y_position'.
            # We will approximate acceleration vector direction same as velocity for simplicity,
            # or just use scalar acceleration for energy.
            # For vector decomp, we need a_vector.
            # Let's assume acceleration acts primarily along the direction of motion (sa)
            # plus some centripetal.
            # Given data limitations, we'll project the scalar 'acceleration' onto the collision axis
            # weighted by the alignment of 'direction' and collision axis.
            # A better approach: Use the scalar features directly in the final set,
            # but for "Radial Acceleration", we project: a_radial = a * cos(theta_diff).

        return df_track

    def _merge_data(self, df_meta, df_track):
        """
        Merges metadata with tracking data for P1 and P2.
        Handles Ground ('G') interactions.
        """
        with Timer("Merge Tracking Data"):
            # Select relevant columns
            track_cols = [
                "game_play",
                "step",
                "nfl_player_id",
                "x_position",
                "y_position",
                "speed",
                "acceleration",
                "jerk",
                "v_x",
                "v_y",
                "direction_rad",
                "orientation_rad",
            ]
            df_t = df_track[track_cols]

            # --- Merge Player 1 ---
            df_merged = df_meta.merge(
                df_t,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id"])

            # Rename P1 columns
            rename_p1 = {
                c: f"{c}_p1"
                for c in track_cols
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_merged = df_merged.rename(columns=rename_p1)

            # --- Split Ground vs Player-Player ---
            mask_ground = df_merged["nfl_player_id_2"] == "G"
            df_ground = df_merged[mask_ground].copy()
            df_pp = df_merged[~mask_ground].copy()

            # --- Merge Player 2 (for PP) ---
            # Ensure ID is int
            df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

            df_pp = df_pp.merge(
                df_t,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=(
                    "",
                    "_p2",
                ),  # P1 cols already renamed, new cols get _p2 or nothing?
                # P1 cols are x_position_p1. New cols are x_position.
                # We need to rename new cols.
            ).drop(columns=["nfl_player_id"])

            # Rename P2 columns
            rename_p2 = {
                c: f"{c}_p2"
                for c in track_cols
                if c not in ["game_play", "step", "nfl_player_id"]
            }
            df_pp = df_pp.rename(columns=rename_p2)

            # --- Handle Ground P2 Columns ---
            # Fill P2 columns with 0 or NaN for Ground rows to align schema
            for col in rename_p2.values():
                df_ground[col] = 0.0

            # Recombine
            df_full = pd.concat([df_pp, df_ground], axis=0).reset_index(drop=True)

            # Ensure nfl_player_id_2 is string to handle 'G' mixed with ints for Parquet
            df_full["nfl_player_id_2"] = df_full["nfl_player_id_2"].astype(str)

            # Drop rows where P1 tracking is missing (critical failure)
            df_full = df_full.dropna(subset=["x_position_p1"])

            # For PP, drop if P2 tracking missing
            # (Ground rows have x_position_p2 as 0.0, so not NaN)
            # We need to be careful not to drop Ground rows.
            # The merge for PP was 'left', so P2 could be NaN.
            mask_pp_missing = (df_full["nfl_player_id_2"] != "G") & (
                df_full["x_position_p2"].isna()
            )
            df_full = df_full[~mask_pp_missing].reset_index(drop=True)

        return df_full

    def _compute_vector_physics(self, df):
        """
        Computes Vector-Decomposed Kinematics.
        """
        with Timer("Vector Decomposition"):
            # 1. Relative Positions
            # If Ground, x_p2 is 0. We treat Ground distance as Sentinel.
            # We calculate raw deltas first.
            dx = df["x_position_p1"] - df["x_position_p2"]
            dy = df["y_position_p1"] - df["y_position_p2"]

            # Euclidean Distance
            dist = np.sqrt(dx**2 + dy**2)

            # --- Apply Sentinel for Ground ---
            mask_ground = df["nfl_player_id_2"] == "G"

            # For Ground, set distance to Sentinel
            # We use loc to avoid SettingWithCopy
            df.loc[mask_ground, "distance"] = Config.GROUND_DISTANCE_SENTINEL
            df.loc[~mask_ground, "distance"] = dist[~mask_ground]

            # 2. Collision Axis (Unit Vector r_hat)
            # Points from P2 to P1: r = p1 - p2
            # Avoid div by zero
            safe_dist = dist.copy()
            safe_dist[safe_dist < 1e-6] = 1e-6

            rx_hat = dx / safe_dist
            ry_hat = dy / safe_dist

            # 3. Relative Velocity
            # v_rel = v1 - v2
            dvx = df["v_x_p1"] - df["v_x_p2"]
            dvy = df["v_y_p1"] - df["v_y_p2"]

            # 4. Vector Decomposition: Radial Velocity (Impact Speed)
            # v_radial = v_rel . r_hat
            # Negative = Closing, Positive = Opening
            v_radial = dvx * rx_hat + dvy * ry_hat

            # 5. Vector Decomposition: Tangential Velocity (Shear Speed)
            # v_tangential vector = v_rel - v_radial * r_hat
            # We just want magnitude or signed component against orthogonal vector
            # Orthogonal vector to r_hat (2D): (-ry, rx)
            # v_tan = v_rel . ortho_hat
            v_tangential = dvx * (-ry_hat) + dvy * (rx_hat)

            # 6. Acceleration Decomposition
            # We only have scalar acceleration 'acceleration_p1/p2' and direction.
            # We projected 'speed' to get v_x/v_y.
            # We can try to project acceleration similarly assuming it aligns with direction (approximation).
            # ax = acc * sin(dir), ay = acc * cos(dir)
            ax_p1 = df["acceleration_p1"] * np.sin(df["direction_rad_p1"])
            ay_p1 = df["acceleration_p1"] * np.cos(df["direction_rad_p1"])

            ax_p2 = df["acceleration_p2"] * np.sin(df["direction_rad_p2"])
            ay_p2 = df["acceleration_p2"] * np.cos(df["direction_rad_p2"])

            dax = ax_p1 - ax_p2
            day = ay_p1 - ay_p2

            a_radial = dax * rx_hat + day * ry_hat
            a_tangential = dax * (-ry_hat) + day * (rx_hat)

            # --- Assign Features ---
            # For Ground, these relative metrics are meaningless physically,
            # but we want the model to see P1's raw stats.
            # We map P1 stats to "Radial" for Ground (Impact with ground)

            # Initialize columns
            df["v_radial"] = v_radial
            df["v_tangential"] = v_tangential
            df["a_radial"] = a_radial
            df["a_tangential"] = a_tangential

            # Overwrite Ground rows
            # Radial Velocity -> P1 Speed (Impact intensity)
            df.loc[mask_ground, "v_radial"] = df.loc[mask_ground, "speed_p1"]
            # Tangential -> 0
            df.loc[mask_ground, "v_tangential"] = 0.0
            # Radial Acc -> P1 Accel
            df.loc[mask_ground, "a_radial"] = df.loc[mask_ground, "acceleration_p1"]
            df.loc[mask_ground, "a_tangential"] = 0.0

            # 7. Advanced Physics Primitives
            # Time to Collision: distance / closing_speed (-v_radial)
            # Only valid if closing (v_radial < 0)
            closing_speed = -df["v_radial"]
            ttc = df["distance"] / (closing_speed + 1e-6)
            # Clip and mask
            ttc[closing_speed <= 0] = Config.SENTINEL_VALUE  # Not closing
            ttc[mask_ground] = Config.SENTINEL_VALUE
            df["time_to_collision"] = ttc

            # Radial Acc Energy (Jerk Sum)
            df["radial_acc_energy"] = df["jerk_p1"] + df["jerk_p2"]

            # Orientation Diffs
            df["orientation_diff"] = np.abs(
                df["orientation_rad_p1"] - df["orientation_rad_p2"]
            )
            df["direction_diff"] = np.abs(
                df["direction_rad_p1"] - df["direction_rad_p2"]
            )

            # Fill NaNs created by diffs or sentinels
            df = df.fillna(0.0)

        return df

    def _apply_gating(self, df):
        """
        Applies Relaxed Quadratic Reachability Gating.
        Filters out pairs that are physically unlikely to contact.
        """
        with Timer("Relaxed Quadratic Gating"):
            # Logic:
            # 1. Keep all Ground interactions (distance sentinel -1.0 < 3.0)
            # 2. Keep all pairs with current distance < GATING_DISTANCE
            # 3. Keep pairs where predicted min distance < GATING_DISTANCE

            # Current distance check
            # Note: Ground distance is -1.0, so it passes "dist < 3.0" automatically.
            mask_keep = df["distance"] < Config.GATING_DISTANCE

            # Quadratic Lookahead for those currently far away
            # Only apply to Player-Player (dist > 0)
            mask_check = (~mask_keep) & (df["nfl_player_id_2"] != "G")

            if mask_check.sum() > 0:
                # Subset for calculation
                d = df.loc[mask_check, "distance"]
                v = df.loc[mask_check, "v_radial"]  # closing speed is -v
                a = df.loc[mask_check, "a_radial"]

                # We want min d(t) = d + v*t + 0.5*a*t^2 for t in [0, 1.0]
                # Vertex t* = -v / a
                # If a is small, linear projection: d(1) = d + v

                # Vectorized Vertex Calculation
                # Avoid div by zero
                a_safe = a.replace(0, 1e-6)
                t_vertex = -v / a_safe

                # Check distance at vertex if t_vertex in [0, 1.5] (relaxed window)
                valid_vertex = (t_vertex > 0) & (t_vertex < 1.5)

                d_vertex = d + v * t_vertex + 0.5 * a * (t_vertex**2)

                # Check distance at t=1.0 (Linear/End of window)
                d_end = d + v * 1.0 + 0.5 * a * (1.0**2)

                # Reachable if vertex dist < Threshold OR end dist < Threshold
                reachable = (valid_vertex & (d_vertex < Config.GATING_DISTANCE)) | (
                    d_end < Config.GATING_DISTANCE
                )

                # Update keep mask
                # We need to map back to original indices.
                # mask_keep is boolean series aligned with df.
                # reachable is aligned with d (subset).
                # We can use index alignment.
                mask_keep.loc[mask_check] = reachable

            # Filter
            original_len = len(df)
            df_gated = df[mask_keep].reset_index(drop=True)
            print(
                f"Gating: {original_len} -> {len(df_gated)} rows ({len(df_gated)/original_len:.2%})"
            )

        return df_gated

    def generate_features(self, load_cached_data=True):
        """
        Main pipeline execution.
        """
        # Cache Filename
        if self.metadata_path == Config.TRAIN_METADATA_PATH:
            cache_file = "train_features_gated_full.parquet"
        elif self.metadata_path == Config.VAL_METADATA_PATH:
            # Use 'full' suffix to indicate ungated validation set
            cache_file = "val_features_full.parquet"
        else:
            cache_file = "test_features_full.parquet"

        cache_path = os.path.join(self.cache_dir, cache_file)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            return pd.read_parquet(cache_path)

        # 2. Compute
        print("Generating features from scratch...")

        # Load
        df_meta, df_track = self._load_raw_data()

        # Preprocess Tracking
        df_track = self._preprocess_tracking(df_track)

        # Merge
        df = self._merge_data(df_meta, df_track)

        # Free memory
        del df_meta, df_track
        gc.collect()

        # Compute Physics
        df = self._compute_vector_physics(df)

        # Gating Logic
        # Only gate the training data. Validation and Test must remain full distribution.
        if self.metadata_path == Config.TRAIN_METADATA_PATH:
            df = self._apply_gating(df)
        else:
            print(
                "Skipping Gating for Validation/Test Data (Full Distribution Required)."
            )

        # Memory Reduction
        df = memory_reduction(df)

        # 3. Save Cache
        print(f"Saving features to {cache_path}...")
        df.to_parquet(cache_path, index=False)

        return df
