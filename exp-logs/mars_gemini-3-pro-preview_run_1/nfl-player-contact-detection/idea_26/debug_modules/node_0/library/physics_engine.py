import pandas as pd
import numpy as np
import os
import gc
import library.config as config
from library.utils import reduce_mem_usage


class FeatureManager:
    """
    Manages data loading, merging, kinematic gating, and vector-aligned feature engineering
    for the VASM-E strategy.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, split_name):
        """Generates the cache file path for a given split."""
        return os.path.join(self.cache_dir, f"features_{split_name}_vasm_e.parquet")

    def load_tracking_data(self, path):
        """
        Loads and preprocesses tracking data.
        Converts polar velocity/acceleration to Cartesian components.
        """
        print(f"Loading tracking data from {path}...")
        df = pd.read_csv(path)

        # Standardize direction (motion angle).
        # In NFL tracking data: 0 is usually along Y-axis (North), 90 along X-axis (East).
        # We convert to standard math radians for projection.
        df["dir_rad"] = np.radians(df["direction"])

        # Velocity Components
        df["v_x"] = df["speed"] * np.sin(df["dir_rad"])
        df["v_y"] = df["speed"] * np.cos(df["dir_rad"])

        # Acceleration Components
        # We approximate the acceleration vector direction using the motion direction.
        # While 'sa' (signed acceleration) exists, projecting total 'acceleration' magnitude
        # along the motion vector is a robust proxy for the kinematic state.
        df["a_x"] = df["acceleration"] * np.sin(df["dir_rad"])
        df["a_y"] = df["acceleration"] * np.cos(df["dir_rad"])

        # Select relevant columns
        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
            "v_x",
            "v_y",
            "a_x",
            "a_y",
        ]

        # Filter columns that exist (just in case)
        cols = [c for c in cols if c in df.columns]

        return df[cols]

    def merge_data(self, df_meta, df_track):
        """
        Merges metadata with tracking data for Player 1 and Player 2.
        Handles Ground (G) interactions.
        """
        print("Merging metadata with tracking data...")

        # Ensure join keys match types
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_track["game_play"] = df_track["game_play"].astype(str)

        # Split into Ground vs Player-Player for efficient merging
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_pp = df_meta[~mask_ground].copy()
        df_g = df_meta[mask_ground].copy()

        # --- Process Player-Player ---
        if not df_pp.empty:
            df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

            # Merge Player 1
            df_pp = df_pp.merge(
                df_track,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            df_pp = df_pp.rename(
                columns={
                    c: f"{c}_p1"
                    for c in df_track.columns
                    if c not in ["game_play", "step"]
                }
            )

            # Merge Player 2
            df_pp = df_pp.merge(
                df_track,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            df_pp = df_pp.rename(
                columns={
                    c: f"{c}_p2"
                    for c in df_track.columns
                    if c not in ["game_play", "step"]
                }
            )

            # Drop redundant merge keys
            df_pp = df_pp.drop(
                columns=["nfl_player_id_x", "nfl_player_id_y"], errors="ignore"
            )

        # --- Process Ground ---
        if not df_g.empty:
            # Merge Player 1 only
            df_g = df_g.merge(
                df_track,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            df_g = df_g.rename(
                columns={
                    c: f"{c}_p1"
                    for c in df_track.columns
                    if c not in ["game_play", "step"]
                }
            )
            df_g = df_g.drop(columns=["nfl_player_id"], errors="ignore")

            # Create placeholder columns for Player 2 (Ground)
            # These will be handled via Sentinel values in feature engineering
            p2_cols = [
                f"{c}_p2"
                for c in df_track.columns
                if c not in ["game_play", "step", "nfl_player_id"]
            ]
            for col in p2_cols:
                df_g[col] = 0.0

        # Recombine
        df_combined = pd.concat([df_pp, df_g], axis=0, ignore_index=True)
        return df_combined

    def apply_kinematic_gating(self, df):
        """
        Applies Relaxed Kinematic Reachability Gating.
        Filters pairs where the projected minimum distance over a 1.0s horizon
        is greater than GATING_THRESHOLD.
        """
        print(
            f"Applying Kinematic Gating (Threshold < {config.GATING_THRESHOLD} yds)..."
        )

        # Always keep Ground interactions
        mask_ground = df["nfl_player_id_2"] == "G"

        # Identify Player-Player rows
        mask_pp = ~mask_ground

        if not mask_pp.any():
            return df

        # Extract relative kinematics for PP
        df_pp = df[mask_pp].copy()

        # Relative Position
        dx = df_pp["x_position_p1"] - df_pp["x_position_p2"]
        dy = df_pp["y_position_p1"] - df_pp["y_position_p2"]

        # Relative Velocity
        dvx = df_pp["v_x_p1"] - df_pp["v_x_p2"]
        dvy = df_pp["v_y_p1"] - df_pp["v_y_p2"]

        # Relative Acceleration
        dax = df_pp["a_x_p1"] - df_pp["a_x_p2"]
        day = df_pp["a_y_p1"] - df_pp["a_y_p2"]

        # Calculate projected distance squared at multiple time steps
        # Horizon: 0.0s to 1.0s
        min_dist_sq = np.full(len(df_pp), np.inf)
        time_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for t in time_steps:
            # P(t) = P0 + V*t + 0.5*A*t^2
            px_t = dx + dvx * t + 0.5 * dax * (t**2)
            py_t = dy + dvy * t + 0.5 * day * (t**2)
            dist_sq = px_t**2 + py_t**2
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        # Determine survivors
        threshold_sq = config.GATING_THRESHOLD**2
        keep_mask_pp = min_dist_sq < threshold_sq

        # Map back to full dataframe
        # We keep ALL Ground rows, and ONLY surviving PP rows
        full_keep_mask = mask_ground.copy()
        full_keep_mask[mask_pp] = keep_mask_pp

        filtered_df = df[full_keep_mask].reset_index(drop=True)

        print(f"Gating reduced rows from {len(df)} to {len(filtered_df)}")
        return filtered_df

    def compute_vector_features(self, df):
        """
        Computes Collision-Aligned Vector Features.
        Decomposes kinematics into Radial (Impact) and Tangential (Shear) components.
        """
        print("Computing Collision-Aligned Vector Features...")

        # Initialize feature columns
        new_cols = config.VECTOR_FEATURES + [
            "distance",
            "orientation_diff",
            "direction_diff",
        ]
        for col in new_cols:
            df[col] = 0.0

        # --- Handle Ground (Sentinel Strategy) ---
        mask_ground = df["nfl_player_id_2"] == "G"

        if mask_ground.any():
            # Sentinel Distance
            df.loc[mask_ground, "distance"] = config.GROUND_DISTANCE_SENTINEL
            # Radial components map directly to Player 1's state
            df.loc[mask_ground, "v_radial"] = df.loc[mask_ground, "speed_p1"]
            df.loc[mask_ground, "a_radial"] = df.loc[mask_ground, "acceleration_p1"]
            # Tangential components are 0

        # --- Handle Player-Player ---
        mask_pp = ~mask_ground
        if mask_pp.any():
            # 1. Relative Position Vector (P2 -> P1)
            dx = df.loc[mask_pp, "x_position_p1"] - df.loc[mask_pp, "x_position_p2"]
            dy = df.loc[mask_pp, "y_position_p1"] - df.loc[mask_pp, "y_position_p2"]
            dist = np.sqrt(dx**2 + dy**2)

            # Safe distance for division
            dist_safe = dist.replace(0, 1e-6)

            # 2. Basis Vectors
            # Radial Unit Vector (Collision Axis)
            ur_x = dx / dist_safe
            ur_y = dy / dist_safe

            # Tangential Unit Vector (Orthogonal)
            ut_x = -ur_y
            ut_y = ur_x

            # 3. Relative Kinematics
            dvx = df.loc[mask_pp, "v_x_p1"] - df.loc[mask_pp, "v_x_p2"]
            dvy = df.loc[mask_pp, "v_y_p1"] - df.loc[mask_pp, "v_y_p2"]

            dax = df.loc[mask_pp, "a_x_p1"] - df.loc[mask_pp, "a_x_p2"]
            day = df.loc[mask_pp, "a_y_p1"] - df.loc[mask_pp, "a_y_p2"]

            # 4. Projections (Dot Products)
            df.loc[mask_pp, "v_radial"] = dvx * ur_x + dvy * ur_y
            df.loc[mask_pp, "v_tangential"] = dvx * ut_x + dvy * ut_y

            df.loc[mask_pp, "a_radial"] = dax * ur_x + day * ur_y
            df.loc[mask_pp, "a_tangential"] = dax * ut_x + day * ut_y

            df.loc[mask_pp, "distance"] = dist

            # 5. Angular Differences
            # Calculate shortest arc difference for orientation and direction
            ori_diff = np.abs(
                df.loc[mask_pp, "orientation_p1"] - df.loc[mask_pp, "orientation_p2"]
            )
            ori_diff = np.minimum(ori_diff, 360 - ori_diff)
            df.loc[mask_pp, "orientation_diff"] = ori_diff

            dir_diff = np.abs(
                df.loc[mask_pp, "direction_p1"] - df.loc[mask_pp, "direction_p2"]
            )
            dir_diff = np.minimum(dir_diff, 360 - dir_diff)
            df.loc[mask_pp, "direction_diff"] = dir_diff

        # --- Temporal Features (Jerk & Energy) ---
        # Sort to ensure temporal continuity for diff()
        sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        df = df.sort_values(sort_cols)

        # Efficient Grouping ID
        # Factorize the combination of keys to create an integer group ID
        group_keys = (
            df["game_play"].astype(str)
            + "_"
            + df["nfl_player_id_1"].astype(str)
            + "_"
            + df["nfl_player_id_2"].astype(str)
        )
        group_ids = pd.factorize(group_keys)[0]

        # Calculate Jerk: d(a_radial)/dt
        # We use numpy array shifting for speed
        a_rad = df["a_radial"].values

        # Shift arrays
        a_rad_prev = np.roll(a_rad, 1)
        group_prev = np.roll(group_ids, 1)

        # Calculate diff
        jerk = a_rad - a_rad_prev

        # Mask boundaries (where group changes)
        # The first element of a new group should have jerk=0 (or NaN, here 0)
        mask_boundary = group_ids != group_prev
        mask_boundary[0] = True  # Handle very first element
        jerk[mask_boundary] = 0.0

        df["a_radial_jerk"] = jerk

        # Calculate Energy: Squared magnitude of Jerk
        # Represents the power of the shock signal
        df["a_radial_energy"] = jerk**2

        # Cleanup intermediate columns
        drop_cols = [
            "dir_rad",
            "v_x",
            "v_y",
            "a_x",
            "a_y",
            "v_x_p1",
            "v_y_p1",
            "a_x_p1",
            "a_y_p1",
            "v_x_p2",
            "v_y_p2",
            "a_x_p2",
            "a_y_p2",
        ]
        df = df.drop(columns=drop_cols, errors="ignore")

        return df

    def process_data(self, split="train", load_cached_data=True, debug_sample=None):
        """
        Main execution pipeline.
        Checks cache -> Loads Raw -> Merges -> Gates -> Computes Features -> Caches.
        """
        cache_path = self._get_cache_path(split)

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"Processing {split} data from scratch...")

        # 2. Determine Paths
        if split == "train":
            meta_path = config.TRAIN_METADATA_PATH
            track_path = config.TRACKING_PATH_TRAIN
        elif split == "val":
            meta_path = config.VAL_METADATA_PATH
            track_path = config.TRACKING_PATH_TRAIN
        elif split == "test":
            meta_path = config.TEST_METADATA_PATH
            track_path = config.TRACKING_PATH_TEST
        else:
            raise ValueError(f"Unknown split: {split}")

        # 3. Load Metadata
        df_meta = pd.read_csv(meta_path)

        if debug_sample is not None:
            print(f"DEBUG: Sampling {debug_sample} rows...")
            df_meta = df_meta.iloc[:debug_sample].copy()

        # 4. Load Tracking
        df_track = self.load_tracking_data(track_path)

        # 5. Merge
        df_merged = self.merge_data(df_meta, df_track)
        del df_track
        gc.collect()

        # 6. Kinematic Gating
        # Note: We apply gating to all splits.
        # For Test, this implicitly predicts 0 for gated rows (handled in inference).
        df_gated = self.apply_kinematic_gating(df_merged)
        del df_merged
        gc.collect()

        # 7. Feature Engineering
        df_features = self.compute_vector_features(df_gated)

        # 8. Memory Reduction
        df_features = reduce_mem_usage(df_features)

        # 9. Save to Cache
        print(f"Saving features to {cache_path}...")
        df_features.to_parquet(cache_path, index=False)

        return df_features
