import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import execute_with_cache, setup_logger

logger = setup_logger()


class FeatureEngineer:
    """
    Implements the Ego-Centric Spatial Grid Mining feature engineering pipeline.
    """

    def __init__(self):
        self.tracking_cols = [
            "x_position",
            "y_position",
            "speed",
            "direction",
            "orientation",
            "acceleration",
            "sa",
        ]
        # Features created during tracking processing
        self.derived_tracking_cols = [
            "jerk",
            "angular_jerk",
            "speed_roll_mean",
            "acc_roll_mean",
        ]
        # Grid features will be dynamically named based on bins

    def process_tracking_data(self, tracking_path, load_cached_data=True):
        """
        Loads tracking data, computes Kinematics and Ego-Centric Grids.
        Returns a dataframe with one row per (game_play, step, nfl_player_id).
        """
        cache_key = f"processed_tracking_{os.path.basename(tracking_path).replace('.csv', '')}.parquet"

        def _compute(path):
            logger.info(f"Loading tracking data from {path}...")
            df_track = pd.read_csv(path)

            # 1. Kinematics & Physics
            logger.info("Computing Kinematics (Jerk, Rolling Windows)...")
            df_track = df_track.sort_values(["game_play", "nfl_player_id", "step"])

            # Group by player to respect play boundaries
            g = df_track.groupby(["game_play", "nfl_player_id"])

            # Jerk (Derivative of Acceleration)
            # Time delta is 0.1s
            df_track["jerk"] = g["acceleration"].diff().fillna(0) / 0.1

            # Angular Jerk (Derivative of Orientation/Direction changes)
            # Handle angular wrap-around (0-360)
            d_orient = g["orientation"].diff().fillna(0)
            d_orient = np.where(d_orient > 180, d_orient - 360, d_orient)
            d_orient = np.where(d_orient < -180, d_orient + 360, d_orient)
            df_track["angular_jerk"] = d_orient / 0.1

            # Rolling Windows (Temporal Context)
            # We use a centered window if possible, or lagging.
            # Given the task is predictive/detection, centered is fine as we have the full video.
            window = Config.WINDOW_SIZE
            df_track["speed_roll_mean"] = g["speed"].transform(
                lambda x: x.rolling(window, center=True, min_periods=1).mean()
            )
            df_track["acc_roll_mean"] = g["acceleration"].transform(
                lambda x: x.rolling(window, center=True, min_periods=1).mean()
            )

            # 2. Ego-Centric Spatial Grids
            logger.info("Computing Ego-Centric Spatial Grids (Vectorized)...")

            # We need to find neighbors for every player.
            # Strategy: Self-merge on (game_play, step)
            # To save memory, process only necessary columns for the grid calculation
            grid_cols = [
                "game_play",
                "step",
                "nfl_player_id",
                "x_position",
                "y_position",
                "orientation",
                "speed",
            ]
            df_base = df_track[grid_cols].copy()

            # Merge to get P1 (Ego) and P2 (Neighbor)
            # This expands the dataframe significantly (N_players^2 per step)
            # We process this carefully.
            df_neighbors = pd.merge(
                df_base, df_base, on=["game_play", "step"], suffixes=("", "_neighbor")
            )

            # Filter out self
            df_neighbors = df_neighbors[
                df_neighbors["nfl_player_id"] != df_neighbors["nfl_player_id_neighbor"]
            ]

            # Calculate Relative Position (Global Frame)
            dx = df_neighbors["x_position_neighbor"] - df_neighbors["x_position"]
            dy = df_neighbors["y_position_neighbor"] - df_neighbors["y_position"]

            # Rotate to Ego Frame
            # Orientation is in degrees. 0 usually means Y-axis or X-axis depending on standard.
            # Assuming standard math: 0 is East (X), 90 is North (Y).
            # NFL data: 0 is usually Y-axis (short axis) facing distinct direction?
            # Let's assume standard rotation matrix logic.
            # Convert to radians
            theta = np.radians(df_neighbors["orientation"])

            # Rotation Matrix for aligning Ego's forward vector to Y-axis (or X-axis)
            # Let's align Ego's orientation to the positive Y-axis (Up)
            # x' = dx * cos(theta) + dy * sin(theta)  <-- This projects onto the perpendicular
            # y' = -dx * sin(theta) + dy * cos(theta) <-- This projects onto the forward vector
            # (Standard 2D rotation by -theta + 90 or similar. Let's stick to standard rotation by -theta)
            # x_rot = dx * cos(-t) - dy * sin(-t)
            # y_rot = dx * sin(-t) + dy * cos(-t)
            # If theta is 0 (facing East/X), no rotation.

            c, s = np.cos(-theta), np.sin(-theta)
            df_neighbors["x_local"] = dx * c - dy * s
            df_neighbors["y_local"] = dx * s + dy * c

            # Binning
            # 3x3 Grid:
            # X (Side): [-Inf, -1], [-1, 1], [1, Inf]
            # Y (Front/Back): [-Inf, -1], [-1, 1], [1, Inf]
            res = Config.GRID_RESOLUTION

            # Define bins
            x_bins = [-np.inf, -res, res, np.inf]
            y_bins = [-np.inf, -res, res, np.inf]

            df_neighbors["bin_x"] = pd.cut(
                df_neighbors["x_local"], bins=x_bins, labels=["Left", "Center", "Right"]
            )
            df_neighbors["bin_y"] = pd.cut(
                df_neighbors["y_local"], bins=y_bins, labels=["Back", "Middle", "Front"]
            )

            # Combine labels
            df_neighbors["grid_loc"] = (
                df_neighbors["bin_y"].astype(str)
                + "_"
                + df_neighbors["bin_x"].astype(str)
            )

            # Aggregate
            # Count neighbors and Sum speed in each bin
            grid_feats = (
                df_neighbors.groupby(["game_play", "step", "nfl_player_id", "grid_loc"])
                .agg({"nfl_player_id_neighbor": "count", "speed_neighbor": "mean"})
                .unstack(fill_value=0)
            )

            # Flatten columns
            grid_feats.columns = [
                f"grid_{col[1]}_{col[0]}" for col in grid_feats.columns
            ]
            grid_feats = grid_feats.reset_index()

            # Merge grid features back to main tracking
            logger.info("Merging grid features back to tracking data...")
            df_track = pd.merge(
                df_track,
                grid_feats,
                on=["game_play", "step", "nfl_player_id"],
                how="left",
            )

            # Fill NaNs for players with no neighbors (rare, but possible)
            grid_cols_names = [c for c in df_track.columns if c.startswith("grid_")]
            df_track[grid_cols_names] = df_track[grid_cols_names].fillna(0)

            # Downcast to save memory
            for col in df_track.select_dtypes(include=["float64"]).columns:
                df_track[col] = df_track[col].astype("float32")

            return df_track

        return execute_with_cache(
            cache_key, _compute, load_cached_data=load_cached_data, path=tracking_path
        )

    def generate_dataset(
        self, metadata_path, tracking_path, mode="train", load_cached_data=True
    ):
        """
        Main pipeline to generate X, y for a specific split.
        mode: 'train', 'val', or 'test'
        """
        logger.info(f"Generating dataset for mode: {mode}")

        # 1. Load Metadata
        df_meta = pd.read_csv(metadata_path)

        # 2. Geometric Gating (Training/Val only)
        # For Test, we must predict for all rows in sample_submission.
        # However, to compute features efficiently, we still need the tracking data.

        # Load Processed Tracking (with Grids)
        df_tracking = self.process_tracking_data(
            tracking_path, load_cached_data=load_cached_data
        )

        # 3. Merge P1 Features (Ego)
        logger.info("Merging Player 1 Features...")
        # Ensure types match
        df_meta["game_play"] = df_meta["game_play"].astype(str)
        df_tracking["game_play"] = df_tracking["game_play"].astype(str)

        # Merge P1
        df_merged = pd.merge(
            df_meta,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )

        # Rename P1 columns that didn't get suffix (because they weren't in meta)
        # The merge keeps 'nfl_player_id' from tracking. Drop it.
        if "nfl_player_id" in df_merged.columns:
            df_merged = df_merged.drop(columns=["nfl_player_id"])

        # Identify columns that came from tracking
        track_cols = [
            c
            for c in df_tracking.columns
            if c not in ["game_play", "step", "nfl_player_id"]
        ]
        rename_map = {
            c: f"{c}_p1"
            for c in track_cols
            if not c.endswith("_p1") and f"{c}_p1" not in df_merged.columns
        }
        df_merged = df_merged.rename(columns=rename_map)

        # 4. Merge P2 Features (Target)
        logger.info("Merging Player 2 Features...")

        # Handle Ground
        is_ground = df_merged["nfl_player_id_2"] == "G"

        # Split into Player-Player and Player-Ground
        df_pp = df_merged[~is_ground].copy()
        df_pg = df_merged[is_ground].copy()

        # Process Player-Player
        if not df_pp.empty:
            df_pp["nfl_player_id_2"] = df_pp["nfl_player_id_2"].astype(int)

            # We only need basic kinematics for P2, not the full grid (too expensive/redundant)
            # Or should we include P2's grid? Let's include basic tracking for P2.
            # We can reuse df_tracking.

            df_pp = pd.merge(
                df_pp,
                df_tracking,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p2"),
            )

            if "nfl_player_id" in df_pp.columns:
                df_pp = df_pp.drop(columns=["nfl_player_id"])

            # Rename P2 cols
            rename_map_p2 = {
                c: f"{c}_p2" for c in track_cols if f"{c}_p2" not in df_pp.columns
            }
            df_pp = df_pp.rename(columns=rename_map_p2)

            # Compute Pair Features
            df_pp["distance"] = np.sqrt(
                (df_pp["x_position_p1"] - df_pp["x_position_p2"]) ** 2
                + (df_pp["y_position_p1"] - df_pp["y_position_p2"]) ** 2
            )
            df_pp["speed_diff"] = np.abs(df_pp["speed_p1"] - df_pp["speed_p2"])

            # Gating check (only for train/val optimization, not strictly required by logic but good for memory)
            if mode in ["train", "val"]:
                logger.info(
                    f"Applying Geometric Gating (Threshold: {Config.GATING_THRESHOLD} yards)..."
                )
                initial_len = len(df_pp)
                df_pp = df_pp[df_pp["distance"] <= Config.GATING_THRESHOLD]
                logger.info(
                    f"Dropped {initial_len - len(df_pp)} pairs based on distance."
                )

        # Process Player-Ground
        if not df_pg.empty:
            # Create dummy P2 features
            # Cite debug_lesson_16: Enforce Strict Schema Consistency When Generating Synthetic Rows
            for c in track_cols:
                # Only initialize numeric columns to 0.0 to avoid mixed-type errors (e.g. datetime)
                if pd.api.types.is_numeric_dtype(df_tracking[c]):
                    df_pg[f"{c}_p2"] = 0.0

            df_pg["distance"] = (
                0.0  # Distance to ground is conceptually 0 for contact logic, or handled by height (not available)
            )
            # Actually, distance to ground is usually 0 in this schema,
            # but we rely on P1's features (Jerk, etc.) to detect ground contact.
            df_pg["speed_diff"] = df_pg["speed_p1"]  # Relative to 0 speed ground

        # Recombine
        df_final = pd.concat([df_pp, df_pg], axis=0).sort_index()

        # Ensure nfl_player_id_2 is string to avoid Parquet mixed type errors
        df_final["nfl_player_id_2"] = df_final["nfl_player_id_2"].astype(str)

        # 5. Final Feature Selection
        # Identify feature columns
        feature_cols = [
            c
            for c in df_final.columns
            if c.endswith("_p1") or c.endswith("_p2") or c in ["distance", "speed_diff"]
        ]
        # Add is_ground indicator
        df_final["is_ground"] = (df_final["nfl_player_id_2"] == "G").astype(int)
        feature_cols.append("is_ground")

        # Drop non-numeric or ID columns from features
        exclude = [
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "contact",
            "contact_id",
        ]
        exclude += [c for c in df_final.columns if "video_path" in c]

        X = df_final[feature_cols].copy()

        # Handle NaNs (e.g., missing tracking data)
        X = X.fillna(0)

        if "contact" in df_final.columns:
            y = df_final["contact"]
        else:
            y = None

        # Add metadata for identification later if needed
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
        ]
        meta = df_final[meta_cols].copy()

        logger.info(f"Dataset Generated. Shape: {X.shape}")
        return X, y, meta
