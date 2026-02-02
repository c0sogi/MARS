import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
from tqdm import tqdm
import gc

from library.config import Config
from library.utils import (
    load_raw_data,
    save_to_parquet,
    load_from_parquet,
    generate_cache_key,
)


class FeatureProcessor:
    def __init__(self):
        self.config = Config
        self.cache_key = generate_cache_key()

    def _compute_physics_derivatives(self, df_track):
        """
        Computes Jerk and Angular Jerk on the tracking data.
        Assumes df_track is sorted by game_play, nfl_player_id, step.
        """
        print("Computing physics derivatives (Jerk, Angular Jerk)...")
        # Ensure sorting
        df_track = df_track.sort_values(by=["game_play", "nfl_player_id", "step"])

        # Group by play and player to prevent diffing across boundaries
        # We assume 0.1s steps for the derivative, so simple diff is proportional to derivative
        grp = df_track.groupby(["game_play", "nfl_player_id"])

        df_track["jerk"] = grp["acceleration"].diff().fillna(0)

        # Angular jerk: diff of orientation. Handle potential wrap-around if needed,
        # but simple diff is usually sufficient for "sudden change" detection in this context.
        df_track["angular_jerk"] = grp["orientation"].diff().fillna(0)

        # Fill NaNs created by diff (first step of each player)
        df_track["jerk"] = df_track["jerk"].fillna(0).astype("float32")
        df_track["angular_jerk"] = df_track["angular_jerk"].fillna(0).astype("float32")

        return df_track

    def _compute_spatial_density(self, df_track):
        """
        Computes the number of other players within 1.5 yards for each player/step.
        Uses cKDTree for efficiency.
        """
        print("Computing spatial density features...")

        # We need to process by (game_play, step)
        # To vectorise, we can iterate over unique frames

        # Initialize result array
        # We will map results back to the dataframe index
        density_map = np.zeros(len(df_track), dtype=np.int8)

        # Create a mapping from index to row number for fast assignment
        # Or better: iterate groups, compute, and assign by index

        # Filter columns to minimal needed
        coords = df_track[["x_position", "y_position"]].values
        indices = df_track.index.values

        # Group by game_play and step
        # We use a composite key for grouping
        df_track["_frame_id"] = (
            df_track["game_play"].astype(str) + "_" + df_track["step"].astype(str)
        )
        groups = df_track.groupby("_frame_id")

        # Iterate over groups
        # Note: This loop can be slow if there are many frames.
        # Train set ~3.4M rows, tracking ~1.2M rows.
        # Number of unique frames is roughly 1.2M / 22 players ~= 55k frames.
        # Tqdm makes it visible.

        for _, group_indices in tqdm(
            groups.indices.items(), desc="Spatial Density", miniters=1000
        ):
            # Get coordinates for this frame
            frame_coords = coords[group_indices]

            if len(frame_coords) > 1:
                # Build Tree
                tree = cKDTree(frame_coords)

                # Query: count neighbors within 1.5 yards
                # k=None returns all neighbors within distance
                # query_ball_point returns list of lists of neighbors
                neighbors = tree.query_ball_point(frame_coords, r=1.5)

                # Count neighbors minus self (1)
                counts = [len(n) - 1 for n in neighbors]

                # Assign back
                density_map[group_indices] = counts
            else:
                density_map[group_indices] = 0

        df_track["spatial_density"] = density_map
        df_track = df_track.drop(columns=["_frame_id"])
        return df_track

    def _preprocess_tracking(self, df_track):
        """
        Applies all tracking-level feature engineering.
        """
        # 1. Physics
        if self.config.USE_IMPACT_PHYSICS:
            df_track = self._compute_physics_derivatives(df_track)
        else:
            df_track["jerk"] = 0.0
            df_track["angular_jerk"] = 0.0

        # 2. Spatial Density
        if self.config.USE_SPATIAL_DENSITY:
            df_track = self._compute_spatial_density(df_track)
        else:
            df_track["spatial_density"] = 0

        return df_track

    def _merge_metadata(self, df_meta, df_track):
        """
        Merges metadata with processed tracking data for Player 1 and Player 2.
        """
        print("Merging metadata with tracking data...")

        # Prepare Metadata
        # Handle Ground: nfl_player_id_2 can be 'G'
        df_meta["is_ground"] = (df_meta["nfl_player_id_2"] == "G").astype(int)

        # Convert IDs to numeric for merging, 'G' becomes NaN/Sentinel
        df_meta["p1_id"] = pd.to_numeric(df_meta["nfl_player_id_1"], errors="coerce")
        df_meta["p2_id"] = pd.to_numeric(df_meta["nfl_player_id_2"], errors="coerce")

        # Select relevant tracking columns
        track_cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "orientation",
            "direction",
            "jerk",
            "angular_jerk",
            "spatial_density",
        ]
        # Ensure tracking types match metadata for merge keys
        df_track["game_play"] = df_track["game_play"].astype(str)
        df_track["step"] = df_track["step"].astype(int)
        df_track["nfl_player_id"] = pd.to_numeric(
            df_track["nfl_player_id"], errors="coerce"
        )

        df_track_sub = df_track[track_cols].copy()

        # --- Merge Player 1 ---
        df_merged = df_meta.merge(
            df_track_sub,
            left_on=["game_play", "step", "p1_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        # Rename P1 columns
        rename_p1 = {
            c: f"{c}_p1"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_p1)
        df_merged = df_merged.drop(columns=["nfl_player_id"])  # Drop join key

        # --- Merge Player 2 ---
        # For Ground rows, this merge will yield NaNs, which is expected
        df_merged = df_merged.merge(
            df_track_sub,
            left_on=["game_play", "step", "p2_id"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        # Rename P2 columns
        rename_p2 = {
            c: f"{c}_p2"
            for c in track_cols
            if c not in ["game_play", "step", "nfl_player_id"]
        }
        df_merged = df_merged.rename(columns=rename_p2)
        df_merged = df_merged.drop(columns=["nfl_player_id"])

        # Fill NaNs for Player 2 (Ground or missing tracking)
        # For Ground, positions are technically undefined, but we use 0 or fill forward?
        # 0 is safer for tree models than arbitrary imputation, provided we have 'is_ground'
        p2_features = list(rename_p2.values())
        df_merged[p2_features] = df_merged[p2_features].fillna(0)

        # Also fill P1 NaNs if any (missing tracking)
        p1_features = list(rename_p1.values())
        df_merged[p1_features] = df_merged[p1_features].fillna(0)

        return df_merged

    def _create_interaction_features(self, df):
        """
        Creates features based on the interaction between P1 and P2.
        """
        print("Creating interaction features...")

        # Distance
        # If Ground, distance is meaningless in X/Y plane.
        # We set it to a distinct value or 0.
        # Since we have is_ground, the model can split on that.
        # However, large distance usually means no contact.
        # For ground, we want the model to focus on P1's motion, not distance.
        # Setting distance to -1 or 0 for ground.

        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Mask distance for ground interactions
        df["distance"] = np.where(df["is_ground"] == 1, -1.0, dist)

        # Speed Diff
        df["speed_diff"] = np.abs(df["speed_p1"] - df["speed_p2"])

        # Orientation Diff
        # Simple absolute difference
        df["orientation_diff"] = np.abs(df["orientation_p1"] - df["orientation_p2"])

        return df

    def _create_windowed_features(self, df):
        """
        Creates flattened temporal features using shift.
        Window size defined in Config.
        """
        print(
            f"Creating windowed features (Window Size: +/- {self.config.WINDOW_HALF_SIZE})..."
        )

        # Features to window
        # We include interaction features and individual kinematics
        features_to_window = [
            "distance",
            "speed_diff",
            "speed_p1",
            "acceleration_p1",
            "jerk_p1",
            "spatial_density_p1",
            "speed_p2",
            "acceleration_p2",
            "jerk_p2",
        ]

        # Sort is critical for shift
        sort_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

        # We need to ensure we don't shift data from one pair to another.
        # We can construct a group ID.
        # Using factorize is fast.
        # Note: nfl_player_id_2 can be string 'G', so we use the columns as is.
        # We combine them into a single ID for grouping check.

        # Optimisation: Check continuity using the columns directly in mask
        # mask = (current_group == shifted_group) & (current_step + lag == shifted_step)

        # Pre-calculate group identifiers for fast comparison
        # We can just check if game_play, p1, p2 are same
        g_play = df["game_play"].values
        p1 = df["nfl_player_id_1"].values
        p2 = df["nfl_player_id_2"].values
        steps = df["step"].values

        window_size = self.config.WINDOW_HALF_SIZE

        # Loop for lags
        for lag in range(-window_size, window_size + 1):
            if lag == 0:
                continue

            suffix = f"_lag{lag}" if lag > 0 else f"_lead{abs(lag)}"

            # Shift data
            # shift(lag): positive lag takes previous data (t-k) -> Lag feature
            # shift(-lag): negative lag takes future data (t+k) -> Lead feature
            # We want t-10 to t+10.
            # If lag is +1 (t-1), we use shift(1).
            # If lag is -1 (t+1), we use shift(-1).
            # Loop range is -10 to +10.
            # if i = -10 (past), we want shift(10).
            # if i = +10 (future), we want shift(-10).
            # Let's stick to convention: lag k means t-k.
            # So loop i from -10 to 10.
            # feature_t+i.
            # If i=-10, feature at t-10. We need shift(10).
            # If i=10, feature at t+10. We need shift(-10).

            shift_amount = -lag  # To get data at t+lag

            # Perform shift on feature block
            shifted_data = df[features_to_window].shift(shift_amount)

            # Verify validity
            # We need:
            # 1. Same Game/Play/Pair
            # 2. Step continuity: step[shifted_idx] == step[current] + lag

            # Get shifted identifiers
            g_play_s = pd.Series(g_play).shift(shift_amount)
            p1_s = pd.Series(p1).shift(shift_amount)
            p2_s = pd.Series(p2).shift(shift_amount)
            steps_s = pd.Series(steps).shift(shift_amount)

            # Create Mask
            # Note: Series comparison handles NaNs from shift automatically (False)
            valid_mask = (
                (g_play_s == g_play)
                & (p1_s == p1)
                & (p2_s == p2)
                & (steps_s == steps + lag)
            )

            # Apply mask: where invalid, set to 0 or NaN?
            # GBDT handles NaN, but 0 is often used for padding.
            # Let's use 0 for padding to keep consistent with missing tracking data.
            shifted_data[~valid_mask] = 0

            # Rename columns
            shifted_data.columns = [f"{c}{suffix}" for c in features_to_window]

            # Concatenate efficiently
            df = pd.concat([df, shifted_data], axis=1)

        return df

    def process_split(self, split="train", load_cached_data=True):
        """
        Main pipeline execution for a data split.
        """
        # 1. Define Cache Path
        base_name = getattr(self.config, f"CACHE_{split.upper()}_FEATURES")
        cache_filename = f"{base_name}_{self.cache_key}.parquet"

        # 2. Try Load Cache
        if load_cached_data:
            df = load_from_parquet(cache_filename)
            if df is not None:
                print(f"Loaded {split} features from cache.")
                return df

        print(f"Generating {split} features from scratch...")

        # 3. Load Raw Data
        df_meta, df_track = load_raw_data(split=split, load_tracking=True)

        # 4. Preprocess Tracking (Physics + Density)
        # Only if tracking exists (it should)
        if df_track is not None:
            df_track = self._preprocess_tracking(df_track)

            # 5. Merge
            df_features = self._merge_metadata(df_meta, df_track)
        else:
            raise ValueError("Tracking data is missing.")

        # 6. Interaction Features
        df_features = self._create_interaction_features(df_features)

        # 7. Windowing
        df_features = self._create_windowed_features(df_features)

        # 8. Cleanup & Save
        # Drop raw path columns to save space
        cols_to_drop = [
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "datetime",
        ]
        df_features = df_features.drop(
            columns=[c for c in cols_to_drop if c in df_features.columns]
        )

        # Downcast floats to float32 to save memory
        f_cols = df_features.select_dtypes(include=["float64"]).columns
        df_features[f_cols] = df_features[f_cols].astype("float32")

        save_to_parquet(df_features, cache_filename)

        # Force GC
        del df_meta, df_track
        gc.collect()

        return df_features
