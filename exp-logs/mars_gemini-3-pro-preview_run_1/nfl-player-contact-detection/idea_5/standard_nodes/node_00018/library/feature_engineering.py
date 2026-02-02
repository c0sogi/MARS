import os
import pandas as pd
import numpy as np
import json
import hashlib
from scipy.spatial import cKDTree
from library.config import Config


class FeatureEngine:
    def __init__(self):
        self.config = Config
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

    def generate_features(
        self, df_merged, df_tracking, split_name, load_cached_data=True
    ):
        """
        Main pipeline to generate features with caching.

        Args:
            df_merged (pd.DataFrame): Metadata merged with P1/P2 tracking data.
            df_tracking (pd.DataFrame): Raw tracking data for topology context.
            split_name (str): Name of the split (train/val/test) for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The processed dataframe with flattened features.
        """
        # 1. Generate Hash & Cache Path
        feature_hash = self.config.get_feature_hash()
        # Include split_name and row count to avoid collisions between train/val/test/sample
        cache_filename = (
            f"features_{split_name}_{len(df_merged)}_{feature_hash}.parquet"
        )
        cache_path = os.path.join(self.working_dir, cache_filename)

        # 2. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Feature Cache hit! Loading from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Error loading feature cache: {e}. Recomputing...")

        print(f"Feature Cache miss. Computing features for {len(df_merged)} rows...")

        # 3. Compute Features
        # A. Physics Derivatives
        print("Computing Physics Derivatives...")
        df_proc = self._compute_physics_derivatives(df_merged)

        # B. Topological Features / Spatial Density
        if self.config.USE_TOPOLOGY:
            print("Computing Topological Features...")
            df_proc = self._compute_topological_features(df_proc, df_tracking)

        if self.config.USE_SPATIAL_DENSITY:
            print("Computing Spatial Density...")
            df_proc = self._compute_spatial_density(df_proc, df_tracking)

        # C. Temporal Windowing
        print("Applying Temporal Windowing...")
        df_final = self._create_temporal_windows(df_proc)

        # 4. Save Cache
        print(f"Saving features to {cache_path}...")
        df_final.to_parquet(cache_path, index=False)

        return df_final

    def _compute_physics_derivatives(self, df):
        """
        Computes Jerk and Angular Jerk.
        """
        # Ensure data is sorted by time for correct diff
        # Sort by game_play, p1, p2, step
        df = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Group by interaction pair
        grp_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]

        # Calculate Jerk: d(Acceleration)/dt. dt = 0.1s
        # We use transform to keep the index aligned.
        # Note: groupby().diff() preserves index.
        df["jerk_p1"] = df.groupby(grp_cols)["acceleration_p1"].diff().fillna(0) / 0.1
        df["jerk_p2"] = df.groupby(grp_cols)["acceleration_p2"].diff().fillna(0) / 0.1

        # Calculate Angular Jerk: d(Direction)/dt
        # Handle circular difference for degrees
        def get_angular_diff(series):
            diff = series.diff().fillna(0)
            # Normalize to [-180, 180]
            diff = (diff + 180) % 360 - 180
            return diff / 0.1

        df["angular_jerk_p1"] = df.groupby(grp_cols)["direction_p1"].transform(
            get_angular_diff
        )
        df["angular_jerk_p2"] = df.groupby(grp_cols)["direction_p2"].transform(
            get_angular_diff
        )

        return df

    def _compute_spatial_density(self, df_meta, df_tracking):
        """
        Computes spatial density (neighbor count) for each player.
        Cite solution_lesson_node_00012: Spatial context via neighbor count.
        """
        # 1. Filter tracking data
        relevant_plays = df_meta["game_play"].unique()
        df_track_rel = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()

        # 2. Prepare storage
        density_results = []

        # 3. Iterate over unique frames
        grouped = df_track_rel.groupby(["game_play", "step"])

        for (game_play, step), group in grouped:
            ids = group["nfl_player_id"].values
            coords = group[["x_position", "y_position"]].values

            if len(ids) == 0:
                continue

            # Build Tree
            tree = cKDTree(coords)

            # Query neighbors within radius
            # query_ball_point returns list of indices
            # We want count.
            # k=100 to find all neighbors essentially
            counts = tree.query_ball_point(
                coords, r=self.config.DENSITY_RADIUS, return_length=True
            )

            # Count includes self, so subtract 1
            counts = counts - 1

            for i, pid in enumerate(ids):
                density_results.append((game_play, step, pid, counts[i]))

        # 4. Convert to DataFrame
        df_density = pd.DataFrame(
            density_results,
            columns=["game_play", "step", "nfl_player_id", "spatial_density"],
        )

        # 5. Merge back to metadata
        # P1
        df_meta = (
            df_meta.merge(
                df_density,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            )
            .drop(columns=["nfl_player_id"])
            .rename(columns={"spatial_density": "spatial_density_p1"})
        )

        # P2
        df_meta["p2_join"] = pd.to_numeric(df_meta["nfl_player_id_2"], errors="coerce")
        df_meta = (
            df_meta.merge(
                df_density,
                left_on=["game_play", "step", "p2_join"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
                suffixes=("", "_p2_dens"),
            )
            .drop(columns=["nfl_player_id", "p2_join"])
            .rename(columns={"spatial_density": "spatial_density_p2"})
        )

        # Fill NaNs
        df_meta[["spatial_density_p1", "spatial_density_p2"]] = df_meta[
            ["spatial_density_p1", "spatial_density_p2"]
        ].fillna(0)

        return df_meta

    def _create_temporal_windows(self, df):
        """
        Flattens temporal window (+/- WINDOW_SIZE) into a single row.
        """
        # Sort
        df = df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Define features to flatten
        feature_candidates = [
            "x_position_p1",
            "y_position_p1",
            "speed_p1",
            "direction_p1",
            "orientation_p1",
            "acceleration_p1",
            "sa_p1",
            "x_position_p2",
            "y_position_p2",
            "speed_p2",
            "direction_p2",
            "orientation_p2",
            "acceleration_p2",
            "sa_p2",
            "jerk_p1",
            "jerk_p2",
            "angular_jerk_p1",
            "angular_jerk_p2",
            "spatial_density_p1",
            "spatial_density_p2",
            "distance",
        ]

        # Calculate distance if missing
        if "distance" not in df.columns and "x_position_p1" in df.columns:
            df["distance"] = np.sqrt(
                (df["x_position_p1"] - df["x_position_p2"]) ** 2
                + (df["y_position_p1"] - df["y_position_p2"]) ** 2
            )

        # Select only existing columns
        features = [c for c in feature_candidates if c in df.columns]

        # Group by pair
        grp = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        # Create shifts
        shifted_dfs = []
        # Range: -WINDOW to +WINDOW
        for s in range(-self.config.WINDOW_SIZE, self.config.WINDOW_SIZE + 1):
            # shift(s): positive s is lag (t-s), negative s is lead (t+s)
            res = grp[features].shift(s)

            # Naming
            if s > 0:
                suffix = f"_lag_{s}"
            elif s < 0:
                suffix = f"_lead_{-s}"
            else:
                suffix = ""

            res.columns = [f"{c}{suffix}" for c in features]
            shifted_dfs.append(res)

        # Concatenate all features
        df_shifts = pd.concat(shifted_dfs, axis=1)

        # Combine with metadata
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "is_ground",
        ]
        meta_cols = [c for c in meta_cols if c in df.columns]

        df_final = pd.concat([df[meta_cols], df_shifts], axis=1)

        # Fill NaNs generated by shifting (start/end of play) with 0
        df_final = df_final.fillna(0)

        return df_final
