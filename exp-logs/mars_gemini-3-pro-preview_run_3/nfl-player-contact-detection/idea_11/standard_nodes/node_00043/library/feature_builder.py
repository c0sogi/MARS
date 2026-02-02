import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.data_manager import DataManager
from library.utils import get_data_hash


class FeatureBuilder:
    """
    Constructs feature sets for the Contact Detection Pipeline.
    Implements the Robust Asymmetric Modality-Selective GBDT architecture.
    """

    def __init__(self):
        self.dm = DataManager()

    def _calculate_iou(self, box1, box2):
        """
        Vectorized Intersection over Union calculation.
        box: [left, top, width, height]
        """
        # Unpack boxes
        # box1: (N, 4), box2: (N, 4)
        # Format: left, top, width, height

        b1_x1 = box1[:, 0]
        b1_y1 = box1[:, 1]
        b1_x2 = b1_x1 + box1[:, 2]
        b1_y2 = b1_y1 + box1[:, 3]
        b1_area = box1[:, 2] * box1[:, 3]

        b2_x1 = box2[:, 0]
        b2_y1 = box2[:, 1]
        b2_x2 = b2_x1 + box2[:, 2]
        b2_y2 = b2_y1 + box2[:, 3]
        b2_area = box2[:, 2] * box2[:, 3]

        # Intersection
        ix1 = np.maximum(b1_x1, b2_x1)
        iy1 = np.maximum(b1_y1, b2_y1)
        ix2 = np.minimum(b1_x2, b2_x2)
        iy2 = np.minimum(b1_y2, b2_y2)

        i_width = np.maximum(0, ix2 - ix1)
        i_height = np.maximum(0, iy2 - iy1)
        i_area = i_width * i_height

        # Union
        u_area = b1_area + b2_area - i_area

        # Avoid division by zero
        iou = np.where(u_area > 0, i_area / u_area, 0.0)
        return iou

    def _calculate_centroid_distance(self, box1, box2):
        """
        Vectorized Centroid Distance calculation.
        """
        c1_x = box1[:, 0] + box1[:, 2] / 2
        c1_y = box1[:, 1] + box1[:, 3] / 2

        c2_x = box2[:, 0] + box2[:, 2] / 2
        c2_y = box2[:, 1] + box2[:, 3] / 2

        dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)
        return dist

    def _create_flattened_features(self, df_tracking, feature_cols, prefix=""):
        """
        Creates flattened time-series features (lags) for tracking data.
        """
        # Sort to ensure correct shifting
        df_sorted = df_tracking.sort_values(
            ["game_play", "nfl_player_id", "step"]
        ).copy()

        # Define lags based on Config
        # Micro: +/- 4, Macro: +/- 15
        # We take a subset of lags to keep dimensionality reasonable while capturing context
        # Lags: 0 (current), +/- 1, 2, 4 (Micro), +/- 8, 15 (Macro)
        lags = [0, -1, 1, -2, 2, -4, 4, -8, 8, -15, 15]

        # Group object for shifting
        grouper = df_sorted.groupby(["game_play", "nfl_player_id"])

        result_dfs = []

        for col in feature_cols:
            for lag in lags:
                col_name = f"{prefix}{col}_lag{lag}" if lag != 0 else f"{prefix}{col}"
                if lag == 0:
                    result_dfs.append(df_sorted[[col]].rename(columns={col: col_name}))
                else:
                    # Shift
                    shifted = grouper[col].shift(
                        -lag
                    )  # shift(-1) gets next row (future), shift(1) gets prev
                    # Note: pandas shift(1) moves data down, so row t gets t-1.
                    # We want lag k to represent t-k. So shift(k).
                    # If lag is negative (e.g. -4), we want t+4 (future). shift(-4).
                    # Let's standardize: lag k means t-k.
                    # If lag=4, we want value from 4 steps ago. shift(4).
                    # If lag=-4, we want value from 4 steps future. shift(-4).

                    s = grouper[col].shift(lag)
                    s.name = col_name
                    result_dfs.append(s)

        # Concatenate all features
        df_features = pd.concat(result_dfs, axis=1)

        # Add keys back for merging
        df_features["game_play"] = df_sorted["game_play"]
        df_features["nfl_player_id"] = df_sorted["nfl_player_id"]
        df_features["step"] = df_sorted["step"]

        return df_features

    def _prepare_helmets(self, df_helmets):
        """
        Prepares helmet data for merging.
        Maps frames to steps and pivots by view.
        """
        # Map frame to step
        # Step 0 = Frame 300 (approx 5s). Step k = Frame 300 + k * 6 (approx)
        # frame = 300 + step * 5.994
        # step = (frame - 300) / 5.994

        df_helmets = df_helmets.copy()
        df_helmets["approx_step"] = np.round(
            (df_helmets["frame"] - 300) / 5.994
        ).astype(int)

        # Filter columns
        cols = [
            "game_play",
            "nfl_player_id",
            "approx_step",
            "view",
            "left",
            "top",
            "width",
            "height",
        ]
        df_helmets = df_helmets[cols]

        # Pivot to have Sideline and Endzone columns in one row per player/step
        df_pivoted = df_helmets.pivot_table(
            index=["game_play", "nfl_player_id", "approx_step"],
            columns="view",
            values=["left", "top", "width", "height"],
            aggfunc="first",  # Take first if duplicates exist (rare)
        )

        # Flatten columns
        df_pivoted.columns = [f"{col[1]}_{col[0]}" for col in df_pivoted.columns]
        df_pivoted = df_pivoted.reset_index()
        df_pivoted = df_pivoted.rename(columns={"approx_step": "step"})

        return df_pivoted

    def build_stream_a_features(self, split, load_cached_data=True):
        """
        Builds features for Stream A (Interaction Model).
        Tracking (P1 & P2) + Conditional Visuals.
        """
        # Check cache
        cache_path = Config.get_feature_cache_path("streamA", split)
        ids_path = cache_path.replace(".parquet", "_ids.npy")
        y_path = cache_path.replace(".parquet", "_y.npy")

        if (
            load_cached_data
            and os.path.exists(cache_path)
            and os.path.exists(ids_path)
            and os.path.exists(y_path)
        ):
            print(f"Loading Stream A features from cache: {cache_path}")
            return (
                pd.read_parquet(cache_path),
                np.load(ids_path, allow_pickle=True),
                np.load(y_path),
            )

        print(f"Building Stream A features for {split}...")

        # 1. Load Data
        df_meta, df_tracking, df_helmets = self.dm.get_data(split, load_cached_data)

        # Filter for Interaction (P2 != G)
        df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(int)

        # 2. Process Tracking (Physics + Flattening)
        # Cite solution_lesson_node_00027: Enrich Kinematic Models
        # Cite solution_lesson_node_00005: Cyclical Feature Encoding
        df_tracking = df_tracking.sort_values(["game_play", "nfl_player_id", "step"])

        # Jerk
        df_tracking["jerk"] = (
            df_tracking.groupby(["game_play", "nfl_player_id"])["acceleration"]
            .diff()
            .fillna(0)
        )

        # Pose-Motion Alignment
        ori_rad = np.radians(df_tracking["orientation"].fillna(0))
        dir_rad = np.radians(df_tracking["direction"].fillna(0))
        df_tracking["pose_motion_alignment"] = np.cos(ori_rad - dir_rad)

        # Cyclical Encoding
        df_tracking["orientation_sin"] = np.sin(ori_rad)
        df_tracking["orientation_cos"] = np.cos(ori_rad)
        df_tracking["direction_sin"] = np.sin(dir_rad)
        df_tracking["direction_cos"] = np.cos(dir_rad)

        # Select columns to flatten (exclude raw angles)
        track_cols = [
            c for c in Config.RAW_TRACKING_COLS if c not in ["orientation", "direction"]
        ]
        track_cols += [
            "jerk",
            "pose_motion_alignment",
            "orientation_sin",
            "orientation_cos",
            "direction_sin",
            "direction_cos",
        ]

        df_flat_track = self._create_flattened_features(df_tracking, track_cols)

        # 3. Merge Tracking P1
        df_merged = pd.merge(
            df_meta,
            df_flat_track.add_suffix("_p1"),
            left_on=["game_play", "nfl_player_id_1", "step"],
            right_on=["game_play_p1", "nfl_player_id_p1", "step_p1"],
            how="left",
        )

        # 4. Merge Tracking P2
        df_merged = pd.merge(
            df_merged,
            df_flat_track.add_suffix("_p2"),
            left_on=["game_play", "nfl_player_id_2", "step"],
            right_on=["game_play_p2", "nfl_player_id_p2", "step_p2"],
            how="left",
        )

        # 5. Calculate Interaction Features (Instantaneous)
        # Using lag0 columns (suffix "")
        p1_x = df_merged["x_position_p1"]
        p1_y = df_merged["y_position_p1"]
        p2_x = df_merged["x_position_p2"]
        p2_y = df_merged["y_position_p2"]

        df_merged["distance"] = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)
        df_merged["speed_diff"] = np.abs(df_merged["speed_p1"] - df_merged["speed_p2"])

        # Cite solution_lesson_node_00005: Cosine Similarity for Angles
        df_merged["cos_sim_orientation"] = (
            df_merged["orientation_cos_p1"] * df_merged["orientation_cos_p2"]
            + df_merged["orientation_sin_p1"] * df_merged["orientation_sin_p2"]
        )
        df_merged["cos_sim_direction"] = (
            df_merged["direction_cos_p1"] * df_merged["direction_cos_p2"]
            + df_merged["direction_sin_p1"] * df_merged["direction_sin_p2"]
        )

        # 6. Process & Merge Visuals
        df_helmets_proc = self._prepare_helmets(df_helmets)

        # Merge P1 Helmets
        df_merged = pd.merge(
            df_merged,
            df_helmets_proc.add_suffix("_p1"),
            left_on=["game_play", "nfl_player_id_1", "step"],
            right_on=["game_play_p1", "nfl_player_id_p1", "step_p1"],
            how="left",
        )

        # Merge P2 Helmets
        df_merged = pd.merge(
            df_merged,
            df_helmets_proc.add_suffix("_p2"),
            left_on=["game_play", "nfl_player_id_2", "step"],
            right_on=["game_play_p2", "nfl_player_id_p2", "step_p2"],
            how="left",
        )

        # 7. Calculate Visual Geometry
        # Sideline
        s_box1 = df_merged[
            [
                "Sideline_left_p1",
                "Sideline_top_p1",
                "Sideline_width_p1",
                "Sideline_height_p1",
            ]
        ].values
        s_box2 = df_merged[
            [
                "Sideline_left_p2",
                "Sideline_top_p2",
                "Sideline_width_p2",
                "Sideline_height_p2",
            ]
        ].values

        df_merged["Sideline_IoU"] = self._calculate_iou(s_box1, s_box2)
        df_merged["Sideline_Dist"] = self._calculate_centroid_distance(s_box1, s_box2)

        # Endzone
        e_box1 = df_merged[
            [
                "Endzone_left_p1",
                "Endzone_top_p1",
                "Endzone_width_p1",
                "Endzone_height_p1",
            ]
        ].values
        e_box2 = df_merged[
            [
                "Endzone_left_p2",
                "Endzone_top_p2",
                "Endzone_width_p2",
                "Endzone_height_p2",
            ]
        ].values

        df_merged["Endzone_IoU"] = self._calculate_iou(e_box1, e_box2)
        df_merged["Endzone_Dist"] = self._calculate_centroid_distance(e_box1, e_box2)

        # 8. Sentinel Imputation for Visuals
        visual_cols = ["Sideline_IoU", "Sideline_Dist", "Endzone_IoU", "Endzone_Dist"]
        df_merged[visual_cols] = df_merged[visual_cols].fillna(Config.SENTINEL_VALUE)

        # 9. Final Selection
        # Identify all flattened tracking columns
        flat_cols_p1 = [
            c
            for c in df_merged.columns
            if c.endswith("_p1")
            and (
                "position" in c
                or "speed" in c
                or "direction" in c
                or "orientation" in c
                or "acceleration" in c
                or "sa" in c
                or "jerk" in c
                or "pose_motion_alignment" in c
            )
        ]
        flat_cols_p2 = [
            c
            for c in df_merged.columns
            if c.endswith("_p2")
            and (
                "position" in c
                or "speed" in c
                or "direction" in c
                or "orientation" in c
                or "acceleration" in c
                or "sa" in c
                or "jerk" in c
                or "pose_motion_alignment" in c
            )
        ]
        # Filter only tracking ones, not helmet ones
        flat_cols_p1 = [
            c for c in flat_cols_p1 if "Sideline" not in c and "Endzone" not in c
        ]
        flat_cols_p2 = [
            c for c in flat_cols_p2 if "Sideline" not in c and "Endzone" not in c
        ]

        derived_cols = [
            "distance",
            "speed_diff",
            "cos_sim_orientation",
            "cos_sim_direction",
        ]

        feature_cols = flat_cols_p1 + flat_cols_p2 + derived_cols + visual_cols

        X = df_merged[feature_cols].copy()
        y = df_merged["contact"].values
        ids = df_merged["contact_id"].values

        # Fill remaining NaNs (tracking gaps) with Sentinel
        X = X.fillna(Config.SENTINEL_VALUE)

        # Save to cache
        print(f"Saving Stream A features to {cache_path}...")
        X.to_parquet(cache_path, index=False)
        np.save(ids_path, ids)
        np.save(y_path, y)

        gc.collect()
        return X, ids, y

    def build_stream_b_features(self, split, load_cached_data=True):
        """
        Builds features for Stream B (Impact Model).
        Tracking (P1 Only) + Physics Derivatives. NO Visuals.
        """
        # Check cache
        cache_path = Config.get_feature_cache_path("streamB", split)
        ids_path = cache_path.replace(".parquet", "_ids.npy")
        y_path = cache_path.replace(".parquet", "_y.npy")

        if (
            load_cached_data
            and os.path.exists(cache_path)
            and os.path.exists(ids_path)
            and os.path.exists(y_path)
        ):
            print(f"Loading Stream B features from cache: {cache_path}")
            return (
                pd.read_parquet(cache_path),
                np.load(ids_path, allow_pickle=True),
                np.load(y_path),
            )

        print(f"Building Stream B features for {split}...")

        # 1. Load Data
        df_meta, df_tracking, _ = self.dm.get_data(split, load_cached_data)

        # Filter for Ground (P2 == G)
        df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        # 2. Enhance Tracking (Physics)
        # Cite solution_lesson_node_00027: Enrich Kinematic Models
        # Cite solution_lesson_node_00005: Cyclical Feature Encoding
        df_tracking = df_tracking.sort_values(["game_play", "nfl_player_id", "step"])

        # Jerk
        df_tracking["jerk"] = (
            df_tracking.groupby(["game_play", "nfl_player_id"])["acceleration"]
            .diff()
            .fillna(0)
        )

        # Pose-Motion Alignment
        ori_rad = np.radians(df_tracking["orientation"].fillna(0))
        dir_rad = np.radians(df_tracking["direction"].fillna(0))
        df_tracking["pose_motion_alignment"] = np.cos(ori_rad - dir_rad)

        # Kinetic Energy
        df_tracking["kinetic_energy"] = 0.5 * (df_tracking["speed"] ** 2)

        # Cyclical Encoding
        df_tracking["orientation_sin"] = np.sin(ori_rad)
        df_tracking["orientation_cos"] = np.cos(ori_rad)
        df_tracking["direction_sin"] = np.sin(dir_rad)
        df_tracking["direction_cos"] = np.cos(dir_rad)

        # 3. Flatten Tracking
        # Exclude raw angles, include processed features
        base_cols = [
            c for c in Config.RAW_TRACKING_COLS if c not in ["orientation", "direction"]
        ]
        base_cols += [
            "jerk",
            "pose_motion_alignment",
            "kinetic_energy",
            "orientation_sin",
            "orientation_cos",
            "direction_sin",
            "direction_cos",
        ]
        df_flat_track = self._create_flattened_features(df_tracking, base_cols)

        # 4. Merge
        df_merged = pd.merge(
            df_meta,
            df_flat_track,  # No suffix needed, only P1
            left_on=["game_play", "nfl_player_id_1", "step"],
            right_on=["game_play", "nfl_player_id", "step"],
            how="left",
        )

        # 5. Select Columns
        # All flattened columns from df_flat_track
        feature_cols = [
            c
            for c in df_flat_track.columns
            if c not in ["game_play", "nfl_player_id", "step"]
        ]

        X = df_merged[feature_cols].copy()
        y = df_merged["contact"].values
        ids = df_merged["contact_id"].values

        # Fill NaNs
        X = X.fillna(Config.SENTINEL_VALUE)

        # Save to cache
        print(f"Saving Stream B features to {cache_path}...")
        X.to_parquet(cache_path, index=False)
        np.save(ids_path, ids)
        np.save(y_path, y)

        gc.collect()
        return X, ids, y
