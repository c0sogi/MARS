import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import CacheManager
from library.data_loader import DataLoader


class FeatureEngine:
    """
    Implements the Biomechanical Invariance Dual-Stream feature engineering pipeline.
    Stream A: Collider (Interaction Model) - Max Context, Relative Geometry, Visuals.
    Stream B: Accelerometer (Impact Model) - Strict Invariance, Ego-Dynamics.
    """

    def __init__(self, debug=Config.DEBUG):
        self.debug = debug
        self.cache_manager = CacheManager()

    def _calculate_frame_from_step(self, step_series):
        """
        Maps tracking steps (10Hz, 0 at snap) to video frames (59.94Hz, snap at 5s).
        """
        # Snap is at 5.0s (approx frame 300)
        # Time = step * 0.1
        # Frame = (Time + 5.0) * 59.94
        return ((5.0 + step_series * 0.1) * 59.94).round().astype(int)

    def _compute_iou(self, box1, box2):
        """
        Vectorized IoU calculation for bounding boxes [left, width, top, height].
        """
        b1_x1, b1_y1 = box1[:, 0], box1[:, 2]
        b1_x2, b1_y2 = b1_x1 + box1[:, 1], b1_y1 + box1[:, 3]

        b2_x1, b2_y1 = box2[:, 0], box2[:, 2]
        b2_x2, b2_y2 = b2_x1 + box2[:, 1], b2_y1 + box2[:, 3]

        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h
        b1_area = box1[:, 1] * box1[:, 3]
        b2_area = box2[:, 1] * box2[:, 3]

        union_area = b1_area + b2_area - inter_area

        # Avoid division by zero
        return np.where(union_area > 0, inter_area / union_area, 0.0)

    def _compute_box_dist(self, box1, box2):
        """
        Vectorized centroid distance calculation.
        """
        c1_x = box1[:, 0] + box1[:, 1] / 2
        c1_y = box1[:, 2] + box1[:, 3] / 2

        c2_x = box2[:, 0] + box2[:, 1] / 2
        c2_y = box2[:, 2] + box2[:, 3] / 2

        return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    def _add_lags(self, df, feature_cols, lags, group_cols):
        """
        Generates flattened temporal pyramids (lags) for specified features.
        """
        # Ensure data is sorted for shifting
        df = df.sort_values(by=group_cols + ["step"])

        out_df = df.copy()
        grouper = out_df.groupby(group_cols)

        for lag in lags:
            if lag == 0:
                continue

            lag_suffix = f"_lag_{lag}"
            # Shift features
            shifted = grouper[feature_cols].shift(lag)
            shifted.columns = [f"{c}{lag_suffix}" for c in feature_cols]

            out_df = pd.concat([out_df, shifted], axis=1)

        # Fill NaNs created by lags with 0 (neutral for kinematics)
        # For visual features, -999 is handled before this or acceptable as 0 in tree models
        out_df = out_df.fillna(0)

        return out_df

    def compute_ego_dynamics(self, df_tracking):
        """
        Stream B: Computes Finite-Difference Ego-Dynamics (Surge, Sway, Jerk).
        Projects global velocity onto player orientation.
        """
        # Ensure sorted for finite difference
        df = df_tracking.sort_values(by=["game_play", "nfl_player_id", "step"]).copy()

        deg2rad = np.pi / 180.0

        # 1. Calculate Velocity Vector (Global)
        # Assuming 0=North(Y), 90=East(X) standard NFL orientation for projection logic
        v_x = df["speed"] * np.sin(df["direction"] * deg2rad)
        v_y = df["speed"] * np.cos(df["direction"] * deg2rad)

        # 2. Calculate Orientation Vector (Ego)
        o_x = np.sin(df["orientation"] * deg2rad)
        o_y = np.cos(df["orientation"] * deg2rad)

        # 3. Calculate Orthogonal Vector (Sway axis)
        # Orthogonal to (o_x, o_y) is (o_y, -o_x)
        s_x = o_y
        s_y = -o_x

        # 4. Project Velocity
        df["surge"] = v_x * o_x + v_y * o_y
        df["sway"] = v_x * s_x + v_y * s_y

        # 5. Finite Difference Derivatives
        # Group by player to isolate dynamics
        g = df.groupby(["game_play", "nfl_player_id"])
        dt = 0.1

        # Ego Acceleration
        df["ego_acc_surge"] = g["surge"].diff() / dt
        df["ego_acc_sway"] = g["sway"].diff() / dt

        # Ego Jerk (Shock)
        df["ego_jerk_surge"] = g["ego_acc_surge"].diff() / dt
        df["ego_jerk_sway"] = g["ego_acc_sway"].diff() / dt

        # Fill NaNs from diff
        df.fillna(0, inplace=True)

        return df

    def compute_relational_metrics(self, df_merged):
        """
        Stream A: Computes interaction metrics (Distance, Closure, Relative Angle).
        """
        df = df_merged.copy()

        # Euclidean Distance
        df["distance"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )

        # Relative Velocity Vector (P2 - P1)
        deg2rad = np.pi / 180.0

        v_x_p1 = df["speed_p1"] * np.sin(df["direction_p1"] * deg2rad)
        v_y_p1 = df["speed_p1"] * np.cos(df["direction_p1"] * deg2rad)

        v_x_p2 = df["speed_p2"] * np.sin(df["direction_p2"] * deg2rad)
        v_y_p2 = df["speed_p2"] * np.cos(df["direction_p2"] * deg2rad)

        rel_vx = v_x_p2 - v_x_p1
        rel_vy = v_y_p2 - v_y_p1

        # Relative Speed (Scalar)
        df["rel_speed"] = np.sqrt(rel_vx**2 + rel_vy**2)

        # Closure Rate (d(Dist)/dt)
        # Projection of relative velocity onto distance vector
        d_x = df["x_position_p2"] - df["x_position_p1"]
        d_y = df["y_position_p2"] - df["y_position_p1"]
        dist = df["distance"] + 1e-6

        # Negative when closing
        df["closure_rate"] = -(d_x * rel_vx + d_y * rel_vy) / dist

        # Ego-Relational Projections (Angle of Attack)
        # Project Relative Velocity onto P1's Orientation
        o_x_p1 = np.sin(df["orientation_p1"] * deg2rad)
        o_y_p1 = np.cos(df["orientation_p1"] * deg2rad)

        s_x_p1 = o_y_p1
        s_y_p1 = -o_x_p1

        df["rel_surge"] = rel_vx * o_x_p1 + rel_vy * o_y_p1
        df["rel_sway"] = rel_vx * s_x_p1 + rel_vy * s_y_p1

        return df

    def compute_visual_pyramids(self, df_main, df_helmets):
        """
        Stream A: Computes visual features (IoU, Distance) from helmet boxes.
        """
        if df_helmets is None or df_helmets.empty:
            for col in [
                "iou_sideline",
                "iou_endzone",
                "dist_sideline",
                "dist_endzone",
                "iou_diff",
            ]:
                df_main[col] = -999
            return df_main

        # Map step to frame
        df_main["frame"] = self._calculate_frame_from_step(df_main["step"])

        # Filter helmets to relevant plays
        relevant_plays = df_main["game_play"].unique()
        df_h = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # Helper to merge specific view
        def merge_view(df_m, h_view, suffix):
            # Prepare P1 boxes
            h_p1 = h_view[
                [
                    "game_play",
                    "frame",
                    "nfl_player_id",
                    "left",
                    "width",
                    "top",
                    "height",
                ]
            ].copy()
            h_p1["nfl_player_id"] = h_p1["nfl_player_id"].astype(str)
            h_p1.columns = [
                "game_play",
                "frame",
                "nfl_player_id_1",
                "l1",
                "w1",
                "t1",
                "h1",
            ]

            df_m = pd.merge(
                df_m, h_p1, on=["game_play", "frame", "nfl_player_id_1"], how="left"
            )

            # Prepare P2 boxes
            h_p2 = h_view[
                [
                    "game_play",
                    "frame",
                    "nfl_player_id",
                    "left",
                    "width",
                    "top",
                    "height",
                ]
            ].copy()
            h_p2["nfl_player_id"] = h_p2["nfl_player_id"].astype(str)
            h_p2.columns = [
                "game_play",
                "frame",
                "nfl_player_id_2",
                "l2",
                "w2",
                "t2",
                "h2",
            ]

            df_m = pd.merge(
                df_m, h_p2, on=["game_play", "frame", "nfl_player_id_2"], how="left"
            )

            # Compute Metrics
            box1 = df_m[["l1", "w1", "t1", "h1"]].fillna(0).values
            box2 = df_m[["l2", "w2", "t2", "h2"]].fillna(0).values

            iou = self._compute_iou(box1, box2)
            dist = self._compute_box_dist(box1, box2)

            # Impute missing with -999
            missing_mask = df_m["l1"].isna() | df_m["l2"].isna()

            df_m[f"iou_{suffix}"] = iou
            df_m[f"dist_{suffix}"] = dist
            df_m.loc[missing_mask, [f"iou_{suffix}", f"dist_{suffix}"]] = -999

            # Cleanup
            df_m.drop(
                columns=["l1", "w1", "t1", "h1", "l2", "w2", "t2", "h2"], inplace=True
            )
            return df_m

        # Process Sideline and Endzone
        h_side = df_h[df_h["view"] == "Sideline"]
        h_end = df_h[df_h["view"] == "Endzone"]

        df_main = merge_view(df_main, h_side, "sideline")
        df_main = merge_view(df_main, h_end, "endzone")

        # Consensus (Uncertainty)
        s_valid = df_main["iou_sideline"] != -999
        e_valid = df_main["iou_endzone"] != -999
        both_valid = s_valid & e_valid

        df_main["iou_diff"] = -999.0
        df_main.loc[both_valid, "iou_diff"] = (
            df_main.loc[both_valid, "iou_sideline"]
            - df_main.loc[both_valid, "iou_endzone"]
        ).abs()

        return df_main

    def generate_stream_a_features(
        self, df_labels, df_tracking, df_helmets, load_cached_data=True
    ):
        """
        Orchestrates Stream A (Collider) feature generation.
        """
        # Cache Check
        config_dict = {
            "stream": "A",
            "lags": Config.LAG_OFFSETS,
            "features": Config.STREAM_A_FEATURES,
            "labels_len": len(df_labels),
            "tracking_len": len(df_tracking),
            "helmets_len": len(df_helmets) if df_helmets is not None else 0,
        }
        cache_id = self.cache_manager.generate_cache_id(
            config_dict, prefix="features_streamA"
        )

        if load_cached_data:
            cached = self.cache_manager.load(cache_id)
            if cached is not None:
                return cached

        # 1. Merge Tracking
        dl = DataLoader(debug=self.debug)
        df_merged = dl.merge_tracking_to_labels(
            df_labels, df_tracking, load_cached_data=load_cached_data
        )

        # 2. Compute Relational Metrics
        df_feats = self.compute_relational_metrics(df_merged)

        # 3. Compute Visual Pyramids
        df_feats = self.compute_visual_pyramids(df_feats, df_helmets)

        # 4. Add Lags
        base_features = [f for f in Config.STREAM_A_FEATURES if f in df_feats.columns]
        df_feats = self._add_lags(
            df_feats,
            base_features,
            Config.LAG_OFFSETS,
            group_cols=["game_play", "nfl_player_id_1", "nfl_player_id_2"],
        )

        # 5. Select Columns
        final_cols = []
        for f in Config.STREAM_A_FEATURES:
            if f in df_feats.columns:
                final_cols.append(f)
                for lag in Config.LAG_OFFSETS:
                    if lag != 0:
                        final_cols.append(f"{f}_lag_{lag}")

        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        keep_cols = [c for c in meta_cols if c in df_feats.columns] + final_cols

        result = df_feats[keep_cols].copy()

        # Save to Cache
        self.cache_manager.save(result, cache_id)

        return result

    def generate_stream_b_features(self, df_labels, df_tracking, load_cached_data=True):
        """
        Orchestrates Stream B (Accelerometer) feature generation.
        """
        # Cache Check
        config_dict = {
            "stream": "B",
            "lags": Config.LAG_OFFSETS,
            "features": Config.STREAM_B_FEATURES,
            "labels_len": len(df_labels),
            "tracking_len": len(df_tracking),
        }
        cache_id = self.cache_manager.generate_cache_id(
            config_dict, prefix="features_streamB"
        )

        if load_cached_data:
            cached = self.cache_manager.load(cache_id)
            if cached is not None:
                return cached

        # 1. Pre-compute Ego Dynamics on Tracking (P1 only)
        # Filter tracking to relevant P1s to save memory
        p1_ids = df_labels["nfl_player_id_1"].unique()
        games = df_labels["game_play"].unique()

        df_track_filtered = df_tracking[
            df_tracking["game_play"].isin(games)
            & df_tracking["nfl_player_id"].isin(p1_ids.astype(str))
        ].copy()

        df_track_aug = self.compute_ego_dynamics(df_track_filtered)

        # 2. Merge onto Labels
        df_labels = df_labels.copy()
        df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
        df_track_aug["nfl_player_id"] = df_track_aug["nfl_player_id"].astype(str)

        df_merged = pd.merge(
            df_labels,
            df_track_aug,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 3. Add Lags
        # Group by P1 (since P2 is Ground)
        base_features = [f for f in Config.STREAM_B_FEATURES if f in df_merged.columns]
        df_merged = self._add_lags(
            df_merged,
            base_features,
            Config.LAG_OFFSETS,
            group_cols=["game_play", "nfl_player_id_1", "nfl_player_id_2"],
        )

        # 4. Select Columns
        final_cols = []
        for f in Config.STREAM_B_FEATURES:
            if f in df_merged.columns:
                final_cols.append(f)
                for lag in Config.LAG_OFFSETS:
                    if lag != 0:
                        final_cols.append(f"{f}_lag_{lag}")

        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
        ]
        keep_cols = [c for c in meta_cols if c in df_merged.columns] + final_cols

        result = df_merged[keep_cols].copy()

        # Save to Cache
        self.cache_manager.save(result, cache_id)

        return result
