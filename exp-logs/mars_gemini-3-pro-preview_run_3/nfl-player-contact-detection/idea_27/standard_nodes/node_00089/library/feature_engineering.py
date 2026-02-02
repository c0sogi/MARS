import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import (
    get_cache_path,
    save_cache,
    load_cache,
    check_cache_exists,
    validate_schema,
)
from library.data_loader import DataLoader


class FeatureEngineer:
    """
    Implements the Invariant-Physics Temporal Pyramid feature engineering pipeline.
    Separates processing into Stream A (Interactions) and Stream B (Ground Impacts).
    """

    def __init__(self, mode="train", debug=False):
        self.mode = mode
        self.debug = debug
        self.loader = DataLoader(mode=mode, debug=debug)

    def _apply_temporal_pyramid(self, df, feature_cols, lags):
        """
        Generates and flattens features across exponentially spaced lags.
        """
        # Sort to ensure temporal order
        df = df.sort_values(by=["game_play", "step"]).reset_index(drop=True)

        # Base features are already in df, we need to add lagged versions
        # We process by group to prevent bleeding between plays
        grouper = df.groupby("game_play")

        result_dfs = [df]  # Start with original features (lag 0)

        for lag in lags:
            # Lag t-k
            shifted_neg = grouper[feature_cols].shift(lag)
            shifted_neg.columns = [f"{col}_lag_{lag}" for col in feature_cols]

            # Lag t+k
            shifted_pos = grouper[feature_cols].shift(-lag)
            shifted_pos.columns = [f"{col}_lag_pos_{lag}" for col in feature_cols]

            result_dfs.extend([shifted_neg, shifted_pos])

        # Concatenate all features
        df_pyramid = pd.concat(result_dfs, axis=1)

        return df_pyramid

    def _compute_iou(self, box1, box2):
        """
        Vectorized IoU calculation.
        Box format: [left, width, top, height]
        """
        # box: x, w, y, h
        # Convert to x1, y1, x2, y2
        b1_x1, b1_x2 = box1[:, 0], box1[:, 0] + box1[:, 1]
        b1_y1, b1_y2 = box1[:, 2], box1[:, 2] + box1[:, 3]

        b2_x1, b2_x2 = box2[:, 0], box2[:, 0] + box2[:, 1]
        b2_y1, b2_y2 = box2[:, 2], box2[:, 2] + box2[:, 3]

        # Intersection
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        b1_area = box1[:, 1] * box1[:, 3]
        b2_area = box2[:, 1] * box2[:, 3]
        union_area = b1_area + b2_area - inter_area

        # IoU
        iou = np.zeros_like(union_area)
        mask = union_area > 0
        iou[mask] = inter_area[mask] / union_area[mask]

        return iou

    def _compute_visual_features(self, df_main, df_helmets):
        """
        Computes IoU features by mapping steps to frames.
        Approximation: frame = 300 + step * 6
        """
        if df_helmets is None or df_helmets.empty:
            # Return with sentinel values if no helmet data
            for col in [
                "sideline_iou",
                "endzone_iou",
                "max_iou",
                "min_iou",
                "iou_diff",
            ]:
                df_main[col] = -999.0
            df_main["looming_mismatch"] = 0.0
            return df_main

        # Map step to frame
        # Snap is 5s (300 frames), step 0 is snap. 1 step = 0.1s = 6 frames
        df_main["frame_approx"] = 300 + (df_main["step"] * 6).astype(int)

        # Prepare Helmets
        # We need efficient lookup. Pivot or Merge? Merge is safer.
        # Filter helmets to relevant game_plays
        relevant_plays = df_main["game_play"].unique()
        df_h = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # We need P1 and P2 boxes for both views
        # Views: Sideline, Endzone

        # Helper to merge specific view
        def get_view_boxes(view_name):
            view_h = df_h[df_h["view"] == view_name]
            if view_h.empty:
                return None

            # Merge for P1
            # Keys: game_play, frame, nfl_player_id
            p1_boxes = pd.merge(
                df_main[["game_play", "frame_approx", "nfl_player_id_1"]],
                view_h[
                    [
                        "game_play",
                        "frame",
                        "nfl_player_id",
                        "left",
                        "width",
                        "top",
                        "height",
                    ]
                ],
                left_on=["game_play", "frame_approx", "nfl_player_id_1"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )

            # Merge for P2
            p2_boxes = pd.merge(
                df_main[["game_play", "frame_approx", "nfl_player_id_2_numeric"]],
                view_h[
                    [
                        "game_play",
                        "frame",
                        "nfl_player_id",
                        "left",
                        "width",
                        "top",
                        "height",
                    ]
                ],
                left_on=["game_play", "frame_approx", "nfl_player_id_2_numeric"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )

            # Extract arrays
            b1 = p1_boxes[["left", "width", "top", "height"]].fillna(0).values
            b2 = p2_boxes[["left", "width", "top", "height"]].fillna(0).values

            # If any box is all 0, IoU is 0 (or sentinel)
            # We used fillna(0), so _compute_iou handles it (area 0 -> iou 0)
            return self._compute_iou(b1, b2)

        # Compute IoUs
        # P2 ID needs to be numeric for merge
        df_main["nfl_player_id_2_numeric"] = pd.to_numeric(
            df_main["nfl_player_id_2"], errors="coerce"
        )

        sideline_iou = get_view_boxes("Sideline")
        endzone_iou = get_view_boxes("Endzone")

        # Assign
        df_main["sideline_iou"] = sideline_iou if sideline_iou is not None else -999.0
        df_main["endzone_iou"] = endzone_iou if endzone_iou is not None else -999.0

        # Fill missing merges with sentinel
        df_main["sideline_iou"] = df_main["sideline_iou"].fillna(-999.0)
        df_main["endzone_iou"] = df_main["endzone_iou"].fillna(-999.0)

        # Aggregates
        # Handle sentinels for max/min logic
        s_valid = df_main["sideline_iou"] != -999
        e_valid = df_main["endzone_iou"] != -999

        # Default to 0 if valid, else -999
        df_main["max_iou"] = np.maximum(
            np.where(s_valid, df_main["sideline_iou"], -1),
            np.where(e_valid, df_main["endzone_iou"], -1),
        )
        df_main["max_iou"] = np.where(
            df_main["max_iou"] == -1, -999, df_main["max_iou"]
        )

        df_main["min_iou"] = np.minimum(
            np.where(s_valid, df_main["sideline_iou"], 1.1),
            np.where(e_valid, df_main["endzone_iou"], 1.1),
        )
        df_main["min_iou"] = np.where(
            df_main["min_iou"] == 1.1, -999, df_main["min_iou"]
        )

        df_main["iou_diff"] = np.abs(df_main["sideline_iou"] - df_main["endzone_iou"])
        df_main.loc[(~s_valid) | (~e_valid), "iou_diff"] = -999.0

        # Looming Mismatch: Diff between kinematic closure and visual looming
        # Visual looming rate ~ diff of IoU over time.
        # For simplicity in this iteration, we compare IoU magnitude to physical distance
        # normalized: (1 - IoU) vs (Distance / Max_Dist).
        # Here we just set a placeholder or simple interaction term as requested by prompt
        df_main["looming_mismatch"] = df_main["closure_rate"] * df_main["max_iou"]

        # Cleanup
        df_main.drop(columns=["frame_approx", "nfl_player_id_2_numeric"], inplace=True)
        return df_main

    def construct_stream_a(self, load_cached_data=True):
        """
        Constructs features for Stream A (Interaction Model).
        Target: Player-Player Contact.
        """
        cache_params = {"mode": self.mode, "debug": self.debug, "stream": "A"}
        cache_path_X = get_cache_path("features_streamA_X", cache_params, ".parquet")
        cache_path_y = get_cache_path("features_streamA_y", cache_params, ".npy")
        cache_path_ids = get_cache_path("features_streamA_ids", cache_params, ".npy")

        if load_cached_data and check_cache_exists(cache_path_X):
            print("Loading Stream A features from cache...")
            return (
                load_cache(cache_path_X),
                load_cache(cache_path_y),
                load_cache(cache_path_ids),
            )

        print("Constructing Stream A features...")

        # 1. Load Data
        df_meta = self.loader.load_metadata()

        # Filter for Player-Player interactions
        df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        if df_meta.empty:
            print("Warning: No Player-Player interactions found.")
            return pd.DataFrame(), np.array([]), np.array([])

        game_plays = df_meta["game_play"].unique()
        df_tracking = self.loader.load_tracking_data(game_plays, load_cached_data)
        df_helmets = self.loader.load_helmet_data(game_plays, load_cached_data)

        # 2. Merge Tracking
        df = self.loader.merge_labels_with_tracking(df_meta, df_tracking)

        # 3. Compute Relational Features
        # Distance
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        df["distance"] = np.sqrt(dx**2 + dy**2)

        # Closure Rate (Finite Difference)
        # Sort first
        df = df.sort_values(by=["game_play", "step"])
        df["closure_rate"] = (
            df.groupby("game_play")["distance"].diff().fillna(0) * -1
        )  # Positive = closing

        # Rename tracking cols to match Config.STREAM_A_FEATURES
        # Tracking merge gave us speed_p1, speed_p2, acceleration_p1, etc.
        # These match Config requirements directly.

        # 4. Compute Visual Features
        df = self._compute_visual_features(df, df_helmets)

        # 5. Apply Temporal Pyramid
        # Features to lag
        features_to_lag = [
            "distance",
            "closure_rate",
            "speed_p1",
            "speed_p2",
            "acceleration_p1",
            "acceleration_p2",
            "sideline_iou",
            "endzone_iou",
            "max_iou",
            "min_iou",
            "iou_diff",
            "looming_mismatch",
        ]

        # Ensure all columns exist
        for col in features_to_lag:
            if col not in df.columns:
                df[col] = 0.0

        df_pyramid = self._apply_temporal_pyramid(df, features_to_lag, Config.LAGS)

        # 6. Finalize X, y
        # Identify feature columns (original + lagged)
        feature_cols = []
        for col in features_to_lag:
            feature_cols.append(col)
            for lag in Config.LAGS:
                feature_cols.append(f"{col}_lag_{lag}")
                feature_cols.append(f"{col}_lag_pos_{lag}")

        X = df_pyramid[feature_cols].copy()
        y = df_pyramid["contact"].values
        ids = df_pyramid["contact_id"].values

        # Fill NaNs (XGBoost handles them, but for safety with lags at start of play)
        # Visual features have -999 sentinel, others can be 0 or NaN.
        # We leave as NaN for XGBoost to learn "missing data" (e.g. start of play)

        # 7. Cache
        print(f"Stream A Shape: {X.shape}")
        save_cache(X, cache_path_X)
        save_cache(y, cache_path_y)
        save_cache(ids, cache_path_ids)

        return X, y, ids

    def construct_stream_b(self, load_cached_data=True):
        """
        Constructs features for Stream B (Impact Model).
        Target: Player-Ground Contact.
        Innovation: Invariant Ego-Centric Kinematics (Surge/Sway).
        """
        cache_params = {"mode": self.mode, "debug": self.debug, "stream": "B"}
        cache_path_X = get_cache_path("features_streamB_X", cache_params, ".parquet")
        cache_path_y = get_cache_path("features_streamB_y", cache_params, ".npy")
        cache_path_ids = get_cache_path("features_streamB_ids", cache_params, ".npy")

        if load_cached_data and check_cache_exists(cache_path_X):
            print("Loading Stream B features from cache...")
            return (
                load_cache(cache_path_X),
                load_cache(cache_path_y),
                load_cache(cache_path_ids),
            )

        print("Constructing Stream B features...")

        # 1. Load Data
        df_meta = self.loader.load_metadata()

        # Filter for Player-Ground interactions
        df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        if df_meta.empty:
            print("Warning: No Player-Ground interactions found.")
            return pd.DataFrame(), np.array([]), np.array([])

        game_plays = df_meta["game_play"].unique()
        df_tracking = self.loader.load_tracking_data(game_plays, load_cached_data)

        # 2. Merge Tracking (Only P1 matters)
        df = self.loader.merge_labels_with_tracking(df_meta, df_tracking)

        # 3. Compute Invariant Ego-Centric Kinematics
        # Inputs: speed_p1, direction_p1, orientation_p1, acceleration_p1

        # Convert angles to radians
        # NFL Tracking: 0=Y, 90=X.
        # We need relative angle: (Direction - Orientation)
        # This works regardless of the coordinate system as long as both are consistent.

        dir_rad = np.radians(df["direction_p1"].fillna(0))
        ori_rad = np.radians(df["orientation_p1"].fillna(0))
        theta_diff = dir_rad - ori_rad

        # Velocity Projections
        # Surge: Forward velocity relative to body
        # Sway: Lateral velocity relative to body
        # Assuming standard trig convention relative to body axis: Cosine=Parallel, Sine=Perpendicular
        df["v_surge"] = df["speed_p1"] * np.cos(theta_diff)
        df["v_sway"] = df["speed_p1"] * np.sin(theta_diff)

        # Acceleration Projections
        # We can project the raw acceleration magnitude if we assume it aligns with direction,
        # OR use finite differences of V_surge/V_sway.
        # Finite difference is more robust to "change in motion" (Jerk-like features).
        # Let's use finite differences as per "Calculate Ego-Acceleration via finite difference"

        df = df.sort_values(by=["game_play", "step"])
        grouper = df.groupby("game_play")

        df["a_surge"] = grouper["v_surge"].diff().fillna(0)
        df["a_sway"] = grouper["v_sway"].diff().fillna(0)

        # Rename for consistency with Config
        df["speed"] = df["speed_p1"]
        df["acceleration"] = df["acceleration_p1"]

        # 4. Apply Temporal Pyramid
        features_to_lag = [
            "speed",
            "acceleration",
            "v_surge",
            "v_sway",
            "a_surge",
            "a_sway",
        ]

        df_pyramid = self._apply_temporal_pyramid(df, features_to_lag, Config.LAGS)

        # 5. Finalize X, y
        feature_cols = []
        for col in features_to_lag:
            feature_cols.append(col)
            for lag in Config.LAGS:
                feature_cols.append(f"{col}_lag_{lag}")
                feature_cols.append(f"{col}_lag_pos_{lag}")

        X = df_pyramid[feature_cols].copy()
        y = df_pyramid["contact"].values
        ids = df_pyramid["contact_id"].values

        # 6. Cache
        print(f"Stream B Shape: {X.shape}")
        save_cache(X, cache_path_X)
        save_cache(y, cache_path_y)
        save_cache(ids, cache_path_ids)

        return X, y, ids
