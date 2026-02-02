import os
import numpy as np
import pandas as pd
import gc
from library.config import Config
from library.utils import get_config_hash


class FeatureGenerator:
    """
    Generates features for the Ego-Centric Dual-Stream GBDT architecture.
    Handles Stream A (Interaction) and Stream B (Impact) separately with caching.
    """

    def __init__(self, run_mode="train"):
        self.run_mode = run_mode
        self.cache_dir = Config.WORKING_DIR
        self.lags = Config.LAGS

    def generate_features(self, df, stream="stream_a", load_cached_data=True):
        """
        Main entry point to generate features for a specific stream.

        Args:
            df (pd.DataFrame): Merged dataframe containing labels, tracking, and helmets.
            stream (str): 'stream_a' (Interaction) or 'stream_b' (Impact).
            load_cached_data (bool): Whether to use disk caching.

        Returns:
            tuple: (X, y, ids)
                X (pd.DataFrame): Feature matrix.
                y (np.ndarray): Target labels.
                ids (np.ndarray): Contact IDs.
        """
        # 1. Determine Configuration and Hash
        if stream == "stream_a":
            config_dict = Config.STREAM_A_CONFIG
            # Stream A focuses on Player-Player interactions
            # Filter: Player 2 is NOT Ground
            mask = df["nfl_player_id_2"] != "G"
        elif stream == "stream_b":
            config_dict = Config.STREAM_B_CONFIG
            # Stream B focuses on Player-Ground impacts
            # Filter: Player 2 IS Ground
            mask = df["nfl_player_id_2"] == "G"
        else:
            raise ValueError(f"Unknown stream: {stream}")

        # Create a unique hash for caching based on config, lags, and run mode
        cache_key = {
            "config": config_dict,
            "lags": self.lags,
            "run_mode": self.run_mode,
            "stream": stream,
        }
        config_hash = get_config_hash(cache_key)

        cache_path_X = os.path.join(
            self.cache_dir, f"features_{self.run_mode}_{stream}_{config_hash}_X.parquet"
        )
        cache_path_y = os.path.join(
            self.cache_dir, f"features_{self.run_mode}_{stream}_{config_hash}_y.npy"
        )
        cache_path_ids = os.path.join(
            self.cache_dir, f"features_{self.run_mode}_{stream}_{config_hash}_ids.npy"
        )

        # 2. Try Loading from Cache
        if (
            load_cached_data
            and os.path.exists(cache_path_X)
            and os.path.exists(cache_path_y)
        ):
            print(f"[{stream}] Loading features from cache: {cache_path_X}")
            X = pd.read_parquet(cache_path_X)
            y = np.load(cache_path_y)
            ids = np.load(cache_path_ids)
            return X, y, ids

        # 3. Compute Features (Cache Miss)
        print(f"[{stream}] Computing features from scratch...")

        # Apply filter mask
        df_filtered = df[mask].copy()

        # Sort for temporal operations
        df_filtered.sort_values(by=["game_play", "step"], inplace=True)
        df_filtered.reset_index(drop=True, inplace=True)

        if stream == "stream_a":
            X = self._process_stream_a(df_filtered)
        else:
            X = self._process_stream_b(df_filtered)

        y = df_filtered["contact"].values.astype(np.int8)
        ids = df_filtered["contact_id"].values

        # 4. Save to Cache
        print(f"[{stream}] Saving features to cache: {cache_path_X}")
        X.to_parquet(cache_path_X, index=False)
        np.save(cache_path_y, y)
        np.save(cache_path_ids, ids)

        # Cleanup
        del df_filtered
        gc.collect()

        return X, y, ids

    def _process_stream_a(self, df):
        """
        Generates features for Stream A: Interaction Model (Player-Player).
        Focus: Relational Tracking + Visual Geometry.
        """
        features = pd.DataFrame(index=df.index)

        # --- 1. Relational Tracking Features ---
        # Distance
        dx = df["x_position_p1"] - df["x_position_p2"]
        dy = df["y_position_p1"] - df["y_position_p2"]
        features["dist_p1_p2"] = np.sqrt(dx**2 + dy**2)

        # Relative Speed
        # We can use vector subtraction of velocities if available, or scalar diff
        # Let's derive velocity vectors first
        v_x_p1, v_y_p1 = self._get_velocity_vectors(df["speed_p1"], df["direction_p1"])
        v_x_p2, v_y_p2 = self._get_velocity_vectors(df["speed_p2"], df["direction_p2"])

        dv_x = v_x_p1 - v_x_p2
        dv_y = v_y_p1 - v_y_p2
        features["rel_speed"] = np.sqrt(dv_x**2 + dv_y**2)

        # Orientation Diff
        features["orient_diff"] = np.abs(df["orientation_p1"] - df["orientation_p2"])
        features["orient_diff"] = np.minimum(
            features["orient_diff"], 360 - features["orient_diff"]
        )

        # --- 2. Visual Features (IoU) ---
        # Sideline
        features["view_sideline_iou"] = self._compute_iou(
            df["view_sideline_left_p1"],
            df["view_sideline_top_p1"],
            df["view_sideline_width_p1"],
            df["view_sideline_height_p1"],
            df["view_sideline_left_p2"],
            df["view_sideline_top_p2"],
            df["view_sideline_width_p2"],
            df["view_sideline_height_p2"],
        )
        # Endzone
        features["view_endzone_iou"] = self._compute_iou(
            df["view_endzone_left_p1"],
            df["view_endzone_top_p1"],
            df["view_endzone_width_p1"],
            df["view_endzone_height_p1"],
            df["view_endzone_left_p2"],
            df["view_endzone_top_p2"],
            df["view_endzone_width_p2"],
            df["view_endzone_height_p2"],
        )

        # Fill Missing Visuals (e.g. if helmet not detected)
        features.fillna(-1, inplace=True)

        # --- 3. Temporal Pyramids ---
        # Columns to lag
        cols_to_lag = [
            "dist_p1_p2",
            "rel_speed",
            "view_sideline_iou",
            "view_endzone_iou",
        ]

        X_final = self._apply_temporal_pyramids(features, df["game_play"], cols_to_lag)

        return X_final.astype(np.float32)

    def _process_stream_b(self, df):
        """
        Generates features for Stream B: Impact Model (Player-Ground).
        Focus: Ego-Centric Kinematics (Surge/Sway).
        """
        # We only care about Player 1 (Player 2 is Ground)

        # --- 1. Derive Kinematics ---
        # Convert Speed/Direction to Cartesian Velocity
        vx, vy = self._get_velocity_vectors(df["speed_p1"], df["direction_p1"])

        # Assign to DataFrame for robust grouping
        df["vx"] = vx
        df["vy"] = vy

        # Derive Acceleration (Discrete Diff)
        # Group by game_play to avoid boundary issues
        # Note: We assume data is sorted by game_play, step
        groups = df.groupby("game_play")

        # Use transform to maintain index alignment
        ax = groups["vx"].transform(lambda x: x.diff().fillna(0))
        ay = groups["vy"].transform(lambda x: x.diff().fillna(0))

        # Derive Jerk
        df["ax"] = ax
        df["ay"] = ay

        jx = groups["ax"].transform(lambda x: x.diff().fillna(0))
        jy = groups["ay"].transform(lambda x: x.diff().fillna(0))

        # --- 2. Ego-Centric Projection ---
        # We project motion onto the player's CURRENT orientation (t=0 for the window)
        # Orientation is in degrees. NFL: 0=North(Y), 90=East(X).
        # Unit vector for orientation: (sin(theta), cos(theta))

        theta_rad = np.radians(df["orientation_p1"])
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        # Surge: Parallel to orientation (v dot u)
        # u = (sin, cos) for 0=North
        # v = (vx, vy)
        # surge = vx * sin + vy * cos
        surge_v = vx * sin_theta + vy * cos_theta
        surge_a = ax * sin_theta + ay * cos_theta
        surge_j = jx * sin_theta + jy * cos_theta

        # Sway: Perpendicular to orientation (v dot u_perp)
        # u_perp = (cos, -sin)  (Right hand rule / 90 deg rotation)
        # sway = vx * cos - vy * sin
        sway_v = vx * cos_theta - vy * sin_theta
        sway_a = ax * cos_theta - ay * sin_theta
        sway_j = jx * cos_theta - jy * sin_theta

        features = pd.DataFrame(
            {
                "surge_v": surge_v,
                "sway_v": sway_v,
                "surge_a": surge_a,
                "sway_a": sway_a,
                "surge_j": surge_j,
                "sway_j": sway_j,
            },
            index=df.index,
        )

        # --- 3. Temporal Pyramids ---
        # For Stream B, we want the history of these ego-features.
        # Note: The projection above uses CURRENT orientation.
        # Ideally, for a lag t-k, we should project v_(t-k) onto orientation_t.
        # The current implementation calculates v_t projected on o_t.
        # When we shift this feature, we get (v_(t-k) projected on o_(t-k)).
        # This is "Ego-motion at that time". This is a valid feature: "Was I moving sideways 1 sec ago?"
        # The prompt mentions "Project ... into ... Coordinate System... invariant".
        # Using historical ego-motion is the standard interpretation of "Ego-Centric Features".

        cols_to_lag = ["surge_v", "sway_v", "surge_a", "sway_a", "surge_j", "sway_j"]
        X_final = self._apply_temporal_pyramids(features, df["game_play"], cols_to_lag)

        return X_final.astype(np.float32)

    def _apply_temporal_pyramids(self, feature_df, group_col, cols):
        """
        Flattens features across exponential time lags.
        """
        # Base features (lag 0)
        result = feature_df[cols].copy()
        result.columns = [f"{c}_lag0" for c in cols]

        # We need to perform shifts respecting the group (game_play)
        # To make this efficient, we can rely on the fact that data is sorted by game_play, step.
        # We can just shift and then mask out boundaries where game_play changes.

        # However, pandas groupby shift is safer and reasonably fast for this size.
        grouped = feature_df[cols].groupby(group_col)

        for lag in self.lags:
            if lag == 0:
                continue

            # Past (t - lag)
            shifted_past = grouped.shift(lag)
            shifted_past.columns = [f"{c}_lag_neg{lag}" for c in cols]

            # Future (t + lag)
            shifted_future = grouped.shift(-lag)
            shifted_future.columns = [f"{c}_lag_pos{lag}" for c in cols]

            result = pd.concat([result, shifted_past, shifted_future], axis=1)

        # Fill NaNs created by shifting (edges of plays) with 0
        result.fillna(0, inplace=True)
        return result

    def _get_velocity_vectors(self, speed, direction):
        """
        Converts speed and direction (degrees, 0=N, CW) to vx, vy.
        """
        # NFL Tracking: 0 is North (Y+), 90 is East (X+)
        # angle in rads
        rads = np.radians(direction)
        # vx = speed * sin(theta)
        vx = speed * np.sin(rads)
        # vy = speed * cos(theta)
        vy = speed * np.cos(rads)
        return vx, vy

    def _compute_iou(self, left1, top1, w1, h1, left2, top2, w2, h2):
        """
        Vectorized IoU computation.
        """
        # Right and Bottom coordinates
        right1 = left1 + w1
        bottom1 = top1 + h1
        right2 = left2 + w2
        bottom2 = top2 + h2

        # Intersection coordinates
        xi1 = np.maximum(left1, left2)
        yi1 = np.maximum(top1, top2)
        xi2 = np.minimum(right1, right2)
        yi2 = np.minimum(bottom1, bottom2)

        inter_width = np.maximum(0, xi2 - xi1)
        inter_height = np.maximum(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # Avoid division by zero
        iou = np.where(union_area > 0, inter_area / union_area, 0)
        return iou
