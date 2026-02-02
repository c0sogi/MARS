import os
import numpy as np
import pandas as pd
from library.config import (
    WORKING_DIR,
    TRACKING_DELTAS,
    VISUAL_DELTAS,
    TRACKING_BASE_COLS,
    VISUAL_BASE_COLS,
    INTERACTION_COLS,
    KINEMATIC_DERIVATIVES,
    STREAM_A_FEATURES,
    STREAM_B_FEATURES,
)


class FeatureEngineer:
    """
    Engineers features for the contact detection model, including:
    1. Kinematic derivatives (Jerk, Alignment)
    2. Visual metrics (IoU, Centroid Distance)
    3. Interaction metrics (Distance, Speed Diff)
    4. Multi-Modal Exponential Temporal Pyramids (Lags)
    """

    def __init__(self):
        self.tracking_deltas = TRACKING_DELTAS
        self.visual_deltas = VISUAL_DELTAS

    def _calculate_iou(self, df, view):
        """Calculates Intersection over Union (IoU) for a specific view."""
        # Extract coordinates
        p1_left = df[f"p1_{view}_left"]
        p1_top = df[f"p1_{view}_top"]
        p1_width = df[f"p1_{view}_width"]
        p1_height = df[f"p1_{view}_height"]
        p1_right = p1_left + p1_width
        p1_bottom = p1_top + p1_height

        p2_left = df[f"p2_{view}_left"]
        p2_top = df[f"p2_{view}_top"]
        p2_width = df[f"p2_{view}_width"]
        p2_height = df[f"p2_{view}_height"]
        p2_right = p2_left + p2_width
        p2_bottom = p2_top + p2_height

        # Identify invalid boxes (sentinel -999)
        invalid_mask = (p1_left == -999) | (p2_left == -999)

        # Calculate Intersection
        x_left = np.maximum(p1_left, p2_left)
        y_top = np.maximum(p1_top, p2_top)
        x_right = np.minimum(p1_right, p2_right)
        y_bottom = np.minimum(p1_bottom, p2_bottom)

        intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
            0, y_bottom - y_top
        )

        # Calculate Union
        p1_area = p1_width * p1_height
        p2_area = p2_width * p2_height
        union_area = p1_area + p2_area - intersection_area

        # Compute IoU
        iou = intersection_area / np.maximum(1e-6, union_area)

        # Set invalid entries to -1 (distinct from 0 overlap)
        iou[invalid_mask] = -1

        return iou

    def _calculate_centroid_dist(self, df, view):
        """Calculates Euclidean distance between centroids in pixel space."""
        p1_cx = df[f"p1_{view}_left"] + df[f"p1_{view}_width"] / 2
        p1_cy = df[f"p1_{view}_top"] + df[f"p1_{view}_height"] / 2

        p2_cx = df[f"p2_{view}_left"] + df[f"p2_{view}_width"] / 2
        p2_cy = df[f"p2_{view}_top"] + df[f"p2_{view}_height"] / 2

        dist = np.sqrt((p1_cx - p2_cx) ** 2 + (p1_cy - p2_cy) ** 2)

        # Mask invalid
        invalid_mask = (df[f"p1_{view}_left"] == -999) | (df[f"p2_{view}_left"] == -999)
        dist[invalid_mask] = -999

        return dist

    def add_visual_metrics(self, df):
        """Adds IoU and Distance features for Sideline and Endzone views."""
        for view in ["sideline", "endzone"]:
            df[f"{view}_iou"] = self._calculate_iou(df, view)
            df[f"{view}_dist"] = self._calculate_centroid_dist(df, view)
        return df

    def add_kinematics(self, df):
        """Adds derived kinematic features: Jerk and Pose-Motion Alignment."""
        # Ensure data is sorted for temporal derivatives
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

        # Group by contact pair to respect boundaries
        grp = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        # 1. Jerk (Derivative of Acceleration) for P1
        # da/dt approx (a_t - a_{t-1}) / 0.1s
        p1_acc_prev = grp["p1_acceleration"].shift(1)
        df["p1_jerk"] = (df["p1_acceleration"] - p1_acc_prev) / 0.1
        df["p1_jerk"] = df["p1_jerk"].fillna(0)

        # 2. Pose-Motion Alignment for P1
        # Dot product of orientation and direction vectors
        p1_ori_rad = np.radians(df["p1_orientation"])
        p1_dir_rad = np.radians(df["p1_direction"])
        df["p1_pose_motion_alignment"] = np.cos(p1_ori_rad - p1_dir_rad)

        # 3. Angular Velocity for P1
        # d(theta)/dt with wrap-around handling
        p1_ori = df["p1_orientation"]
        p1_ori_prev = grp["p1_orientation"].shift(1)
        ori_diff = p1_ori - p1_ori_prev
        # Map to [-180, 180]
        ori_diff = (ori_diff + 180) % 360 - 180
        df["p1_angular_velocity"] = ori_diff / 0.1
        df["p1_angular_velocity"] = df["p1_angular_velocity"].fillna(0)

        return df

    def add_interaction_features(self, df):
        """Adds relative features between P1 and P2."""
        # Euclidean Distance (Tracking)
        df["distance"] = np.sqrt(
            (df["p1_x_position"] - df["p2_x_position"]) ** 2
            + (df["p1_y_position"] - df["p2_y_position"]) ** 2
        )

        # Speed Difference (Scalar)
        df["speed_diff"] = df["p1_speed"] - df["p2_speed"]

        # Direction Difference (Smallest angular difference)
        diff = np.abs(df["p1_direction"] - df["p2_direction"])
        df["direction_diff"] = np.minimum(diff, 360 - diff)

        # Orientation Cosine Similarity (Alignment)
        p1_rad = np.radians(df["p1_orientation"])
        p2_rad = np.radians(df["p2_orientation"])
        df["orientation_cos_diff"] = np.cos(p1_rad - p2_rad)

        # Relative Speed (Vector Magnitude)
        # Convert polar (speed, direction) to Cartesian (vx, vy)
        # Tracking data: 0=Y axis (North), increasing clockwise.
        # vx = speed * sin(rad), vy = speed * cos(rad)
        p1_dir_rad = np.radians(df["p1_direction"])
        p2_dir_rad = np.radians(df["p2_direction"])

        p1_vx = df["p1_speed"] * np.sin(p1_dir_rad)
        p1_vy = df["p1_speed"] * np.cos(p1_dir_rad)
        p2_vx = df["p2_speed"] * np.sin(p2_dir_rad)
        p2_vy = df["p2_speed"] * np.cos(p2_dir_rad)

        df["rel_speed"] = np.sqrt((p1_vx - p2_vx) ** 2 + (p1_vy - p2_vy) ** 2)

        # Handle Ground contacts (P2='G') where P2 features are sentinels
        mask = (df["nfl_player_id_2"] == "G") | (df["p2_x_position"] == -999)
        cols_to_mask = [
            "distance",
            "speed_diff",
            "direction_diff",
            "orientation_cos_diff",
            "rel_speed",
        ]
        df.loc[mask, cols_to_mask] = -999

        return df

    def apply_temporal_pyramid(self, df):
        """Flattens features at exponentially spaced temporal lags."""
        # Ensure sort order
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])
        grp = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        def create_lags(cols, deltas):
            for col in cols:
                if col not in df.columns:
                    continue
                for delta in deltas:
                    if delta == 0:
                        continue  # Base column exists

                    # Naming convention: col_lag{delta}
                    # delta > 0 is past (shift positive), delta < 0 is future (shift negative)
                    col_name = f"{col}_lag{delta}"
                    df[col_name] = grp[col].shift(delta)
                    df[col_name] = df[col_name].fillna(-999)

        # 1. Tracking Lags (P1 & P2)
        p1_track_cols = [f"p1_{c}" for c in TRACKING_BASE_COLS]
        p2_track_cols = [f"p2_{c}" for c in TRACKING_BASE_COLS]
        create_lags(p1_track_cols, self.tracking_deltas)
        create_lags(p2_track_cols, self.tracking_deltas)

        # 2. Interaction Lags
        create_lags(INTERACTION_COLS, self.tracking_deltas)

        # 3. Kinematic Lags (P1 only)
        p1_kin_cols = [f"p1_{c}" for c in KINEMATIC_DERIVATIVES]
        create_lags(p1_kin_cols, self.tracking_deltas)

        # 4. Visual Lags
        create_lags(VISUAL_BASE_COLS, self.visual_deltas)

        return df

    def process(self, df):
        """Executes the full feature engineering pipeline."""
        df = self.add_kinematics(df)
        df = self.add_visual_metrics(df)
        df = self.add_interaction_features(df)
        df = self.apply_temporal_pyramid(df)
        return df


def split_streams(df):
    """
    Splits the dataframe into Stream A (Player-Player) and Stream B (Player-Ground).
    Selects only the relevant features for each stream as defined in config.
    """
    # Stream B: Player 2 is Ground ('G')
    mask_b = df["nfl_player_id_2"] == "G"
    mask_a = ~mask_b

    # Split DataFrames
    df_a = df[mask_a].copy()
    df_b = df[mask_b].copy()

    # --- Prepare Stream A ---
    # Ensure all expected columns exist (fill missing with sentinel if any)
    for col in STREAM_A_FEATURES:
        if col not in df_a.columns:
            df_a[col] = -999

    X_a = df_a[STREAM_A_FEATURES]
    y_a = df_a["contact"].values
    ids_a = df_a["contact_id"].values

    # --- Prepare Stream B ---
    for col in STREAM_B_FEATURES:
        if col not in df_b.columns:
            df_b[col] = -999

    X_b = df_b[STREAM_B_FEATURES]
    y_b = df_b["contact"].values
    ids_b = df_b["contact_id"].values

    return {
        "stream_a": {"X": X_a, "y": y_a, "ids": ids_a},
        "stream_b": {"X": X_b, "y": y_b, "ids": ids_b},
    }


def generate_features(df, mode="train", load_cached_data=True):
    """
    Main entry point for feature generation.
    Handles caching of the split streams to disk.
    """
    # Define cache paths
    cache_a_X = os.path.join(WORKING_DIR, f"features_{mode}_streamA_X.parquet")
    cache_a_y = os.path.join(WORKING_DIR, f"features_{mode}_streamA_y.npy")
    cache_a_ids = os.path.join(WORKING_DIR, f"features_{mode}_streamA_ids.npy")

    cache_b_X = os.path.join(WORKING_DIR, f"features_{mode}_streamB_X.parquet")
    cache_b_y = os.path.join(WORKING_DIR, f"features_{mode}_streamB_y.npy")
    cache_b_ids = os.path.join(WORKING_DIR, f"features_{mode}_streamB_ids.npy")

    # Check if all cache files exist
    files_exist = all(
        os.path.exists(p)
        for p in [cache_a_X, cache_a_y, cache_a_ids, cache_b_X, cache_b_y, cache_b_ids]
    )

    if load_cached_data and files_exist:
        print(f"Loading cached features for {mode}...")
        return {
            "stream_a": {
                "X": pd.read_parquet(cache_a_X),
                "y": np.load(cache_a_y),
                "ids": np.load(cache_a_ids),
            },
            "stream_b": {
                "X": pd.read_parquet(cache_b_X),
                "y": np.load(cache_b_y),
                "ids": np.load(cache_b_ids),
            },
        }

    print(f"Computing features for {mode}...")

    # Engineer Features
    fe = FeatureEngineer()
    df_processed = fe.process(df)

    # Split Streams
    streams = split_streams(df_processed)

    # Cache Results
    print(f"Caching features for {mode}...")

    # Stream A
    streams["stream_a"]["X"].to_parquet(cache_a_X, index=False)
    np.save(cache_a_y, streams["stream_a"]["y"])
    np.save(cache_a_ids, streams["stream_a"]["ids"])

    # Stream B
    streams["stream_b"]["X"].to_parquet(cache_b_X, index=False)
    np.save(cache_b_y, streams["stream_b"]["y"])
    np.save(cache_b_ids, streams["stream_b"]["ids"])

    return streams
