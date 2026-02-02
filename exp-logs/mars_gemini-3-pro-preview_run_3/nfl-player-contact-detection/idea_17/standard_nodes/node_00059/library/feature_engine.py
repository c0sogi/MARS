import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import get_hashed_cache_path


class FeatureEngine:
    """
    Implements the Hybrid Coordinate Dual-Stream feature engineering pipeline.

    Stream A: Interaction Model (Player-Player) -> Relational Tracking + Visuals
    Stream B: Hybrid Impact Model (Player-Ground) -> Field-Centric + Ego-Centric Kinematics
    """

    def __init__(self):
        self.exp_lags = Config.EXP_LAGS
        self.visual_lags = Config.VISUAL_LAGS
        self.tracking_cols = Config.TRACKING_COLS

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def _compute_ego_motion(self, df_tracking):
        """
        Projects Field-Centric motion into Ego-Centric Surge/Sway components.
        Computes Velocity, Acceleration, and Jerk in the Ego frame.
        """
        # Ensure data is sorted for delta calculations
        df = df_tracking.sort_values(by=["game_play", "nfl_player_id", "step"]).copy()

        # Convert degrees to radians
        # Assumption: direction and orientation are 0-360 degrees.
        # Relative Angle: Difference between motion direction and player facing orientation
        # theta = 0 means moving forward (Surge +)
        # theta = 90 means moving right (Sway +)

        # Note: We use (Direction - Orientation).
        # If Dir=90 (East), Orient=0 (North), Angle=90. Cos(90)=0 (Surge), Sin(90)=1 (Sway). Correct.

        theta_rad = np.radians(df["direction"] - df["orientation"])

        # 1. Ego-Centric Velocity
        df["surge_v"] = df["speed"] * np.cos(theta_rad)
        df["sway_v"] = df["speed"] * np.sin(theta_rad)

        # 2. Ego-Centric Acceleration (Finite Differences)
        # Group by player to prevent bleeding across plays
        grp = df.groupby(["game_play", "nfl_player_id"])

        # Time delta is 0.1s
        dt = 0.1

        df["surge_a"] = grp["surge_v"].diff() / dt
        df["sway_a"] = grp["sway_v"].diff() / dt

        # Fill NaNs created by diff (first step of each play) with 0
        df["surge_a"] = df["surge_a"].fillna(0)
        df["sway_a"] = df["sway_a"].fillna(0)

        # 3. Ego-Centric Jerk (Finite Differences of Acceleration)
        df["surge_j"] = grp["surge_a"].diff() / dt
        df["sway_j"] = grp["sway_a"].diff() / dt

        df["surge_j"] = df["surge_j"].fillna(0)
        df["sway_j"] = df["sway_j"].fillna(0)

        return df

    def _add_lags(self, df, feature_cols, lags, group_cols):
        """
        Generates exponential temporal pyramids (lags) for specified features.
        """
        # Ensure sort order
        df = df.sort_values(by=group_cols + ["step"])

        grouped = df.groupby(group_cols)[feature_cols]

        new_dfs = []
        for lag in lags:
            if lag == 0:
                continue

            # Naming convention: lag_1 (t-1), lag_m1 (t+1)
            # Note: shift(k) takes value from index i-k and places it at i.
            # So shift(1) brings t-1 to t.
            suffix = f"_lag_{lag}" if lag > 0 else f"_lag_m{abs(lag)}"

            shifted = grouped.shift(lag)
            shifted.columns = [c + suffix for c in feature_cols]
            new_dfs.append(shifted)

        if new_dfs:
            df = pd.concat([df] + new_dfs, axis=1)

        return df

    def _calculate_iou(self, df):
        """
        Vectorized IoU calculation for bounding boxes.
        Expects columns: left_1, width_1, top_1, height_1, left_2, ...
        """
        # Box 1
        x1_1 = df["left_1"]
        y1_1 = df["top_1"]
        x2_1 = df["left_1"] + df["width_1"]
        y2_1 = df["top_1"] + df["height_1"]

        # Box 2
        x1_2 = df["left_2"]
        y1_2 = df["top_2"]
        x2_2 = df["left_2"] + df["width_2"]
        y2_2 = df["top_2"] + df["height_2"]

        # Intersection
        xi1 = np.maximum(x1_1, x1_2)
        yi1 = np.maximum(y1_1, y1_2)
        xi2 = np.minimum(x2_1, x2_2)
        yi2 = np.minimum(y2_1, y2_2)

        inter_width = np.maximum(0, xi2 - xi1)
        inter_height = np.maximum(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        box1_area = df["width_1"] * df["height_1"]
        box2_area = df["width_2"] * df["height_2"]
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = inter_area / np.maximum(union_area, 1e-6)  # Avoid div by zero

        # Centroid Distance
        c1_x = x1_1 + df["width_1"] / 2
        c1_y = y1_1 + df["height_1"] / 2
        c2_x = x1_2 + df["width_2"] / 2
        c2_y = y1_2 + df["height_2"] / 2

        dist = np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

        return iou, dist

    def build_stream_a(
        self, df_meta, df_tracking, df_helmets, mode="train", load_cached_data=True
    ):
        """
        Constructs features for Stream A (Player-Player Interaction).
        Features: Relational Tracking + Visuals + Lags.
        """
        # Determine cache path based on mode
        if mode == "train":
            cache_path = Config.CACHE_STREAM_A_TRAIN
        elif mode == "validation":
            cache_path = Config.CACHE_STREAM_A_VAL
        elif mode == "test":
            cache_path = Config.CACHE_STREAM_A_TEST
        else:
            # Fallback for custom modes
            cache_path = os.path.join(Config.WORKING_DIR, f"streamA_{mode}.parquet")

        # Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"[Stream A] Loading features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"[Stream A] Building features for {mode}...")

        # 1. Filter Tracking Data
        # We need tracking for both P1 and P2
        # df_meta has nfl_player_id_1 and nfl_player_id_2

        # Ensure IDs are strings/ints consistently
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        # 2. Merge Tracking P1
        track_cols = ["game_play", "step", "nfl_player_id"] + self.tracking_cols
        p1_track = df_tracking[track_cols].add_suffix("_p1")

        df_merged = pd.merge(
            df_meta,
            p1_track,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # 3. Merge Tracking P2
        p2_track = df_tracking[track_cols].add_suffix("_p2")

        df_merged = pd.merge(
            df_merged,
            p2_track,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # 4. Relational Features (Tracking)
        # Euclidean Distance
        df_merged["dist_p1_p2"] = np.sqrt(
            (df_merged["x_position_p1"] - df_merged["x_position_p2"]) ** 2
            + (df_merged["y_position_p1"] - df_merged["y_position_p2"]) ** 2
        )

        # Relative Speed & Angular Features (Cite solution_lesson_node_00057)
        # Convert angles to radians
        dir_p1_rad = np.radians(df_merged["direction_p1"].fillna(0))
        dir_p2_rad = np.radians(df_merged["direction_p2"].fillna(0))
        orient_p1_rad = np.radians(df_merged["orientation_p1"].fillna(0))
        orient_p2_rad = np.radians(df_merged["orientation_p2"].fillna(0))

        # Velocity Vectors (0 deg is North/Y-axis)
        vx_p1 = df_merged["speed_p1"] * np.sin(dir_p1_rad)
        vy_p1 = df_merged["speed_p1"] * np.cos(dir_p1_rad)
        vx_p2 = df_merged["speed_p2"] * np.sin(dir_p2_rad)
        vy_p2 = df_merged["speed_p2"] * np.cos(dir_p2_rad)

        # Relative Speed (Magnitude of velocity difference)
        df_merged["rel_speed"] = np.sqrt((vx_p1 - vx_p2) ** 2 + (vy_p1 - vy_p2) ** 2)

        # Angular Similarities
        df_merged["cos_sim_dir"] = np.cos(dir_p1_rad - dir_p2_rad)
        df_merged["cos_sim_orient"] = np.cos(orient_p1_rad - orient_p2_rad)

        # Pose-Motion Alignment (Cite solution_lesson_node_00027)
        df_merged["cos_orient_dir_p1"] = np.cos(orient_p1_rad - dir_p1_rad)
        df_merged["cos_orient_dir_p2"] = np.cos(orient_p2_rad - dir_p2_rad)

        # Closure Rate (Derivative of distance)
        # We need to group by pair to calculate diff
        # Create a unique pair ID for grouping
        df_merged["pair_id"] = (
            df_merged["game_play"]
            + "_"
            + df_merged["nfl_player_id_1"]
            + "_"
            + df_merged["nfl_player_id_2"]
        )

        # Sort for diffs
        df_merged = df_merged.sort_values(by=["pair_id", "step"])

        # Closure Rate: -(dist(t) - dist(t-1))/dt. Positive means closing in.
        grp_pair = df_merged.groupby("pair_id")
        df_merged["closure_rate"] = -grp_pair["dist_p1_p2"].diff() / 0.1
        df_merged["closure_rate"] = df_merged["closure_rate"].fillna(0)

        # 5. Visual Features (Helmets)
        # Map step to frame: frame = ((step * 0.1 + 5.0) * 59.94)
        # Snap is 5s in. Step 0 is snap.
        df_merged["frame"] = (
            ((df_merged["step"] * 0.1 + 5.0) * 59.94).round().astype(int)
        )

        # Filter helmets to Sideline view (primary visual source)
        helmets_side = df_helmets[df_helmets["view"] == "Sideline"].copy()

        # Prepare Helmets for Merge
        # We need columns: game_play, frame, nfl_player_id, left, width, top, height
        h_cols = [
            "game_play",
            "frame",
            "nfl_player_id",
            "left",
            "width",
            "top",
            "height",
        ]
        helmets_side["nfl_player_id"] = helmets_side["nfl_player_id"].astype(str)

        # Merge P1 Helmets
        h_p1 = helmets_side[h_cols].add_suffix("_1")
        # Rename join keys back
        h_p1 = h_p1.rename(
            columns={
                "game_play_1": "game_play",
                "frame_1": "frame",
                "nfl_player_id_1": "nfl_player_id_1",
            }
        )

        df_merged = pd.merge(
            df_merged, h_p1, on=["game_play", "frame", "nfl_player_id_1"], how="left"
        )

        # Merge P2 Helmets
        h_p2 = helmets_side[h_cols].add_suffix("_2")
        h_p2 = h_p2.rename(
            columns={
                "game_play_2": "game_play",
                "frame_2": "frame",
                "nfl_player_id_2": "nfl_player_id_2",
            }
        )

        df_merged = pd.merge(
            df_merged, h_p2, on=["game_play", "frame", "nfl_player_id_2"], how="left"
        )

        # Compute IoU and Centroid Dist
        # Fill missing boxes with sentinel/NaN logic handled by calculation (NaN propagates)
        iou, c_dist = self._calculate_iou(df_merged)
        df_merged["iou"] = iou.fillna(-1)  # Sentinel for no overlap/missing
        df_merged["visual_dist"] = c_dist.fillna(-1)

        # 6. Temporal Lags
        # Tracking Lags
        track_feats = [
            "dist_p1_p2",
            "closure_rate",
            "speed_p1",
            "speed_p2",
            "acceleration_p1",
            "acceleration_p2",
            "rel_speed",
            "cos_sim_dir",
            "cos_sim_orient",
            "cos_orient_dir_p1",
            "cos_orient_dir_p2",
        ]
        df_merged = self._add_lags(df_merged, track_feats, self.exp_lags, ["pair_id"])

        # Visual Lags
        vis_feats = ["iou", "visual_dist"]
        df_merged = self._add_lags(df_merged, vis_feats, self.visual_lags, ["pair_id"])

        # 7. Cleanup and Selection
        # Drop raw helper columns
        drop_cols = [
            c
            for c in df_merged.columns
            if "game_play_" in c or "step_" in c or "nfl_player_id_" in c
        ]
        df_merged = df_merged.drop(columns=drop_cols, errors="ignore")

        # Keep identifiers + features + target
        # Identify feature columns (numeric)
        exclude = [
            "contact_id",
            "game_play",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
            "pair_id",
            "frame",
        ]

        feature_cols = [c for c in df_merged.columns if c not in exclude]

        # Ensure target is present (it might be in df_meta)
        if "contact" not in df_merged.columns and "contact" in df_meta.columns:
            # It should be there from the initial merge if df_meta had it
            pass

        # Save to cache
        print(f"[Stream A] Saving {df_merged.shape} to {cache_path}...")
        # Convert object columns to category for parquet efficiency if needed, or just save
        df_merged.to_parquet(cache_path, index=False)

        return df_merged

    def build_stream_b(
        self, df_meta, df_tracking, df_helmets, mode="train", load_cached_data=True
    ):
        """
        Constructs features for Stream B (Player-Ground Impact).
        Features: Field-Centric + Ego-Centric Kinematics + Lags.
        Excludes Visuals.
        """
        if mode == "train":
            cache_path = Config.CACHE_STREAM_B_TRAIN
        elif mode == "validation":
            cache_path = Config.CACHE_STREAM_B_VAL
        elif mode == "test":
            cache_path = Config.CACHE_STREAM_B_TEST
        else:
            cache_path = os.path.join(Config.WORKING_DIR, f"streamB_{mode}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"[Stream B] Loading features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(f"[Stream B] Building features for {mode}...")

        # 1. Process Tracking Data (Ego Motion)
        # We only care about P1 (P2 is Ground)
        # Pre-calculate ego motion for all players in tracking to save time during merge?
        # Or just filter relevant players first.

        # Filter tracking to only players in stream B meta
        relevant_players = df_meta["nfl_player_id_1"].astype(str).unique()
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        track_subset = df_tracking[
            df_tracking["nfl_player_id"].isin(relevant_players)
        ].copy()

        # Compute Ego Motion (Surge/Sway V, A, J)
        track_aug = self._compute_ego_motion(track_subset)

        # 2. Merge with Labels
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)

        # Columns to keep from tracking
        # Field Centric
        field_cols = self.tracking_cols  # x, y, speed, dir, orient, accel, sa
        # Ego Centric
        ego_cols = ["surge_v", "sway_v", "surge_a", "sway_a", "surge_j", "sway_j"]

        track_cols_final = (
            ["game_play", "step", "nfl_player_id"] + field_cols + ego_cols
        )

        df_merged = pd.merge(
            df_meta,
            track_aug[track_cols_final],
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 3. Temporal Lags
        # Apply lags to both Field and Ego features
        feats_to_lag = field_cols + ego_cols
        # Remove position from lags if not needed? Usually delta/velocity is more important than absolute x/y for contact.
        # But prompt says "Flattened Exponential Pyramids of raw Position...". So keep x/y.

        # Group by Player (since P2 is Ground, interaction is purely single-body physics relative to world)
        # Use game_play + player_id
        df_merged = self._add_lags(
            df_merged, feats_to_lag, self.exp_lags, ["game_play", "nfl_player_id_1"]
        )

        # 4. Cleanup
        drop_cols = ["nfl_player_id"]  # Duplicate from merge
        df_merged = df_merged.drop(columns=drop_cols, errors="ignore")

        exclude = [
            "contact_id",
            "game_play",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
        ]

        # Save
        print(f"[Stream B] Saving {df_merged.shape} to {cache_path}...")
        df_merged.to_parquet(cache_path, index=False)

        return df_merged
