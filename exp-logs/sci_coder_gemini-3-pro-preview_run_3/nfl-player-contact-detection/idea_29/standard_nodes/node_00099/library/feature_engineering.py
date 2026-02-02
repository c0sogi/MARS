import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.data_loader import DataLoader
from library.utils import generate_config_hash


class FeatureEngineer:
    """
    Implements feature engineering for the Hybrid-Context Dual-Stream GBDT.
    Handles Stream A (Interaction) and Stream B (Impact) separately with
    strict caching and vectorized transformations.
    """

    def __init__(self, split: str, load_cached_data: bool = True):
        self.split = split
        self.load_cached_data = load_cached_data
        self.meta_df = DataLoader.load_metadata(split)

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def process_stream_a(self):
        """
        Generates features for Stream A (Player-Player Contact).
        Strategy: Consistency & Verification (Tracking + Visuals).
        """
        stream_name = "A"
        config_hash = generate_config_hash(Config.get_hashable_config(stream_name))

        # Cache paths
        path_X = os.path.join(
            Config.WORKING_DIR,
            f"features_{self.split}_stream{stream_name}_{config_hash}_X.parquet",
        )
        path_ids = os.path.join(
            Config.WORKING_DIR,
            f"features_{self.split}_stream{stream_name}_{config_hash}_ids.npy",
        )
        path_y = os.path.join(
            Config.WORKING_DIR,
            f"features_{self.split}_stream{stream_name}_{config_hash}_y.npy",
        )

        if self.load_cached_data and os.path.exists(path_X):
            print(f"Loading cached Stream A features for {self.split}...")
            return (
                pd.read_parquet(path_X),
                np.load(path_ids, allow_pickle=True),
                np.load(path_y),
            )

        print(f"Generating Stream A features for {self.split}...")

        # 1. Filter Metadata for Player-Player interactions (player2 != 'G')
        df = self.meta_df[self.meta_df["nfl_player_id_2"] != "G"].copy()
        if df.empty:
            print("No Player-Player interactions found in this split.")
            return pd.DataFrame(), np.array([]), np.array([])

        # 2. Load Tracking and Helmets
        track_df = DataLoader.load_tracking(self.split, self.load_cached_data)
        helmet_df = DataLoader.load_helmets(self.split, self.load_cached_data)

        # 3. Type Conversion for Merge Safety
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(str)
        track_df["nfl_player_id"] = track_df["nfl_player_id"].astype(str)
        helmet_df["nfl_player_id"] = helmet_df["nfl_player_id"].astype(str)

        # 4. Merge Tracking Data (P1 and P2)
        print("Merging tracking data...")
        # Merge P1
        df = df.merge(
            track_df,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p1"),
        )
        df = df.rename(
            columns={
                c: c + "_p1"
                for c in track_df.columns
                if c not in ["game_play", "step", "nfl_player_id"]
                and c + "_p1" not in df.columns
            }
        )
        df = df.drop(columns=["nfl_player_id"], errors="ignore")

        # Merge P2
        df = df.merge(
            track_df,
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_p2"),
        )
        df = df.rename(
            columns={
                c: c + "_p2"
                for c in track_df.columns
                if c not in ["game_play", "step", "nfl_player_id"]
                and c + "_p2" not in df.columns
            }
        )
        df = df.drop(columns=["nfl_player_id"], errors="ignore")

        # 5. Compute Relational Features
        print("Computing relational features...")
        # Euclidean Distance
        df["distance"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        ).fillna(
            100.0
        )  # Default large distance if missing

        # Cosine Similarities (Cite Lesson 00074)
        o_rad_p1 = np.deg2rad(df["orientation_p1"].fillna(0))
        o_rad_p2 = np.deg2rad(df["orientation_p2"].fillna(0))
        d_rad_p1 = np.deg2rad(df["direction_p1"].fillna(0))
        d_rad_p2 = np.deg2rad(df["direction_p2"].fillna(0))

        df["orientation_cos_sim"] = np.cos(o_rad_p1 - o_rad_p2)
        df["direction_cos_sim"] = np.cos(d_rad_p1 - d_rad_p2)

        # Closure Rate (Finite Difference)
        # Sort to ensure correct temporal order for diff
        df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

        # Group by pair to prevent bleeding across plays/pairs
        # Note: This assumes the input df has sequential steps. If sparse, this is approx.
        # Given train_labels is 10Hz dense, this works.
        df["dist_prev"] = df.groupby(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        )["distance"].shift(1)
        df["closure_rate"] = -(
            df["distance"] - df["dist_prev"]
        )  # Positive = Closing in
        df["closure_rate"] = df["closure_rate"].fillna(0)

        # 6. Compute Visual Consensus (IoU)
        print("Computing visual features...")
        # Map step to frame: Frame = 300 + step * 5.994
        df["frame_approx"] = (300 + df["step"] * 5.994).round().astype(int)

        # Helper to merge specific view IoU
        def compute_view_iou(main_df, h_df, view_name):
            view_h = h_df[h_df["view"] == view_name]

            # Strictly project columns to avoid schema pollution (Cite debug_lesson_11)
            cols_to_keep = [
                "game_play",
                "frame",
                "nfl_player_id",
                "left",
                "width",
                "top",
                "height",
            ]
            view_h = view_h[cols_to_keep]

            # Merge P1 Box
            merged = main_df.merge(
                view_h,
                left_on=["game_play", "frame_approx", "nfl_player_id_1"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )
            merged = merged.rename(
                columns={
                    c: f"{view_name}_{c}_p1" for c in ["left", "width", "top", "height"]
                }
            )
            merged = merged.drop(
                columns=["nfl_player_id", "frame", "view", "player_label"],
                errors="ignore",
            )

            # Merge P2 Box
            merged = merged.merge(
                view_h,
                left_on=["game_play", "frame_approx", "nfl_player_id_2"],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )
            merged = merged.rename(
                columns={
                    c: f"{view_name}_{c}_p2" for c in ["left", "width", "top", "height"]
                }
            )
            merged = merged.drop(
                columns=["nfl_player_id", "frame", "view", "player_label"],
                errors="ignore",
            )

            # Calculate Intersection
            x_left = np.maximum(
                merged[f"{view_name}_left_p1"], merged[f"{view_name}_left_p2"]
            )
            y_top = np.maximum(
                merged[f"{view_name}_top_p1"], merged[f"{view_name}_top_p2"]
            )
            x_right = np.minimum(
                merged[f"{view_name}_left_p1"] + merged[f"{view_name}_width_p1"],
                merged[f"{view_name}_left_p2"] + merged[f"{view_name}_width_p2"],
            )
            y_bottom = np.minimum(
                merged[f"{view_name}_top_p1"] + merged[f"{view_name}_height_p1"],
                merged[f"{view_name}_top_p2"] + merged[f"{view_name}_height_p2"],
            )

            intersection = np.maximum(0, x_right - x_left) * np.maximum(
                0, y_bottom - y_top
            )
            area_p1 = merged[f"{view_name}_width_p1"] * merged[f"{view_name}_height_p1"]
            area_p2 = merged[f"{view_name}_width_p2"] * merged[f"{view_name}_height_p2"]
            union = area_p1 + area_p2 - intersection

            iou_col = f"{view_name.lower()}_iou"
            merged[iou_col] = (intersection / union).fillna(0)  # 0 if boxes missing

            # Cleanup temp cols
            cols_to_drop = [
                c for c in merged.columns if f"{view_name}_" in c and c != iou_col
            ]
            return merged.drop(columns=cols_to_drop)

        df = compute_view_iou(df, helmet_df, "Sideline")
        df = compute_view_iou(df, helmet_df, "Endzone")

        df["max_iou"] = df[["sideline_iou", "endzone_iou"]].max(axis=1)
        df["min_iou"] = df[["sideline_iou", "endzone_iou"]].min(axis=1)
        df["iou_diff"] = (df["sideline_iou"] - df["endzone_iou"]).abs()

        # 7. Cross-Modal Consistency
        # Visual Looming: Derivative of Max IoU
        df["max_iou_prev"] = df.groupby(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        )["max_iou"].shift(1)
        df["visual_looming"] = (df["max_iou"] - df["max_iou_prev"]).fillna(0)

        # Consistency Metric: Closure Rate vs Visual Looming
        # Scaling factor 10 is heuristic to bring IoU rate (0-1) closer to Speed (yds/s)
        df["consistency_metric"] = df["closure_rate"] - (df["visual_looming"] * 10)

        # 8. Temporal Flattening (Pyramids)
        print("Applying temporal flattening...")
        features_to_lag = (
            Config.STREAM_A_FEATURES["relational"]
            + Config.STREAM_A_FEATURES["visual"]
            + Config.STREAM_A_FEATURES["cross_modal"]
            + Config.STREAM_A_FEATURES["energy"]
        )

        # Ensure all base features exist
        for f in features_to_lag:
            if f not in df.columns:
                df[f] = 0.0

        # Apply lags
        grouper = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])
        lagged_cols = []

        for lag in Config.LAG_SCHEDULE:
            if lag == 0:
                lagged_cols.extend(features_to_lag)
                continue

            suffix = f"_lag_{lag}" if lag < 0 else f"_lead_{lag}"
            suffix = suffix.replace("-", "m")

            # Lag k means t+k. If k=-15, we want t-15.
            # Pandas shift(k): shifts data down by k. Value at t becomes t+k.
            # To get value from t-15 at t, we need to look 'up' 15 rows? No.
            # df['prev'] = df['curr'].shift(1). Row t gets value from t-1.
            # So shift(15) gets value from t-15.
            # shift(-15) gets value from t+15.
            # Config lag -15 means past. shift(-lag) = shift(15).
            shift_amount = -lag

            for col in features_to_lag:
                new_col = f"{col}{suffix}"
                df[new_col] = grouper[col].shift(shift_amount).fillna(0)
                lagged_cols.append(new_col)

        # 9. Final Output Preparation
        X = df[lagged_cols].astype(np.float32)
        y = df["contact"].astype(np.int8).values
        ids = df["contact_id"].values

        print(f"Stream A Features Shape: {X.shape}")

        # Save to Cache
        X.to_parquet(path_X)
        np.save(path_ids, ids)
        np.save(path_y, y)

        return X, ids, y

    def process_stream_b(self):
        """
        Generates features for Stream B (Player-Ground Contact).
        Strategy: Hybrid Context + Ego-Dynamics (No Visuals).
        """
        stream_name = "B"
        config_hash = generate_config_hash(Config.get_hashable_config(stream_name))

        path_X = os.path.join(
            Config.WORKING_DIR,
            f"features_{self.split}_stream{stream_name}_{config_hash}_X.parquet",
        )
        path_ids = os.path.join(
            Config.WORKING_DIR,
            f"features_{self.split}_stream{stream_name}_{config_hash}_ids.npy",
        )
        path_y = os.path.join(
            Config.WORKING_DIR,
            f"features_{self.split}_stream{stream_name}_{config_hash}_y.npy",
        )

        if self.load_cached_data and os.path.exists(path_X):
            print(f"Loading cached Stream B features for {self.split}...")
            return (
                pd.read_parquet(path_X),
                np.load(path_ids, allow_pickle=True),
                np.load(path_y),
            )

        print(f"Generating Stream B features for {self.split}...")

        # 1. Filter Metadata for Player-Ground (player2 == 'G')
        df = self.meta_df[self.meta_df["nfl_player_id_2"] == "G"].copy()
        if df.empty:
            print("No Player-Ground interactions found in this split.")
            return pd.DataFrame(), np.array([]), np.array([])

        # 2. Load Tracking
        track_df = DataLoader.load_tracking(self.split, self.load_cached_data)

        # 3. Precompute Ego-Centric Dynamics on full tracking data
        print("Computing Ego-Centric Dynamics...")
        track_df = track_df.sort_values(["game_play", "nfl_player_id", "step"])

        # Convert angles to radians
        # NFL orientation: 0-360. We convert to unit vectors.
        rad_o = np.deg2rad(track_df["orientation"].fillna(0))
        rad_d = np.deg2rad(track_df["direction"].fillna(0))

        # Orientation Vector (Heading)
        o_x = np.sin(rad_o)
        o_y = np.cos(rad_o)

        # Velocity Vector
        speed = track_df["speed"].fillna(0)
        v_x = speed * np.sin(rad_d)
        v_y = speed * np.cos(rad_d)

        # Surge: Velocity projected onto Heading
        track_df["v_surge"] = v_x * o_x + v_y * o_y

        # Sway: Velocity projected onto Perpendicular Heading (-y, x)
        track_df["v_sway"] = v_x * (-o_y) + v_y * o_x

        # Derivatives (Acceleration & Jerk)
        # Group by player to handle boundaries
        grp = track_df.groupby(["game_play", "nfl_player_id"])

        track_df["a_surge"] = grp["v_surge"].diff().fillna(0)
        track_df["a_sway"] = grp["v_sway"].diff().fillna(0)

        track_df["j_surge"] = grp["a_surge"].diff().fillna(0)
        track_df["j_sway"] = grp["a_sway"].diff().fillna(0)

        # 4. Merge with Labels
        print("Merging tracking data...")
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(str)
        track_df["nfl_player_id"] = track_df["nfl_player_id"].astype(str)

        df = df.merge(
            track_df,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 5. Temporal Flattening
        print("Applying temporal flattening...")
        features_to_lag = (
            Config.STREAM_B_FEATURES["field_centric"]
            + Config.STREAM_B_FEATURES["ego_centric"]
        )

        for f in features_to_lag:
            if f not in df.columns:
                df[f] = 0.0
            df[f] = df[f].fillna(0)

        grouper = df.groupby(["game_play", "nfl_player_id_1"])
        lagged_cols = []

        for lag in Config.LAG_SCHEDULE:
            if lag == 0:
                lagged_cols.extend(features_to_lag)
                continue

            suffix = f"_lag_{lag}" if lag < 0 else f"_lead_{lag}"
            suffix = suffix.replace("-", "m")
            shift_amount = -lag

            for col in features_to_lag:
                new_col = f"{col}{suffix}"
                df[new_col] = grouper[col].shift(shift_amount).fillna(0)
                lagged_cols.append(new_col)

        # 6. Final Output
        X = df[lagged_cols].astype(np.float32)
        y = df["contact"].astype(np.int8).values
        ids = df["contact_id"].values

        print(f"Stream B Features Shape: {X.shape}")

        # Save
        X.to_parquet(path_X)
        np.save(path_ids, ids)
        np.save(path_y, y)

        return X, ids, y
