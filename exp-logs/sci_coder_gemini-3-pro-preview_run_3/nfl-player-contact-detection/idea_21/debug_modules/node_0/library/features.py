import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage, get_hashed_cache_path


class FeatureGenerator:
    """
    Implements the Orthogonal-Physics Dual-Stream Feature Engineering pipeline.
    Stream A: Interaction (Relational Scalars + Visual Consensus)
    Stream B: Impact (Finite-Difference Ego-Dynamics)
    """

    def __init__(self, mode="train"):
        """
        Args:
            mode (str): 'train', 'validation', or 'test'. Determines which metadata to use.
        """
        self.mode = mode
        self.metadata_path = {
            "train": Config.TRAIN_META_PATH,
            "validation": Config.VAL_META_PATH,
            "test": Config.TEST_META_PATH,
        }[mode]

        # Load metadata to determine relevant game_plays
        self.meta_df = pd.read_csv(self.metadata_path)
        self.game_plays = self.meta_df["game_play"].unique()

        # Determine raw data source
        if mode == "test":
            self.tracking_path = Config.TEST_TRACKING_PATH
            self.helmets_path = Config.TEST_HELMETS_PATH
        else:
            self.tracking_path = Config.TRAIN_TRACKING_PATH
            self.helmets_path = Config.TRAIN_HELMETS_PATH

    def _load_tracking(self):
        """
        Loads tracking data, filters for relevant plays, and computes Ego-Dynamics.
        Returns dense tracking dataframe with physics features.
        """
        df_trk = pd.read_csv(self.tracking_path)
        df_trk = df_trk[df_trk["game_play"].isin(self.game_plays)].copy()

        # Preprocessing
        df_trk = reduce_mem_usage(df_trk)

        # --- Finite-Difference Ego-Dynamics ---
        # 1. Convert angles to radians (0 degrees is usually Y-axis in NFL tracking, but we need relative diffs)
        # Assuming standard mathematical convention or consistent internal relative angles.
        # Orientation (o) and Direction (dir).
        df_trk["o_rad"] = np.deg2rad(df_trk["orientation"])
        df_trk["dir_rad"] = np.deg2rad(df_trk["direction"])

        # 2. Velocity Components
        # speed is scalar. We project it.
        # Surge: Velocity along orientation.
        # Sway: Velocity perpendicular to orientation.
        # Relative angle = dir - o
        df_trk["rel_angle"] = df_trk["dir_rad"] - df_trk["o_rad"]
        df_trk["v_surge"] = df_trk["speed"] * np.cos(df_trk["rel_angle"])
        df_trk["v_sway"] = df_trk["speed"] * np.sin(df_trk["rel_angle"])

        # 3. Finite Differences for Acceleration and Jerk
        # Sort for correct time diffs
        df_trk.sort_values(["game_play", "nfl_player_id", "step"], inplace=True)

        # Group by player to prevent bleeding across plays/players
        grp = df_trk.groupby(["game_play", "nfl_player_id"])

        # Ego-Acceleration (Delta V / Delta t). Delta t is constant 0.1s (10Hz), so proportional to diff.
        df_trk["a_surge"] = grp["v_surge"].diff().fillna(0)
        df_trk["a_sway"] = grp["v_sway"].diff().fillna(0)

        # Ego-Jerk (Delta A / Delta t)
        df_trk["j_surge"] = grp["a_surge"].diff().fillna(0)
        df_trk["j_sway"] = grp["a_sway"].diff().fillna(0)

        # Cleanup intermediate columns to save memory
        drop_cols = ["o_rad", "dir_rad", "rel_angle"]
        df_trk.drop(columns=drop_cols, inplace=True)

        return df_trk

    def _load_helmets(self):
        """
        Loads helmet data, maps frames to steps, and prepares for IoU calculation.
        Mapping: step 0 approx frame 300. 10Hz vs 59.94Hz -> factor ~6.
        """
        df_helm = pd.read_csv(self.helmets_path)
        df_helm = df_helm[df_helm["game_play"].isin(self.game_plays)].copy()

        # Map frame to step
        # Formula: step = round((frame - 300) / 6)
        # We only keep frames that align close to a tracking step
        df_helm["step"] = ((df_helm["frame"] - 300) / 6).round()

        # Filter valid steps (tracking usually goes 0 to ~100-150)
        # We allow a bit of buffer
        df_helm = df_helm[(df_helm["step"] >= -5) & (df_helm["step"] < 200)]
        df_helm["step"] = df_helm["step"].astype(int)

        # Select relevant columns
        cols = [
            "game_play",
            "step",
            "nfl_player_id",
            "view",
            "left",
            "width",
            "top",
            "height",
        ]
        df_helm = df_helm[cols].copy()

        # Calculate x2, y2 for IoU
        df_helm["x2"] = df_helm["left"] + df_helm["width"]
        df_helm["y2"] = df_helm["top"] + df_helm["height"]

        return reduce_mem_usage(df_helm)

    def _compute_iou(self, box1, box2):
        """
        Vectorized IoU calculation.
        box: [left, top, x2, y2]
        """
        # Intersection
        xi1 = np.maximum(box1[0], box2[0])
        yi1 = np.maximum(box1[1], box2[1])
        xi2 = np.minimum(box1[2], box2[2])
        yi2 = np.minimum(box1[3], box2[3])

        inter_width = np.maximum(0, xi2 - xi1)
        inter_height = np.maximum(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        # Avoid division by zero
        iou = np.where(union_area > 0, inter_area / union_area, 0)
        return iou

    def _add_lags(self, df, features, group_cols):
        """
        Adds exponential temporal pyramid lags.
        """
        df = df.sort_values(group_cols + ["step"])
        grp = df.groupby(group_cols)

        for lag in Config.LAGS:
            if lag == 0:
                continue

            # Forward and Backward lags
            for direction, sign in [("prev", 1), ("next", -1)]:
                shifted = grp[features].shift(lag * sign)
                suffix = f"_{direction}_{lag}"
                shifted.columns = [f"{c}{suffix}" for c in features]
                df = pd.concat([df, shifted], axis=1)

        return df

    def generate_stream_a(self, load_cached=True):
        """
        Generates features for Stream A (Player-Player Interaction).
        Features: Relational Scalars, Visual Consensus.
        """
        # Cache Check
        config_hash = {"mode": self.mode, "stream": "A", "lags": Config.LAGS}
        cache_path_X = get_hashed_cache_path(
            f"features_{self.mode}_streamA_X", config_hash
        )
        cache_path_ids = get_hashed_cache_path(
            f"features_{self.mode}_streamA_ids", config_hash, ".npy"
        )
        cache_path_y = get_hashed_cache_path(
            f"features_{self.mode}_streamA_y", config_hash, ".npy"
        )

        if load_cached and os.path.exists(cache_path_X):
            print(f"Loading Stream A features from cache: {cache_path_X}")
            X = pd.read_parquet(cache_path_X)
            ids = np.load(cache_path_ids, allow_pickle=True)
            y = np.load(cache_path_y)
            return X, y, ids

        print("Generating Stream A features...")

        # 1. Filter Labels for Player-Player
        df = self.meta_df[self.meta_df["nfl_player_id_2"] != "G"].copy()

        # 2. Load Tracking
        trk = self._load_tracking()

        # 3. Merge Tracking for P1 and P2
        # Ensure IDs are numeric for merging
        df["nfl_player_id_1"] = pd.to_numeric(df["nfl_player_id_1"])
        df["nfl_player_id_2"] = pd.to_numeric(df["nfl_player_id_2"])

        # Merge P1
        df = df.merge(
            trk.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge P2
        df = df.merge(
            trk.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # 4. Relational Scalars
        df["dist"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )
        df["speed_diff"] = df["speed_p1"] - df["speed_p2"]
        # Closure Rate (derivative of distance, but simple diff is approximation)
        # We leave explicit closure rate to the tree to find via lags of distance

        # 5. Visual Consensus
        # Load Helmets
        helm = self._load_helmets()

        # We need to join helmets for p1 and p2 for both views
        # Create unique keys for join
        # Views: Sideline, Endzone

        for view in ["Sideline", "Endzone"]:
            view_helm = helm[helm["view"] == view]

            # Join P1
            df = df.merge(
                view_helm.add_suffix(f"_p1_{view}"),
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=[
                    f"game_play_p1_{view}",
                    f"step_p1_{view}",
                    f"nfl_player_id_p1_{view}",
                ],
                how="left",
            )

            # Join P2
            df = df.merge(
                view_helm.add_suffix(f"_p2_{view}"),
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=[
                    f"game_play_p2_{view}",
                    f"step_p2_{view}",
                    f"nfl_player_id_p2_{view}",
                ],
                how="left",
            )

            # Calculate IoU for this view
            # Box format: left, top, x2, y2
            b1 = [
                df[f"left_p1_{view}"],
                df[f"top_p1_{view}"],
                df[f"x2_p1_{view}"],
                df[f"y2_p1_{view}"],
            ]
            b2 = [
                df[f"left_p2_{view}"],
                df[f"top_p2_{view}"],
                df[f"x2_p2_{view}"],
                df[f"y2_p2_{view}"],
            ]

            df[f"iou_{view}"] = self._compute_iou(b1, b2)

            # Fill NaNs (missing detections) with -1 (Sentinel)
            df[f"iou_{view}"] = df[f"iou_{view}"].fillna(-1)

        # Consensus Features
        df["iou_max"] = df[["iou_Sideline", "iou_Endzone"]].max(axis=1)
        df["iou_min"] = df[["iou_Sideline", "iou_Endzone"]].min(axis=1)
        df["iou_diff"] = (df["iou_Sideline"] - df["iou_Endzone"]).abs()

        # 6. Lag Features
        base_features = ["dist", "speed_diff", "iou_max", "iou_min", "iou_diff"]
        # We must group by pair for lags
        group_cols = ["game_play", "nfl_player_id_1", "nfl_player_id_2"]
        df = self._add_lags(df, base_features, group_cols)

        # 7. Final Selection
        # Select numeric columns created + base features
        feature_cols = [
            c
            for c in df.columns
            if c in base_features or any(x in c for x in ["_prev_", "_next_"])
        ]

        # Fill remaining NaNs (from lags or missing tracking)
        df[feature_cols] = df[feature_cols].fillna(0)

        X = df[feature_cols].copy()
        y = df["contact"].values
        ids = df["contact_id"].values

        # Save Cache
        X.to_parquet(cache_path_X)
        np.save(cache_path_ids, ids)
        np.save(cache_path_y, y)

        return X, y, ids

    def generate_stream_b(self, load_cached=True):
        """
        Generates features for Stream B (Player-Ground Impact).
        Features: Finite-Difference Ego-Dynamics (Surge/Sway Jerk).
        """
        # Cache Check
        config_hash = {"mode": self.mode, "stream": "B", "lags": Config.LAGS}
        cache_path_X = get_hashed_cache_path(
            f"features_{self.mode}_streamB_X", config_hash
        )
        cache_path_ids = get_hashed_cache_path(
            f"features_{self.mode}_streamB_ids", config_hash, ".npy"
        )
        cache_path_y = get_hashed_cache_path(
            f"features_{self.mode}_streamB_y", config_hash, ".npy"
        )

        if load_cached and os.path.exists(cache_path_X):
            print(f"Loading Stream B features from cache: {cache_path_X}")
            X = pd.read_parquet(cache_path_X)
            ids = np.load(cache_path_ids, allow_pickle=True)
            y = np.load(cache_path_y)
            return X, y, ids

        print("Generating Stream B features...")

        # 1. Filter Labels for Player-Ground
        df = self.meta_df[self.meta_df["nfl_player_id_2"] == "G"].copy()

        # 2. Load Tracking (with pre-computed Ego-Dynamics)
        trk = self._load_tracking()

        # 3. Merge Tracking for P1
        df["nfl_player_id_1"] = pd.to_numeric(df["nfl_player_id_1"])

        # We need the full tracking history to create lags properly,
        # but since we already computed dynamics in _load_tracking,
        # we can just merge and then lag.
        # However, lags need to be contiguous.
        # Since 'df' (labels) might be sparse or subset, we should add lags to 'trk' first?
        # Yes, adding lags to 'trk' before merging is safer for Stream B
        # because Stream B features are purely ego-centric (depend only on P1).

        base_features = [
            "speed",
            "acceleration",
            "v_surge",
            "v_sway",
            "a_surge",
            "a_sway",
            "j_surge",
            "j_sway",
        ]

        # Add lags to tracking dataframe directly
        trk = self._add_lags(trk, base_features, ["game_play", "nfl_player_id"])

        # 4. Merge enriched tracking to labels
        df = df.merge(
            trk,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 5. Final Selection
        # Identify all feature columns
        feature_cols = [
            c
            for c in df.columns
            if c in base_features or any(x in c for x in ["_prev_", "_next_"])
        ]

        # Fill NaNs
        df[feature_cols] = df[feature_cols].fillna(0)

        X = df[feature_cols].copy()
        y = df["contact"].values
        ids = df["contact_id"].values

        # Save Cache
        X.to_parquet(cache_path_X)
        np.save(cache_path_ids, ids)
        np.save(cache_path_y, y)

        return X, y, ids
