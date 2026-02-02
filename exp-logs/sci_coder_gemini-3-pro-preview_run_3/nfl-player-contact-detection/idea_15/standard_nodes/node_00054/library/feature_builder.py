import os
import pandas as pd
import numpy as np
import hashlib
from library.config import Config
from library.utils import setup_logger
from library.data_manager import DataLoader


class FeatureBuilder:
    """
    Constructs feature matrices for Stream A (Player-Player) and Stream B (Player-Ground).
    Implements Context-Augmented Dual-Stream logic with caching.
    """

    def __init__(self):
        self.logger = setup_logger(name="FeatureBuilder")
        self.data_loader = DataLoader()

    def generate_stream_a_features(self, dataset_type="train", load_cache=True):
        """
        Generates features for the Player-Player Interaction Model.
        Includes Kinematics (P1, P2), Interaction Physics, and Visual Geometry.
        """
        cache_prefix = f"features_{dataset_type}_streamA"
        if result := self._load_from_cache(cache_prefix, load_cache):
            return result

        self.logger.info(f"Generating Stream A features for {dataset_type}...")

        # 1. Load Data
        df_meta = self._get_metadata(dataset_type)
        # Filter for Player-Player contacts (P2 != 'G')
        df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        # Load Tracking (Standard)
        df_tracking = self.data_loader.get_processed_tracking(
            dataset_type, load_cached_data=True
        )

        # Deduplicate tracking data to prevent merge explosion (Cite debug_lesson_17)
        df_tracking = df_tracking.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"]
        )

        # Standardize IDs to strings to prevent merge errors (Cite debug_lesson_13)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        # 2. Process Tracking (Kinematics + Temporal Pyramids)
        df_tracking = self._compute_derived_kinematics(df_tracking)

        # Create temporal features for Ego Kinematics
        # We process dense tracking first to get history for P1 and P2
        df_tracking_lagged = self._add_temporal_features(
            df_tracking,
            Config.EGO_FEATURES,
            Config.WINDOW_SIZES,
            group_cols=["game_play", "nfl_player_id"],
        )

        # 3. Merge Tracking to Metadata
        # Merge P1
        df_features = pd.merge(
            df_meta,
            df_tracking_lagged.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge P2
        df_features = pd.merge(
            df_features,
            df_tracking_lagged.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # 4. Compute Interaction Physics (Current Time)
        # We compute these on the merged dataframe
        df_features["dist_p1_p2"] = np.sqrt(
            (df_features["x_position_p1"] - df_features["x_position_p2"]) ** 2
            + (df_features["y_position_p1"] - df_features["y_position_p2"]) ** 2
        )
        df_features["rel_speed"] = np.abs(
            df_features["speed_p1"] - df_features["speed_p2"]
        )
        # Closure rate: derivative of distance (simplified as diff of speed vectors projected on connecting vector)
        # Here we use a simple proxy: sum of speeds if moving towards each other, but simple diff is robust enough for trees
        df_features["closure_rate"] = df_features["speed_p1"] * np.cos(
            np.radians(df_features["direction_p1"])
        ) + df_features["speed_p2"] * np.cos(
            np.radians(df_features["direction_p2"])
        )  # Rough approximation

        df_features["rel_angle"] = np.abs(
            df_features["direction_p1"] - df_features["direction_p2"]
        )
        df_features["rel_angle"] = np.minimum(
            df_features["rel_angle"], 360 - df_features["rel_angle"]
        )

        # Add temporal lags for Interaction Physics
        # Since df_features is a time-series of the pair, we can shift directly
        df_features = self._add_temporal_features(
            df_features,
            Config.INTERACTION_FEATURES,
            Config.WINDOW_SIZES,
            group_cols=["game_play", "nfl_player_id_1", "nfl_player_id_2"],
        )

        # 5. Visual Features
        df_features = self._add_visual_features(df_features, dataset_type)

        # 6. Finalize
        feature_cols = [
            c
            for c in df_features.columns
            if any(f in c for f in Config.EGO_FEATURES)
            or any(f in c for f in Config.INTERACTION_FEATURES)
            or any(f in c for f in Config.VISUAL_FEATURES)
        ]

        # Exclude non-feature columns explicitly just in case
        exclude = [
            "game_play",
            "step",
            "nfl_player_id",
            "contact_id",
            "datetime",
            "contact",
            "video_path",
        ]
        feature_cols = [c for c in feature_cols if not any(ex in c for ex in exclude)]

        # Clean NaNs
        X = df_features[feature_cols].fillna(-999).astype(np.float32)
        y = df_features["contact"].values.astype(np.int8)
        ids = df_features["contact_id"].values

        self._save_to_cache(cache_prefix, X, y, ids)
        return X, y, ids

    def generate_stream_b_features(self, dataset_type="train", load_cache=True):
        """
        Generates features for the Player-Ground Impact Model.
        Includes Ego-Kinematics only (Cite solution_lesson_node_00053).
        """
        cache_prefix = f"features_{dataset_type}_streamB"
        if result := self._load_from_cache(cache_prefix, load_cache):
            return result

        self.logger.info(f"Generating Stream B features for {dataset_type}...")

        # 1. Load Data
        df_meta = self._get_metadata(dataset_type)
        # Filter for Player-Ground contacts (P2 == 'G')
        df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        # Load Tracking
        df_tracking = self.data_loader.get_processed_tracking(
            dataset_type, load_cached_data=True
        )

        # Deduplicate tracking data to prevent merge explosion (Cite debug_lesson_17)
        df_tracking = df_tracking.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"]
        )

        # Standardize IDs to strings (Cite debug_lesson_13)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)

        # 2. Process Tracking
        df_tracking = self._compute_derived_kinematics(df_tracking)

        # Create temporal features for Ego Kinematics
        features_to_lag = Config.EGO_FEATURES

        df_tracking_lagged = self._add_temporal_features(
            df_tracking,
            features_to_lag,
            Config.WINDOW_SIZES,
            group_cols=["game_play", "nfl_player_id"],
        )

        # 3. Merge Tracking to Metadata
        # Only merge P1 (Subject)
        df_features = pd.merge(
            df_meta,
            df_tracking_lagged,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # 4. Finalize
        feature_cols = [
            c for c in df_features.columns if any(f in c for f in features_to_lag)
        ]

        exclude = [
            "game_play",
            "step",
            "nfl_player_id",
            "contact_id",
            "datetime",
            "contact",
            "video_path",
        ]
        feature_cols = [c for c in feature_cols if not any(ex in c for ex in exclude)]

        X = df_features[feature_cols].fillna(-999).astype(np.float32)
        y = df_features["contact"].values.astype(np.int8)
        ids = df_features["contact_id"].values

        self._save_to_cache(cache_prefix, X, y, ids)
        return X, y, ids

    def _compute_derived_kinematics(self, df):
        """Computes Jerk, Angular Velocity, and Pose Alignment."""
        self.logger.info("Computing derived kinematics...")
        df = df.sort_values(["game_play", "nfl_player_id", "step"])

        # Group by player to prevent bleeding across plays/players
        g = df.groupby(["game_play", "nfl_player_id"])

        # Jerk: Derivative of acceleration
        df["jerk"] = g["acceleration"].diff().fillna(0)

        # Angular Velocity: Derivative of orientation
        # Handle 0-360 wrap-around
        ori_diff = g["orientation"].diff().fillna(0)
        ori_diff = np.where(ori_diff > 180, ori_diff - 360, ori_diff)
        ori_diff = np.where(ori_diff < -180, ori_diff + 360, ori_diff)
        df["angular_velocity"] = ori_diff

        # Pose Alignment: Diff between Orientation (facing) and Direction (motion)
        align = np.abs(df["orientation"] - df["direction"])
        align = np.minimum(align, 360 - align)
        df["pose_alignment"] = align

        return df

    def _add_temporal_features(self, df, features, windows, group_cols):
        """
        Adds lag/lead features for specified columns.
        Uses shift() efficiently.
        """
        self.logger.info(f"Adding temporal features for {len(features)} columns...")

        # Ensure sorted
        df = df.sort_values(group_cols + ["step"])
        g = df.groupby(group_cols)

        result_dfs = [df]

        for w in windows:
            if w == 0:
                continue

            # Past (Lag)
            shifted_past = g[features].shift(w)
            shifted_past.columns = [f"{col}_lag_{w}" for col in features]
            result_dfs.append(shifted_past)

            # Future (Lead) - Symmetric window
            shifted_future = g[features].shift(-w)
            shifted_future.columns = [f"{col}_lead_{w}" for col in features]
            result_dfs.append(shifted_future)

        return pd.concat(result_dfs, axis=1)

    def _add_visual_features(self, df_meta, dataset_type):
        """
        Computes IoU and Box Distance for P1-P2 pairs and adds temporal history.
        """
        self.logger.info("Computing visual features...")

        # Load Helmets
        df_helmets = self.data_loader.load_helmets(dataset_type)

        # Standardize IDs (Cite debug_lesson_13)
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

        # Map step to frame (approximate)
        # Snap (step 0) is frame 300. 59.94 fps. 0.1s step.
        # frame = 300 + step * 6
        df_meta["frame_approx"] = (300 + df_meta["step"] * 6).astype(int)

        # To match helmets, we need to be careful. Helmet data is by frame.
        # We'll merge on nearest frame.

        # Prepare Helmet Data
        # We need game_play, frame, nfl_player_id, left, width, top, height
        cols = ["game_play", "frame", "nfl_player_id", "left", "width", "top", "height"]
        df_h = df_helmets[cols].copy()

        # Deduplicate helmet data to prevent merge explosion (Cite debug_lesson_17)
        df_h = df_h.drop_duplicates(subset=["game_play", "frame", "nfl_player_id"])

        # Merge P1 Helmets
        df_merged = pd.merge(
            df_meta,
            df_h.add_suffix("_p1"),
            left_on=["game_play", "frame_approx", "nfl_player_id_1"],
            right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge P2 Helmets
        df_merged = pd.merge(
            df_merged,
            df_h.add_suffix("_p2"),
            left_on=["game_play", "frame_approx", "nfl_player_id_2"],
            right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Calculate IoU
        def calculate_iou(row):
            if pd.isna(row["left_p1"]) or pd.isna(row["left_p2"]):
                return 0.0

            x_left = max(row["left_p1"], row["left_p2"])
            y_top = max(row["top_p1"], row["top_p2"])
            x_right = min(
                row["left_p1"] + row["width_p1"], row["left_p2"] + row["width_p2"]
            )
            y_bottom = min(
                row["top_p1"] + row["height_p1"], row["top_p2"] + row["height_p2"]
            )

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = row["width_p1"] * row["height_p1"]
            area2 = row["width_p2"] * row["height_p2"]

            return intersection_area / float(area1 + area2 - intersection_area)

        # Calculate Centroid Distance
        def calculate_dist(row):
            if pd.isna(row["left_p1"]) or pd.isna(row["left_p2"]):
                return -1.0

            cx1 = row["left_p1"] + row["width_p1"] / 2
            cy1 = row["top_p1"] + row["height_p1"] / 2
            cx2 = row["left_p2"] + row["width_p2"] / 2
            cy2 = row["top_p2"] + row["height_p2"] / 2

            return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        # Vectorized calculation would be faster, but loop is safer for now given complexity
        # Actually, let's vectorize for speed
        x_left = np.maximum(
            df_merged["left_p1"].fillna(0), df_merged["left_p2"].fillna(0)
        )
        y_top = np.maximum(df_merged["top_p1"].fillna(0), df_merged["top_p2"].fillna(0))
        x_right = np.minimum(
            (df_merged["left_p1"] + df_merged["width_p1"]).fillna(0),
            (df_merged["left_p2"] + df_merged["width_p2"]).fillna(0),
        )
        y_bottom = np.minimum(
            (df_merged["top_p1"] + df_merged["height_p1"]).fillna(0),
            (df_merged["top_p2"] + df_merged["height_p2"]).fillna(0),
        )

        inter_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
        area1 = (df_merged["width_p1"] * df_merged["height_p1"]).fillna(0)
        area2 = (df_merged["width_p2"] * df_merged["height_p2"]).fillna(0)
        union = area1 + area2 - inter_area

        df_merged["helmet_iou"] = np.where(union > 0, inter_area / union, 0.0)

        cx1 = df_merged["left_p1"] + df_merged["width_p1"] / 2
        cy1 = df_merged["top_p1"] + df_merged["height_p1"] / 2
        cx2 = df_merged["left_p2"] + df_merged["width_p2"] / 2
        cy2 = df_merged["top_p2"] + df_merged["height_p2"] / 2

        df_merged["helmet_dist"] = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2).fillna(
            9999
        )

        # Temporal Lags for Visuals
        # Group by pair and shift
        df_merged = self._add_temporal_features(
            df_merged,
            Config.VISUAL_FEATURES,
            Config.VISUAL_WINDOW_SIZES,
            group_cols=["game_play", "nfl_player_id_1", "nfl_player_id_2"],
        )

        return df_merged

    def _get_metadata(self, dataset_type):
        if dataset_type == "train":
            return pd.read_csv(Config.TRAIN_META_PATH)
        elif dataset_type == "validation":
            return pd.read_csv(Config.VAL_META_PATH)
        elif dataset_type == "test":
            return pd.read_csv(Config.TEST_META_PATH)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _save_to_cache(self, prefix, X, y, ids):
        X_path = os.path.join(Config.WORKING_DIR, f"{prefix}_X.parquet")
        y_path = os.path.join(Config.WORKING_DIR, f"{prefix}_y.npy")
        ids_path = os.path.join(Config.WORKING_DIR, f"{prefix}_ids.npy")

        self.logger.info(f"Saving features to {X_path}...")
        X.to_parquet(X_path)
        np.save(y_path, y)
        np.save(ids_path, ids)

    def _load_from_cache(self, prefix, load_cache):
        if not load_cache:
            return None

        X_path = os.path.join(Config.WORKING_DIR, f"{prefix}_X.parquet")
        y_path = os.path.join(Config.WORKING_DIR, f"{prefix}_y.npy")
        ids_path = os.path.join(Config.WORKING_DIR, f"{prefix}_ids.npy")

        if (
            os.path.exists(X_path)
            and os.path.exists(y_path)
            and os.path.exists(ids_path)
        ):
            self.logger.info(f"Loading features from cache: {prefix}")
            try:
                X = pd.read_parquet(X_path)
                y = np.load(y_path, allow_pickle=True)
                ids = np.load(ids_path, allow_pickle=True)
                return X, y, ids
            except Exception as e:
                self.logger.warning(f"Failed to load cache {prefix}: {e}")
                return None
        return None
