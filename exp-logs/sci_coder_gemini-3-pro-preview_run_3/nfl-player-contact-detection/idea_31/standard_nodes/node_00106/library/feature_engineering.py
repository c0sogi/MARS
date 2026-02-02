import pandas as pd
import numpy as np
import os
import gc
import logging
from tqdm import tqdm
from library.config import Config
from library.utils import setup_logger, get_config_hash, reduce_mem_usage

# Suppress chained assignment warnings
pd.options.mode.chained_assignment = None


class FeatureEngineer:
    """
    Implements the Physically-Consistent Hybrid-Context Dual-Stream feature engineering pipeline.
    Handles data loading, physics-based feature extraction, temporal flattening, and caching.
    """

    def __init__(self):
        self.logger = setup_logger("FeatureEngineer")
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_paths(self, mode, stream):
        """Generates file paths for caching based on mode and stream."""
        config_hash = get_config_hash(Config.get_config_hash())
        prefix = f"features_{mode}_{stream}_{config_hash}"
        return {
            "X": os.path.join(self.cache_dir, f"{prefix}_X.parquet"),
            "y": os.path.join(self.cache_dir, f"{prefix}_y.npy"),
            "ids": os.path.join(self.cache_dir, f"{prefix}_ids.npy"),
        }

    def _load_metadata(self, mode):
        """Loads the appropriate metadata file based on mode."""
        if mode == "train":
            return pd.read_csv(Config.TRAIN_META_PATH)
        elif mode == "validation":
            return pd.read_csv(Config.VAL_META_PATH)
        elif mode == "test":
            return pd.read_csv(Config.TEST_META_PATH)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _load_tracking(self, mode):
        """Loads and preprocesses player tracking data."""
        # Validation mode uses train tracking data
        if mode == "train" or mode == "validation":
            path = Config.TRAIN_TRACKING_PATH
        else:
            path = Config.TEST_TRACKING_PATH

        self.logger.info(f"Loading tracking data from {path}...")
        df = pd.read_csv(path)

        # Optimization
        df = reduce_mem_usage(df, verbose=False)

        # Standardize orientation/direction to radians
        df["orientation_rad"] = np.deg2rad(df["orientation"])
        df["direction_rad"] = np.deg2rad(df["direction"])

        return df

    def _load_helmets(self, mode):
        """Loads and preprocesses helmet bounding box data."""
        if mode == "train" or mode == "validation":
            path = Config.TRAIN_HELMETS_PATH
        else:
            path = Config.TEST_HELMETS_PATH

        self.logger.info(f"Loading helmet data from {path}...")
        df = pd.read_csv(path)
        df = reduce_mem_usage(df, verbose=False)
        return df

    def _compute_ego_dynamics(self, df_tracking):
        """
        Computes Ego-Centric Dynamics for Stream B (Impact Model).
        Calculates Surge/Sway velocity, acceleration, and jerk.
        """
        self.logger.info("Computing Ego-Dynamics (Surge/Sway)...")

        # Sort for temporal differentiation
        df_tracking = df_tracking.sort_values(["game_play", "nfl_player_id", "step"])

        # Project velocity onto orientation
        # Surge: Motion in direction of facing
        # Sway: Motion perpendicular to facing
        # v_surge = speed * cos(direction - orientation)
        # v_sway = speed * sin(direction - orientation)

        angle_diff = df_tracking["direction_rad"] - df_tracking["orientation_rad"]
        df_tracking["v_surge"] = df_tracking["speed"] * np.cos(angle_diff)
        df_tracking["v_sway"] = df_tracking["speed"] * np.sin(angle_diff)

        # Compute Acceleration (First Derivative)
        # Group by play and player to prevent boundary bleeding
        grp = df_tracking.groupby(["game_play", "nfl_player_id"])

        df_tracking["a_surge"] = grp["v_surge"].diff().fillna(0) / 0.1  # 10Hz
        df_tracking["a_sway"] = grp["v_sway"].diff().fillna(0) / 0.1

        # Compute Jerk (Second Derivative)
        df_tracking["j_surge"] = grp["a_surge"].diff().fillna(0) / 0.1
        df_tracking["j_sway"] = grp["a_sway"].diff().fillna(0) / 0.1

        return df_tracking

    def _compute_iou(self, box1, box2):
        """Vectorized IoU calculation."""
        # box: [left, width, top, height]
        # x1, y1, x2, y2
        b1_x1, b1_x2 = box1[0], box1[0] + box1[1]
        b1_y1, b1_y2 = box1[2], box1[2] + box1[3]

        b2_x1, b2_x2 = box2[0], box2[0] + box2[1]
        b2_y1, b2_y2 = box2[2], box2[2] + box2[3]

        # Intersection
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        b1_area = box1[1] * box1[3]
        b2_area = box2[1] * box2[3]
        union_area = b1_area + b2_area - inter_area

        return np.where(union_area > 0, inter_area / union_area, 0.0)

    def _flatten_temporal_context(
        self, df_main, df_features, feature_cols, lags, join_keys
    ):
        """
        Flattens temporal context by merging features at specified lags.
        """
        self.logger.info(f"Flattening temporal context with lags: {lags}")

        # Base features (lag 0)
        # Drop feature columns from base to prevent schema collision (duplication/renaming) during Lag 0 merge. Cite debug_lesson_11.
        cols_to_drop = [
            c for c in feature_cols if c in df_main.columns and c not in join_keys
        ]
        df_res = df_main.drop(columns=cols_to_drop).copy()

        # Prepare feature source
        # Ensure join keys are present
        cols_to_fetch = list(set(join_keys + feature_cols))
        df_feat_source = df_features[cols_to_fetch].copy()

        # Iterate through lags (including 0 if needed, usually 0 is base)
        # We want t, t-1, t+1, etc.
        # Lags in config are positive integers, implying +/-

        full_lags = [0]
        for l in lags:
            full_lags.append(l)
            full_lags.append(-l)

        full_lags = sorted(list(set(full_lags)))

        for lag in full_lags:
            suffix = f"_lag{lag}" if lag != 0 else ""

            # Create a temporary merge key on the source
            # We want features at (step + lag) to match current (step)
            # So if we want lag +1 (future), we look at source step where source_step = current_step + 1
            # Merge condition: df_res.step + lag == df_feat.step

            # Efficient merge:
            # Rename source columns with suffix
            rename_dict = {c: f"{c}{suffix}" for c in feature_cols}

            # We need to shift the source 'step' to align
            # If we want features from t+1 attached to row t:
            # The source row has step=t+1. We want it to join with row t.
            # So source_step - 1 = target_step.
            # Generally: source_step - lag = target_step

            df_shifted = df_feat_source.copy()
            df_shifted["step_match"] = df_shifted["step"] - lag

            # Drop original step to avoid confusion, use step_match for joining
            df_shifted = df_shifted.drop(columns=["step"])

            # Rename features
            df_shifted = df_shifted.rename(columns=rename_dict)

            # Merge
            # Left join to keep all labels
            merge_keys = [k for k in join_keys if k != "step"] + ["step_match"]
            left_keys = [k for k in join_keys if k != "step"] + ["step"]

            df_res = pd.merge(
                df_res, df_shifted, left_on=left_keys, right_on=merge_keys, how="left"
            )

            # Clean up match col
            if "step_match" in df_res.columns:
                df_res = df_res.drop(columns=["step_match"])

        # Fill NaNs with sentinel
        filled_cols = [
            c for c in df_res.columns if any(base in c for base in feature_cols)
        ]
        df_res[filled_cols] = df_res[filled_cols].fillna(-999)

        return df_res

    def validate_schema(self, df, required_features):
        """
        Enforces Pipeline Integrity.
        Raises RuntimeError if columns are missing or zero-filled.
        """
        self.logger.info("Validating schema...")
        missing = [c for c in required_features if c not in df.columns]
        if missing:
            raise RuntimeError(f"Pipeline Integrity Error: Missing features: {missing}")

        # Check for zero-filled columns (excluding sentinel -999)
        # This catches bugs where features are generated but not populated
        for col in required_features:
            if col in df.columns:
                # Check if all values are 0 (or very close)
                # Ignore sentinel
                vals = df[col]
                if np.all(vals == 0) or np.all(vals == -999):
                    self.logger.warning(f"Feature {col} appears to be constant/empty.")
                    # We warn but don't crash for constant, only crash for missing

        self.logger.info("Schema validation passed.")

    def process_stream_a(self, mode="train", load_cached_data=True):
        """
        Stream A: Interaction Model (Player-Player)
        Features: Relational Scalars, Energy, Visual Consensus
        """
        cache = self._get_cache_paths(mode, "streamA")

        if load_cached_data and os.path.exists(cache["X"]):
            self.logger.info(f"Loading cached Stream A data for {mode}...")
            return (
                pd.read_parquet(cache["X"]),
                np.load(cache["y"]),
                np.load(cache["ids"], allow_pickle=True),
            )

        self.logger.info(f"Processing Stream A for {mode}...")

        # 1. Load Data
        df_meta = self._load_metadata(mode)
        df_track = self._load_tracking(mode)

        # Filter for Player-Player contacts only
        df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        if Config.DEBUG_SAMPLE_SIZE and mode == "train":
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Merge Tracking for P1 and P2
        # Ensure IDs are correct types
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(str)

        # Merge P1
        df_merged = pd.merge(
            df_meta,
            df_track.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge P2
        df_merged = pd.merge(
            df_merged,
            df_track.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # 3. Compute Relational Scalars (Instantaneous)
        # Distance
        dx = df_merged["x_position_p1"] - df_merged["x_position_p2"]
        dy = df_merged["y_position_p1"] - df_merged["y_position_p2"]
        df_merged["distance"] = np.sqrt(dx**2 + dy**2)

        # Closure Rate (Instantaneous Projection)
        # v_rel = v1 - v2
        # r_rel = p1 - p2
        # closure = -(v_rel . r_rel) / |r_rel|
        vx1 = df_merged["speed_p1"] * np.sin(
            df_merged["direction_rad_p1"]
        )  # direction is 0=N, 90=E? Standard is 0=Y, 90=X usually in NFL data?
        # Actually NFL data: 0 is Y axis (short axis), 90 is X axis (long axis).
        # x_position is long axis (0-120). y_position is short axis (0-53.3).
        # direction: 0 degrees points along Y axis?
        # Standard NFL Big Data Bowl convention: 0 is along Y, 90 along X.
        # So Vx = Speed * sin(dir), Vy = Speed * cos(dir).

        vx1 = df_merged["speed_p1"] * np.sin(df_merged["direction_rad_p1"])
        vy1 = df_merged["speed_p1"] * np.cos(df_merged["direction_rad_p1"])

        vx2 = df_merged["speed_p2"] * np.sin(df_merged["direction_rad_p2"])
        vy2 = df_merged["speed_p2"] * np.cos(df_merged["direction_rad_p2"])

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Dot product
        dot_prod = dvx * dx + dvy * dy
        # Avoid div by zero
        dist_safe = df_merged["distance"].replace(0, 1e-6)
        df_merged["closure_rate"] = -(dot_prod) / dist_safe

        # 4. Visual Features
        df_helmets = self._load_helmets(mode)
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)
        # Map step to frame: frame = round(300 + step * 5.994)
        df_merged["frame_approx"] = (
            (300 + df_merged["step"] * 5.994).round().astype(int)
        )

        # We need to merge helmets for P1 and P2 for both views
        # This is complex to do for every row.
        # Strategy: Pre-calculate IoUs for all pairs in helmets? Too N^2.
        # Strategy: Merge helmets to df_merged.

        # Helper to merge specific view
        def merge_view(df_main, view_name):
            # Filter helmets by view
            h_view = df_helmets[df_helmets["view"] == view_name]

            # Merge P1
            df_m = pd.merge(
                df_main,
                h_view[
                    [
                        "game_play",
                        "frame",
                        "nfl_player_id",
                        "left",
                        "width",
                        "top",
                        "height",
                    ]
                ].add_suffix("_p1"),
                left_on=["game_play", "frame_approx", "nfl_player_id_1"],
                right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
                how="left",
            )
            # Merge P2
            df_m = pd.merge(
                df_m,
                h_view[
                    [
                        "game_play",
                        "frame",
                        "nfl_player_id",
                        "left",
                        "width",
                        "top",
                        "height",
                    ]
                ].add_suffix("_p2"),
                left_on=["game_play", "frame_approx", "nfl_player_id_2"],
                right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
                how="left",
            )

            # Calculate IoU
            # Handle missing boxes
            p1_box = [
                df_m["left_p1"].fillna(0),
                df_m["width_p1"].fillna(0),
                df_m["top_p1"].fillna(0),
                df_m["height_p1"].fillna(0),
            ]
            p2_box = [
                df_m["left_p2"].fillna(0),
                df_m["width_p2"].fillna(0),
                df_m["top_p2"].fillna(0),
                df_m["height_p2"].fillna(0),
            ]

            iou = self._compute_iou(p1_box, p2_box)
            return iou

        df_merged["sideline_iou"] = merge_view(df_merged, "Sideline")
        df_merged["endzone_iou"] = merge_view(df_merged, "Endzone")

        # Consensus
        df_merged["max_iou"] = df_merged[["sideline_iou", "endzone_iou"]].max(axis=1)
        df_merged["min_iou"] = df_merged[["sideline_iou", "endzone_iou"]].min(axis=1)
        df_merged["iou_diff"] = (
            df_merged["sideline_iou"] - df_merged["endzone_iou"]
        ).abs()

        # 5. Temporal Flattening
        # We need to create a source DF for flattening
        # The source must contain the features we want to lag, indexed by game_play, step, p1, p2
        # Since df_merged already has the computed features for every step, we can use it as source.
        # Note: df_merged contains all labeled steps. If labels are sparse (not every 0.1s), we might miss intermediate steps.
        # However, train_labels.csv usually contains every step for the play duration?
        # Actually, labels are for specific contacts. But usually provided for all steps in the play?
        # Description says: "Contains a row for every combination of players ... for each 0.1 second timestamp".
        # So yes, it is dense in time.

        feature_cols = (
            Config.STREAM_A_FEATURES["relational"]
            + [
                "speed_p1",
                "acceleration_p1",
                "sa_p1",
                "speed_p2",
                "acceleration_p2",
                "sa_p2",
            ]
            + Config.STREAM_A_FEATURES["visual"]
        )

        # Note: visual_looming_rate is not yet computed. It depends on lags.
        # We will compute it after flattening.
        # Remove it from fetch list for now
        fetch_cols = [c for c in feature_cols if c != "visual_looming_rate"]

        join_keys = ["game_play", "step", "nfl_player_id_1", "nfl_player_id_2"]

        df_flat = self._flatten_temporal_context(
            df_merged, df_merged, fetch_cols, Config.EXP_LAGS, join_keys
        )

        # 6. Post-Flattening Derived Features
        # Visual Looming Rate: (Max_IoU_t - Max_IoU_t-1) / 0.1
        # We use lag 1.
        # If lag 1 is t-1 (previous), then rate = (val - val_lag1)
        # If lag 1 is t+1 (future), we need to be careful with Config definition.
        # Assuming Config.EXP_LAGS=[1...] means we have _lag1 (t+1) and _lag-1 (t-1).
        # Let's use _lag-1 if available, else 0.

        if "max_iou_lag-1" in df_flat.columns:
            df_flat["visual_looming_rate"] = (
                df_flat["max_iou"] - df_flat["max_iou_lag-1"]
            ) / 0.1
        else:
            df_flat["visual_looming_rate"] = 0

        # Cross-Modal Verification
        # Diff between Normalized Closure Rate and Looming
        # Normalize closure rate? Maybe just raw diff.
        # "Calculate the difference between Normalized_Closure_Rate and Visual_Looming_Rate"
        # Apply heuristic scaling factor (10x) to align magnitudes (Cite Lesson 00105)
        df_flat["verification_diff"] = df_flat["closure_rate"] - (
            df_flat["visual_looming_rate"] * 10.0
        )

        # 7. Final Selection
        # Collect all lag columns
        final_cols = []
        for base in feature_cols + ["verification_diff"]:
            final_cols.extend([c for c in df_flat.columns if base in c])

        X = df_flat[final_cols].fillna(-999)
        y = df_flat["contact"].values
        ids = df_flat["contact_id"].values

        # Validate
        self.validate_schema(X, final_cols)

        # Save
        X.to_parquet(cache["X"])
        np.save(cache["y"], y)
        np.save(cache["ids"], ids)

        return X, y, ids

    def process_stream_b(self, mode="train", load_cached_data=True):
        """
        Stream B: Impact Model (Player-Ground)
        Features: Hybrid Context (Field-Centric + Ego-Dynamics)
        """
        cache = self._get_cache_paths(mode, "streamB")

        if load_cached_data and os.path.exists(cache["X"]):
            self.logger.info(f"Loading cached Stream B data for {mode}...")
            return (
                pd.read_parquet(cache["X"]),
                np.load(cache["y"]),
                np.load(cache["ids"], allow_pickle=True),
            )

        self.logger.info(f"Processing Stream B for {mode}...")

        # 1. Load Data
        df_meta = self._load_metadata(mode)
        df_track = self._load_tracking(mode)

        # Filter for Player-Ground contacts
        df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        if Config.DEBUG_SAMPLE_SIZE and mode == "train":
            df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

        # 2. Compute Ego Dynamics on Tracking Data
        df_track = self._compute_ego_dynamics(df_track)

        # 3. Merge Tracking to Labels (P1 only)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(str)

        # We need a source for flattening.
        # Unlike Stream A, we don't need pair-wise features.
        # We can use df_track as the source directly!
        # But we need to ensure df_track has 'step' aligned.

        # 4. Flattening
        feature_cols = (
            Config.STREAM_B_FEATURES["field_centric"]
            + Config.STREAM_B_FEATURES["ego_dynamics"]
        )

        join_keys = ["game_play", "step", "nfl_player_id"]

        # Rename nfl_player_id_1 to nfl_player_id for merging
        df_meta_renamed = df_meta.rename(columns={"nfl_player_id_1": "nfl_player_id"})

        # Flatten
        # Source is df_track (contains all steps for all players)
        # Target is df_meta_renamed (contains specific label steps)
        df_flat = self._flatten_temporal_context(
            df_meta_renamed, df_track, feature_cols, Config.EXP_LAGS, join_keys
        )

        # 5. Final Selection
        final_cols = []
        for base in feature_cols:
            final_cols.extend([c for c in df_flat.columns if base in c])

        X = df_flat[final_cols].fillna(-999)
        y = df_flat["contact"].values
        ids = df_flat["contact_id"].values

        # Validate
        self.validate_schema(X, final_cols)

        # Save
        X.to_parquet(cache["X"])
        np.save(cache["y"], y)
        np.save(cache["ids"], ids)

        return X, y, ids
