import pandas as pd
import numpy as np
import os
import gc
from library.config import Config
from library.utils import reduce_mem_usage, load_data


class FeatureGenerator:
    """
    Handles the generation of features for the Dual-Stream GBDT model.
    Manages caching, data merging, and feature engineering for both
    Player-Player (Stream A) and Player-Ground (Stream B) interactions.
    """

    def __init__(self):
        self.fps = 59.94
        self.snap_frame = 300
        self.step_interval = 0.1  # 10Hz
        self.frames_per_step = self.step_interval * self.fps  # ~5.994

    def generate_features(self, split="train", load_cached_data=True, debug=False):
        """
        Main entry point to generate features for Stream A and Stream B.

        Args:
            split (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.
            debug (bool): If True, processes a subset of data.

        Returns:
            dict: Contains 'stream_a' and 'stream_b' dictionaries, each having 'X', 'y', 'ids'.
        """
        # Define cache paths
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        cache_path_a = os.path.join(cache_dir, f"features_{split}_streamA.parquet")
        cache_path_b = os.path.join(cache_dir, f"features_{split}_streamB.parquet")

        # Check cache
        if (
            load_cached_data
            and os.path.exists(cache_path_a)
            and os.path.exists(cache_path_b)
        ):
            print(f"Loading cached features for {split}...")
            df_a = pd.read_parquet(cache_path_a)
            df_b = pd.read_parquet(cache_path_b)
            return self._format_output(df_a, df_b)

        print(f"Generating features for {split} from scratch...")

        # 1. Load Metadata (Labels/Submission structure)
        if split == "train":
            meta_path = Config.TRAIN_META_PATH
        elif split == "validation":
            meta_path = Config.VAL_META_PATH
        else:
            meta_path = Config.TEST_META_PATH

        df_meta = pd.read_csv(meta_path)

        if debug:
            df_meta = df_meta.head(5000)

        # 2. Load Raw Data
        # Determine which tracking/helmet file to use
        is_test = split == "test"
        tracking_path = (
            Config.TEST_TRACKING_PATH if is_test else Config.TRAIN_TRACKING_PATH
        )
        helmets_path = (
            Config.TEST_HELMETS_PATH if is_test else Config.TRAIN_HELMETS_PATH
        )

        # Filter raw data to only relevant game_plays to save memory
        relevant_plays = df_meta["game_play"].unique()

        print("Loading and processing tracking data...")
        df_tracking = load_data(tracking_path)
        df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)].copy()
        df_tracking = self._process_tracking(df_tracking)

        print("Loading and processing helmet data...")
        df_helmets = load_data(helmets_path)
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()
        df_helmets = self._process_helmets(df_helmets)

        # 3. Generate Stream A (Player vs Player)
        print("Building Stream A (Player-Player)...")
        df_a = self._build_stream_a(df_meta, df_tracking, df_helmets)

        # 4. Generate Stream B (Player vs Ground)
        print("Building Stream B (Player-Ground)...")
        df_b = self._build_stream_b(df_meta, df_tracking, df_helmets)

        # 5. Cache Results
        print("Caching features...")
        df_a.to_parquet(cache_path_a, index=False)
        df_b.to_parquet(cache_path_b, index=False)

        return self._format_output(df_a, df_b)

    def _format_output(self, df_a, df_b):
        """Separates features, targets, and IDs for model consumption."""

        # Helper to extract X, y, ids
        def extract(df, feature_cols):
            # Ensure all features exist, fill missing with 0 or NaN
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            X = df[feature_cols].copy()
            y = df["contact"].values if "contact" in df.columns else np.zeros(len(df))
            ids = df["contact_id"].values
            return X, y, ids

        X_a, y_a, ids_a = extract(df_a, Config.FEATURES_STREAM_A)
        X_b, y_b, ids_b = extract(df_b, Config.FEATURES_STREAM_B)

        return {
            "stream_a": {"X": X_a, "y": y_a, "ids": ids_a},
            "stream_b": {"X": X_b, "y": y_b, "ids": ids_b},
        }

    def _process_tracking(self, df):
        """
        Engineers tracking features: Cyclical encoding and Rolling windows.
        """
        # Sort for rolling calculations
        df = df.sort_values(["game_play", "nfl_player_id", "step"]).reset_index(
            drop=True
        )

        # Cyclical Encoding
        for col in ["orientation", "direction"]:
            # Convert degrees to radians
            rad = np.deg2rad(df[col].fillna(0))
            df[f"{col}_sin"] = np.sin(rad)
            df[f"{col}_cos"] = np.cos(rad)

        # Define columns to roll
        roll_cols = ["speed", "acceleration", "sa"]

        # We need to apply rolling windows per player per play.
        # GroupBy + Rolling is robust.
        # Micro Window (e.g., 4 steps)
        # Macro Window (e.g., 15 steps)

        # To optimize, we can use the fact that data is sorted.
        # We'll use a loop or optimized groupby. Given 1M rows, groupby is fine.

        # Group object
        g = df.groupby(["game_play", "nfl_player_id"])

        for col in roll_cols:
            # Macro Stats
            df[f"roll_mean_{col}"] = g[col].transform(
                lambda x: x.rolling(
                    Config.WINDOW_SIZE_MACRO, min_periods=1, center=True
                ).mean()
            )
            df[f"roll_std_{col}"] = g[col].transform(
                lambda x: x.rolling(
                    Config.WINDOW_SIZE_MACRO, min_periods=1, center=True
                ).std()
            )
            if col == "speed":
                df[f"roll_max_{col}"] = g[col].transform(
                    lambda x: x.rolling(
                        Config.WINDOW_SIZE_MACRO, min_periods=1, center=True
                    ).max()
                )
            if col == "sa":
                df[f"roll_min_{col}"] = g[col].transform(
                    lambda x: x.rolling(
                        Config.WINDOW_SIZE_MACRO, min_periods=1, center=True
                    ).min()
                )

        df = reduce_mem_usage(df, verbose=False)
        return df

    def _process_helmets(self, df):
        """
        Engineers helmet features: Frame-to-Step mapping, Pseudo-3D dynamics, View flattening.
        """
        # 1. Map Frame to Step
        # step = round((frame - 300) / 5.994)
        df["step"] = (
            ((df["frame"] - self.snap_frame) / self.frames_per_step).round().astype(int)
        )

        # Filter invalid steps (tracking starts at 0, usually goes up to ~100-150)
        # We keep a bit of buffer but generally remove very negative steps
        df = df[df["step"] >= -10].copy()

        # 2. Compute Geometry & Dynamics
        # We need to compute dynamics (velocity) BEFORE flattening views,
        # but we need to ensure we are calculating change over time for the same object.

        # Sort: game_play, view, nfl_player_id, frame
        df = df.sort_values(["game_play", "view", "nfl_player_id", "frame"])

        # Calculate Drop Velocity: d(top)/dt
        # Since frames are ~16ms apart, but we map to steps (100ms).
        # Let's calculate raw frame-to-frame velocity then average or pick?
        # Simpler: Calculate velocity at the step level after aggregation?
        # No, multiple frames map to one step. We should pick the frame closest to the step center.
        # The rounding logic above effectively buckets frames.
        # Let's drop duplicates keeping the one closest to the exact step time?
        # For simplicity/speed: drop_duplicates on [game_play, view, player, step].
        df = df.drop_duplicates(subset=["game_play", "view", "nfl_player_id", "step"])

        # Now re-sort by step to calculate step-to-step dynamics
        df = df.sort_values(["game_play", "view", "nfl_player_id", "step"])

        # Group for diffs
        g = df.groupby(["game_play", "view", "nfl_player_id"])

        # Drop Velocity (positive means moving down/falling)
        # step is 0.1s.
        df["helmet_drop_velocity"] = g["top"].diff() / 0.1

        # Visual Compression (Height / Width)
        df["visual_compression"] = df["height"] / df["width"]

        # Area
        df["bbox_area"] = df["width"] * df["height"]

        # 3. Flatten Views (Sideline vs Endzone)
        # We want one row per (game_play, step, nfl_player_id) with cols like sideline_left, endzone_left

        # Pivot
        # We need to preserve the columns we want
        value_cols = [
            "left",
            "top",
            "width",
            "height",
            "helmet_drop_velocity",
            "visual_compression",
            "bbox_area",
        ]

        df_pivot = df.pivot_table(
            index=["game_play", "step", "nfl_player_id"],
            columns="view",
            values=value_cols,
        )

        # Flatten MultiIndex columns
        # e.g., ('left', 'Sideline') -> 'sideline_left'
        df_pivot.columns = [f"{v.lower()}_{c}" for c, v in df_pivot.columns]

        df_pivot = df_pivot.reset_index()

        # Rename nfl_player_id to match tracking merge keys if needed,
        # but tracking uses nfl_player_id (int), helmets is imperfect (sometimes float/int).
        # Ensure ID is int
        df_pivot["nfl_player_id"] = (
            pd.to_numeric(df_pivot["nfl_player_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

        df_pivot = reduce_mem_usage(df_pivot, verbose=False)
        return df_pivot

    def _build_stream_a(self, df_meta, df_tracking, df_helmets):
        """
        Constructs dataset for Player-Player interactions.
        """
        # Filter for Player-Player contacts
        df = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        # Ensure IDs are ints
        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)
        df["nfl_player_id_2"] = df["nfl_player_id_2"].astype(int)

        # --- Merge Tracking ---
        # Player 1
        df = df.merge(
            df_tracking.add_prefix("p1_"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["p1_game_play", "p1_step", "p1_nfl_player_id"],
            how="left",
        )

        # Player 2
        df = df.merge(
            df_tracking.add_prefix("p2_"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["p2_game_play", "p2_step", "p2_nfl_player_id"],
            how="left",
        )

        # --- Merge Helmets ---
        # Player 1
        df = df.merge(
            df_helmets.add_prefix("p1_"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["p1_game_play", "p1_step", "p1_nfl_player_id"],
            how="left",
        )

        # Player 2
        df = df.merge(
            df_helmets.add_prefix("p2_"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["p2_game_play", "p2_step", "p2_nfl_player_id"],
            how="left",
        )

        # --- Feature Engineering (Relative) ---

        # Tracking Distance
        df["x_diff"] = df["p1_x_position"] - df["p2_x_position"]
        df["y_diff"] = df["p1_y_position"] - df["p2_y_position"]
        df["distance_p1_p2"] = np.sqrt(df["x_diff"] ** 2 + df["y_diff"] ** 2)

        # Speed Diff
        df["speed_diff"] = (df["p1_speed"] - df["p2_speed"]).abs()

        # Relative Speed (Closing speed approximation)
        # Simple scalar diff for now, vector diff is better but scalar is in whitelist
        df["rel_speed"] = df["speed_diff"]

        # --- Feature Engineering (Visual IoU) ---
        # Sideline IoU
        df["sideline_iou"] = self._compute_iou(
            df["p1_sideline_left"],
            df["p1_sideline_top"],
            df["p1_sideline_width"],
            df["p1_sideline_height"],
            df["p2_sideline_left"],
            df["p2_sideline_top"],
            df["p2_sideline_width"],
            df["p2_sideline_height"],
        )

        # Endzone IoU
        df["endzone_iou"] = self._compute_iou(
            df["p1_endzone_left"],
            df["p1_endzone_top"],
            df["p1_endzone_width"],
            df["p1_endzone_height"],
            df["p2_endzone_left"],
            df["p2_endzone_top"],
            df["p2_endzone_width"],
            df["p2_endzone_height"],
        )

        # Visual Distance (Pixel Centroids)
        # Sideline
        p1_cx_s = df["p1_sideline_left"] + df["p1_sideline_width"] / 2
        p1_cy_s = df["p1_sideline_top"] + df["p1_sideline_height"] / 2
        p2_cx_s = df["p2_sideline_left"] + df["p2_sideline_width"] / 2
        p2_cy_s = df["p2_sideline_top"] + df["p2_sideline_height"] / 2
        df["sideline_dist_pixel"] = np.sqrt(
            (p1_cx_s - p2_cx_s) ** 2 + (p1_cy_s - p2_cy_s) ** 2
        )

        # Endzone
        p1_cx_e = df["p1_endzone_left"] + df["p1_endzone_width"] / 2
        p1_cy_e = df["p1_endzone_top"] + df["p1_endzone_height"] / 2
        p2_cx_e = df["p2_endzone_left"] + df["p2_endzone_width"] / 2
        p2_cy_e = df["p2_endzone_top"] + df["p2_endzone_height"] / 2
        df["endzone_dist_pixel"] = np.sqrt(
            (p1_cx_e - p2_cx_e) ** 2 + (p1_cy_e - p2_cy_e) ** 2
        )

        # Rename area columns to match whitelist
        df["sideline_p1_area"] = df["p1_sideline_bbox_area"]
        df["sideline_p2_area"] = df["p2_sideline_bbox_area"]
        df["endzone_p1_area"] = df["p1_endzone_bbox_area"]
        df["endzone_p2_area"] = df["p2_endzone_bbox_area"]

        # Fix column names for tracking to match whitelist (remove p1_ prefix for p1 specific if needed?
        # No, whitelist has p1_speed etc.)

        return reduce_mem_usage(df, verbose=False)

    def _build_stream_b(self, df_meta, df_tracking, df_helmets):
        """
        Constructs dataset for Player-Ground interactions.
        """
        # Filter for Player-Ground contacts
        df = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()

        df["nfl_player_id_1"] = df["nfl_player_id_1"].astype(int)

        # --- Merge Tracking (P1 only) ---
        # We don't add prefix 'p1_' here because whitelist expects 'speed', 'acceleration' etc.
        df = df.merge(
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # --- Merge Helmets (P1 only) ---
        # Whitelist expects 'sideline_helmet_drop_velocity' etc.
        # Our processed helmets have 'sideline_helmet_drop_velocity'.
        df = df.merge(
            df_helmets,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        return reduce_mem_usage(df, verbose=False)

    def _compute_iou(self, l1, t1, w1, h1, l2, t2, w2, h2):
        """Vectorized IoU calculation."""
        # Right and Bottom
        r1, b1 = l1 + w1, t1 + h1
        r2, b2 = l2 + w2, t2 + h2

        # Intersection
        x_overlap = np.maximum(0, np.minimum(r1, r2) - np.maximum(l1, l2))
        y_overlap = np.maximum(0, np.minimum(b1, b2) - np.maximum(t1, t2))
        intersection = x_overlap * y_overlap

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        # Avoid div by zero
        iou = intersection / (union + 1e-6)

        # If any box is NaN, IoU is 0 (or NaN, but 0 is safer for features)
        iou = iou.fillna(0)
        return iou
