import os
import gc
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, get_config_hash
from library.data_manager import DataManager


class FeatureEngineer:
    """
    Implements the Differential-Physics Dual-Stream feature engineering pipeline.
    Separates processing into Stream A (Interaction) and Stream B (Impact).
    """

    def __init__(self):
        self.logger = setup_logger("FeatureEngineer")
        self.config_hash = get_config_hash()
        self.working_dir = Config.WORKING_DIR
        self.data_manager = DataManager()

    def create_features(self, mode="train", load_cached_data=True):
        """
        Main entry point to generate or load features for a specific mode.

        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary containing:
                'stream_a': {'X': ..., 'y': ..., 'ids': ...},
                'stream_b': {'X': ..., 'y': ..., 'ids': ...}
        """
        # Define cache paths
        cache_prefix = os.path.join(
            self.working_dir, f"features_{mode}_{self.config_hash}"
        )
        paths = {
            "stream_a_X": f"{cache_prefix}_streamA_X.parquet",
            "stream_a_y": f"{cache_prefix}_streamA_y.npy",
            "stream_a_ids": f"{cache_prefix}_streamA_ids.npy",
            "stream_b_X": f"{cache_prefix}_streamB_X.parquet",
            "stream_b_y": f"{cache_prefix}_streamB_y.npy",
            "stream_b_ids": f"{cache_prefix}_streamB_ids.npy",
        }

        # 1. Try Loading from Cache
        if load_cached_data and all(os.path.exists(p) for p in paths.values()):
            self.logger.info(f"Loading cached features for mode: {mode}")
            return {
                "stream_a": {
                    "X": pd.read_parquet(paths["stream_a_X"]),
                    "y": np.load(paths["stream_a_y"]),
                    "ids": np.load(paths["stream_a_ids"], allow_pickle=True),
                },
                "stream_b": {
                    "X": pd.read_parquet(paths["stream_b_X"]),
                    "y": np.load(paths["stream_b_y"]),
                    "ids": np.load(paths["stream_b_ids"], allow_pickle=True),
                },
            }

        # 2. Process from Scratch
        self.logger.info(f"Generating features for mode: {mode}...")

        # Load Base Data
        df_meta = self.data_manager.load_metadata(mode)
        df_tracking = self.data_manager.load_tracking(mode, metadata_df=df_meta)

        # Load Helmets only if needed (Stream A uses visual features)
        # We load it anyway to ensure consistency, but could optimize.
        df_helmets = self.data_manager.load_helmets(mode, metadata_df=df_meta)

        # Split Metadata into Stream A and Stream B
        # Stream B: Player 2 is Ground ('G')
        mask_ground = df_meta["nfl_player_id_2"] == "G"
        df_meta_b = df_meta[mask_ground].copy()
        df_meta_a = df_meta[~mask_ground].copy()

        self.logger.info(
            f"Stream A samples: {len(df_meta_a)}, Stream B samples: {len(df_meta_b)}"
        )

        # --- Process Stream A ---
        if len(df_meta_a) > 0:
            X_a, y_a, ids_a = self._process_stream_a(df_meta_a, df_tracking, df_helmets)
        else:
            X_a, y_a, ids_a = pd.DataFrame(), np.array([]), np.array([])

        # --- Process Stream B ---
        if len(df_meta_b) > 0:
            X_b, y_b, ids_b = self._process_stream_b(df_meta_b, df_tracking)
        else:
            X_b, y_b, ids_b = pd.DataFrame(), np.array([]), np.array([])

        # Save to Cache
        self.logger.info("Saving features to cache...")
        if not X_a.empty:
            X_a.to_parquet(paths["stream_a_X"], index=False)
            np.save(paths["stream_a_y"], y_a)
            np.save(paths["stream_a_ids"], ids_a)

        if not X_b.empty:
            X_b.to_parquet(paths["stream_b_X"], index=False)
            np.save(paths["stream_b_y"], y_b)
            np.save(paths["stream_b_ids"], ids_b)

        return {
            "stream_a": {"X": X_a, "y": y_a, "ids": ids_a},
            "stream_b": {"X": X_b, "y": y_b, "ids": ids_b},
        }

    def _process_stream_a(self, df_meta, df_tracking, df_helmets):
        """
        Stream A: Interaction Model (Translational + Visual Consensus)
        """
        self.logger.info("Processing Stream A (Interaction)...")

        # Prepare Merge Keys
        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)
        df_tracking["step"] = df_tracking["step"].astype(int)

        # Merge Tracking for Player 1
        df = pd.merge(
            df_meta,
            df_tracking.add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge Tracking for Player 2
        df = pd.merge(
            df,
            df_tracking.add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # --- Translational Differentials ---
        # Euclidean Distance
        df["distance"] = np.sqrt(
            (df["x_position_p1"] - df["x_position_p2"]) ** 2
            + (df["y_position_p1"] - df["y_position_p2"]) ** 2
        )

        # Closure Rate: -(d_t - d_{t-1})
        # We need to sort by play and step to use shift correctly
        df.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
        )

        # Group by pair to calculate diff
        # Note: contact_id is unique per step, so we group by the entity pair (game_play, p1, p2)
        # But wait, df_meta is the labels file. It has one row per timestep.
        # We can group by game_play, p1, p2.
        grp = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])
        df["dist_lag1"] = grp["distance"].shift(1)
        # Fill NA for first step
        df["dist_lag1"] = df["dist_lag1"].fillna(df["distance"])
        # Closure rate: positive means closing in
        df["closure_rate"] = -(df["distance"] - df["dist_lag1"])

        # --- Visual Consensus ---
        # We need IoU for P1 vs P2 in Sideline and Endzone
        # Helper to merge helmets and calc IoU
        df = self._compute_visual_features(df, df_helmets)

        # --- Temporal Lags (System Energy & Visual) ---
        # Features to lag
        energy_cols = ["speed_p1", "speed_p2", "acceleration_p1", "acceleration_p2"]
        visual_cols = ["max_iou", "min_iou", "iou_diff"]

        # Apply Lags
        # Stream A Lags: [-15, ..., 15]
        # We re-use the group object if possible, but columns changed.
        grp = df.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        # Energy Lags
        for col in energy_cols:
            for lag in Config.STREAM_A_LAGS:
                df[f"{col}_lag_{lag}"] = grp[col].shift(lag).fillna(0)

        # Visual Lags
        for col in visual_cols:
            for lag in Config.STREAM_A_VISUAL_LAGS:
                df[f"{col}_lag_{lag}"] = (
                    grp[col].shift(lag).fillna(-999)
                )  # Sentinel for visual

        # Visual Looming: Finite difference of Max IoU
        df["max_iou_prev"] = grp["max_iou"].shift(1).fillna(0)
        df["visual_looming"] = df["max_iou"] - df["max_iou_prev"]

        # --- Final Selection ---
        # Collect all feature names
        feature_cols = Config.STREAM_A_FEATURES.copy()

        # Add generated lag columns
        for col in energy_cols:
            for lag in Config.STREAM_A_LAGS:
                feature_cols.append(f"{col}_lag_{lag}")
        for col in visual_cols:
            for lag in Config.STREAM_A_VISUAL_LAGS:
                feature_cols.append(f"{col}_lag_{lag}")

        # Ensure all columns exist (fill missing with 0 or sentinel)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Fill NaNs in base features
        df[feature_cols] = df[feature_cols].fillna(0)

        return df[feature_cols], df["contact"].values, df["contact_id"].values

    def _process_stream_b(self, df_meta, df_tracking):
        """
        Stream B: Impact Model (Rotational + Invariant)
        """
        self.logger.info("Processing Stream B (Impact)...")

        df_meta["step"] = df_meta["step"].astype(int)
        df_meta["nfl_player_id_1"] = df_meta["nfl_player_id_1"].astype(str)
        df_tracking["nfl_player_id"] = df_tracking["nfl_player_id"].astype(str)
        df_tracking["step"] = df_tracking["step"].astype(int)

        # Merge Tracking for Player 1 (Player 2 is Ground)
        df = pd.merge(
            df_meta,
            df_tracking,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )

        # --- Rotational Differentials ---
        # 1. Project Velocity onto Orientation
        # Convert degrees to radians
        # Note: Check data description for orientation/direction zero-point.
        # Usually, they are consistent relative to each other.
        # Direction: angle of motion. Orientation: angle of player facing.

        # Fill NaNs
        df["speed"] = df["speed"].fillna(0)
        df["direction"] = df["direction"].fillna(0)
        df["orientation"] = df["orientation"].fillna(0)
        df["acceleration"] = df["acceleration"].fillna(0)

        rad_dir = np.radians(df["direction"])
        rad_orient = np.radians(df["orientation"])

        # Velocity Vector
        v_x = df["speed"] * np.sin(
            rad_dir
        )  # Standard assumption: 0 is North (Y), 90 is East (X) or similar.
        v_y = df["speed"] * np.cos(
            rad_dir
        )  # Exact axis doesn't matter as long as projection is relative.

        # Orientation Unit Vector
        o_x = np.sin(rad_orient)
        o_y = np.cos(rad_orient)

        # Orthogonal Orientation Vector (Sway axis)
        # Rotate 90 deg
        o_perp_x = o_y
        o_perp_y = -o_x

        # Project V onto O (Surge) and O_perp (Sway)
        # Dot product
        df["v_surge"] = v_x * o_x + v_y * o_y
        df["v_sway"] = v_x * o_perp_x + v_y * o_perp_y

        # 2. Differentiate to get Ego-Acceleration and Ego-Jerk
        df.sort_values(by=["game_play", "nfl_player_id_1", "step"], inplace=True)
        grp = df.groupby(["game_play", "nfl_player_id_1"])

        # First Derivative (Ego-Accel)
        # Time step is 0.1s. We just take diff, scale is monotonic.
        df["ego_accel_surge"] = grp["v_surge"].diff().fillna(0)
        df["ego_accel_sway"] = grp["v_sway"].diff().fillna(0)

        # Second Derivative (Ego-Jerk)
        df["ego_jerk_surge"] = grp["ego_accel_surge"].diff().fillna(0)
        df["ego_jerk_sway"] = grp["ego_accel_sway"].diff().fillna(0)

        # --- Final Selection ---
        feature_cols = Config.STREAM_B_FEATURES

        # Fill NaNs
        df[feature_cols] = df[feature_cols].fillna(0)

        return df[feature_cols], df["contact"].values, df["contact_id"].values

    def _compute_visual_features(self, df, df_helmets):
        """
        Calculates IoU metrics for Stream A.
        """
        # Ensure datetime alignment
        # Labels have 'datetime' (string or ts). Helmets have 'datetime' (ts).
        if not np.issubdtype(df["datetime"].dtype, np.datetime64):
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        # We match on nearest frame or exact datetime.
        # Helmets are 59.94Hz, Labels 10Hz.
        # We can use merge_asof or just merge on game_play and map steps to frames?
        # Simpler: Map labels to nearest helmet frame.
        # But `df_helmets` is large.

        # Strategy:
        # 1. Filter helmets to relevant game_plays (already done in DataManager usually, but let's be safe).
        # 2. Pivot/Merge helmets to get P1 box and P2 box on the same row.

        # Prepare Helmet Data
        # We need columns: game_play, view, nfl_player_id, datetime, left, width, top, height
        # We need to join this twice: once for P1, once for P2.

        # To handle views (Sideline/Endzone), we can pivot or join separately.
        # Let's join separately for clarity.

        views = ["Sideline", "Endzone"]

        for view in views:
            # Filter helmets for this view
            h_view = df_helmets[df_helmets["view"] == view].copy()
            # Ensure ID is string
            h_view["nfl_player_id"] = h_view["nfl_player_id"].astype(str)

            # Sort for merge_asof
            h_view.sort_values("datetime", inplace=True)
            df.sort_values("datetime", inplace=True)

            # Merge P1
            # We use merge_asof with a tolerance (e.g., 50ms)
            p1_cols = ["left", "width", "top", "height"]
            h_p1 = h_view[["game_play", "nfl_player_id", "datetime"] + p1_cols].rename(
                columns={c: f"{c}_p1_{view}" for c in p1_cols}
            )

            # merge_asof requires sorting. Grouping by game_play/player is hard in asof.
            # Instead, we merge on exact keys if we can, or use a tolerance join.
            # Given the complexity and runtime, let's try a simplified approach:
            # Join on game_play and nfl_player_id, then filter by time difference? Too slow.
            # merge_asof by="game_play" is not supported for multiple by columns in older pandas,
            # but standard is by group.
            # We can concatenate game_play + player_id as a key.

            df["merge_key_p1"] = df["game_play"] + "_" + df["nfl_player_id_1"]
            h_p1["merge_key_p1"] = h_p1["game_play"] + "_" + h_p1["nfl_player_id"]

            df = pd.merge_asof(
                df,
                h_p1.drop(columns=["game_play", "nfl_player_id"]),
                on="datetime",
                by="merge_key_p1",
                tolerance=pd.Timedelta("0.1s"),
                direction="nearest",
            )

            # Merge P2
            h_p2 = h_view[["game_play", "nfl_player_id", "datetime"] + p1_cols].rename(
                columns={c: f"{c}_p2_{view}" for c in p1_cols}
            )
            df["merge_key_p2"] = df["game_play"] + "_" + df["nfl_player_id_2"]
            h_p2["merge_key_p2"] = h_p2["game_play"] + "_" + h_p2["nfl_player_id"]

            df = pd.merge_asof(
                df,
                h_p2.drop(columns=["game_play", "nfl_player_id"]),
                on="datetime",
                by="merge_key_p2",
                tolerance=pd.Timedelta("0.1s"),
                direction="nearest",
            )

            # Calculate IoU for this view
            # Box: [left, top, right, bottom]
            # right = left + width, bottom = top + height

            l1 = df[f"left_p1_{view}"]
            t1 = df[f"top_p1_{view}"]
            r1 = l1 + df[f"width_p1_{view}"]
            b1 = t1 + df[f"height_p1_{view}"]

            l2 = df[f"left_p2_{view}"]
            t2 = df[f"top_p2_{view}"]
            r2 = l2 + df[f"width_p2_{view}"]
            b2 = t2 + df[f"height_p2_{view}"]

            # Intersection
            x_left = np.maximum(l1, l2)
            y_top = np.maximum(t1, t2)
            x_right = np.minimum(r1, r2)
            y_bottom = np.minimum(b1, b2)

            intersection_area = np.maximum(0, x_right - x_left) * np.maximum(
                0, y_bottom - y_top
            )

            area1 = (r1 - l1) * (b1 - t1)
            area2 = (r2 - l2) * (b2 - t2)

            union_area = area1 + area2 - intersection_area

            # Avoid division by zero
            iou = intersection_area / (union_area + 1e-6)

            # If boxes are missing (NaN), IoU is 0 (or -1 sentinel, but 0 makes sense for no overlap)
            df[f"iou_{view}"] = iou.fillna(0)

        # Consensus Metrics
        df["max_iou"] = df[["iou_Sideline", "iou_Endzone"]].max(axis=1)
        df["min_iou"] = df[["iou_Sideline", "iou_Endzone"]].min(axis=1)
        df["iou_diff"] = (df["iou_Sideline"] - df["iou_Endzone"]).abs()

        # Cleanup temp columns
        drop_cols = [
            c
            for c in df.columns
            if "merge_key" in c
            or "_p1_Sideline" in c
            or "_p2_Sideline" in c
            or "_p1_Endzone" in c
            or "_p2_Endzone" in c
        ]
        df.drop(columns=drop_cols, inplace=True)

        return df
