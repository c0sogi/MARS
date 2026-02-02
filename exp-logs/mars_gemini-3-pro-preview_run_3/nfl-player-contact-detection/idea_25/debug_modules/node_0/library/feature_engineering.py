import pandas as pd
import numpy as np
import os
from library.config import FEATURE_CONFIG, WORKING_DIR
from library.utils import CacheManager
from library.data_loader import merge_tracking_data


class FeatureEngineer:
    def __init__(self):
        self.cache_manager = CacheManager(WORKING_DIR)
        self.config = FEATURE_CONFIG

    def generate_features(
        self, df_labels, df_tracking, df_helmets, mode="train", load_cached_data=True
    ):
        """
        Main entry point to generate features for both streams.
        Splits data into Player-Player (Stream A) and Player-Ground (Stream B).
        """
        # Define cache keys based on mode and config
        cache_key_a = f"features_{mode}_streamA"
        cache_key_b = f"features_{mode}_streamB"

        # --- Stream A: Player vs Player ---
        # Filter for non-ground contacts
        df_a = df_labels[df_labels["nfl_player_id_2"] != "G"].copy()

        # Check cache for Stream A
        file_X_a = self.cache_manager.get_hashed_filename(
            cache_key_a + "_X", self.config["stream_a"], "parquet"
        )
        file_ids_a = self.cache_manager.get_hashed_filename(
            cache_key_a + "_ids", self.config["stream_a"], "npy"
        )
        file_y_a = self.cache_manager.get_hashed_filename(
            cache_key_a + "_y", self.config["stream_a"], "npy"
        )

        if load_cached_data and self.cache_manager.exists(file_X_a):
            X_a = self.cache_manager.load(file_X_a)
            ids_a = self.cache_manager.load(file_ids_a)
            y_a = self.cache_manager.load(file_y_a)
        else:
            # Compute Stream A
            X_a, ids_a, y_a = self._compute_stream_a(df_a, df_tracking, df_helmets)
            # Save
            self.cache_manager.save(file_X_a, X_a)
            self.cache_manager.save(file_ids_a, ids_a)
            self.cache_manager.save(file_y_a, y_a)

        # --- Stream B: Player vs Ground ---
        # Filter for ground contacts
        df_b = df_labels[df_labels["nfl_player_id_2"] == "G"].copy()

        # Check cache for Stream B
        file_X_b = self.cache_manager.get_hashed_filename(
            cache_key_b + "_X", self.config["stream_b"], "parquet"
        )
        file_ids_b = self.cache_manager.get_hashed_filename(
            cache_key_b + "_ids", self.config["stream_b"], "npy"
        )
        file_y_b = self.cache_manager.get_hashed_filename(
            cache_key_b + "_y", self.config["stream_b"], "npy"
        )

        if load_cached_data and self.cache_manager.exists(file_X_b):
            X_b = self.cache_manager.load(file_X_b)
            ids_b = self.cache_manager.load(file_ids_b)
            y_b = self.cache_manager.load(file_y_b)
        else:
            # Compute Stream B
            X_b, ids_b, y_b = self._compute_stream_b(df_b, df_tracking)
            # Save
            self.cache_manager.save(file_X_b, X_b)
            self.cache_manager.save(file_ids_b, ids_b)
            self.cache_manager.save(file_y_b, y_b)

        return {
            "stream_a": {"X": X_a, "y": y_a, "ids": ids_a},
            "stream_b": {"X": X_b, "y": y_b, "ids": ids_b},
        }

    def _compute_stream_a(self, df_labels, df_tracking, df_helmets):
        """
        Engineers features for Stream A (Interaction Model).
        Focus: Cross-Modal Alignment (Visual vs Physical).
        """
        if df_labels.empty:
            return pd.DataFrame(), np.array([]), np.array([])

        # 1. Merge Tracking Data
        df_merged = merge_tracking_data(df_labels, df_tracking)

        # 2. Process Visuals (Helmets)
        # Map 60Hz frames to 10Hz steps
        # Formula: step approx (frame - 300) / 6
        # We aggregate helmets to step level
        if df_helmets is not None and not df_helmets.empty:
            df_helmets = df_helmets.copy()
            df_helmets["step"] = ((df_helmets["frame"] - 300) / 6).round().astype(int)

            # Filter relevant game_plays
            relevant_plays = df_merged["game_play"].unique()
            df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)]

            # Aggregate boxes per step (take mean of boxes in that 0.1s window)
            # We need boxes for specific players
            # Group by game_play, step, view, nfl_player_id
            df_vis_agg = (
                df_helmets.groupby(["game_play", "step", "view", "nfl_player_id"])[
                    ["left", "width", "top", "height"]
                ]
                .mean()
                .reset_index()
            )

            # Pivot to separate views
            # We need to join this onto the main dataframe for P1 and P2
            # Helper to merge specific player visual info
            def merge_view_info(df_main, df_vis, suffix):
                # Merge Sideline
                df_side = df_vis[df_vis["view"] == "Sideline"].copy()
                df_main = pd.merge(
                    df_main,
                    df_side,
                    left_on=["game_play", "step", f"nfl_player_id_{suffix}"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                    suffixes=("", f"_side_{suffix}"),
                )
                df_main.drop(
                    columns=["nfl_player_id", "view"], inplace=True, errors="ignore"
                )

                # Merge Endzone
                df_end = df_vis[df_vis["view"] == "Endzone"].copy()
                df_main = pd.merge(
                    df_main,
                    df_end,
                    left_on=["game_play", "step", f"nfl_player_id_{suffix}"],
                    right_on=["game_play", "step", "nfl_player_id"],
                    how="left",
                    suffixes=("", f"_end_{suffix}"),
                )
                df_main.drop(
                    columns=["nfl_player_id", "view"], inplace=True, errors="ignore"
                )

                # Rename columns manually if suffixes didn't catch (first merge doesn't add suffix to left)
                # The merge above adds _side_{suffix} to the right columns (left, width, etc)
                # We need to rename the raw box cols
                rename_map = {}
                for box_col in ["left", "width", "top", "height"]:
                    if f"{box_col}_side_{suffix}" not in df_main.columns:
                        # This happens if column names collided differently or it's the first merge
                        # Actually, pandas merge suffixes applies to overlapping columns.
                        # Since df_main doesn't have 'left' initially, it stays 'left'.
                        # We must rename explicitly.
                        pass

                # To be safe, let's rename df_vis columns before merging
                return df_main

            # Pre-rename visual columns for safety
            df_vis_side = df_vis_agg[df_vis_agg["view"] == "Sideline"].rename(
                columns={c: f"{c}_side" for c in ["left", "width", "top", "height"]}
            )
            df_vis_end = df_vis_agg[df_vis_agg["view"] == "Endzone"].rename(
                columns={c: f"{c}_end" for c in ["left", "width", "top", "height"]}
            )

            # Merge P1
            df_merged = pd.merge(
                df_merged,
                df_vis_side,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id", "view"], errors="ignore")
            df_merged = df_merged.rename(
                columns={
                    c: f"{c}_p1"
                    for c in ["left_side", "width_side", "top_side", "height_side"]
                }
            )

            df_merged = pd.merge(
                df_merged,
                df_vis_end,
                left_on=["game_play", "step", "nfl_player_id_1"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id", "view"], errors="ignore")
            df_merged = df_merged.rename(
                columns={
                    c: f"{c}_p1"
                    for c in ["left_end", "width_end", "top_end", "height_end"]
                }
            )

            # Merge P2
            df_merged = pd.merge(
                df_merged,
                df_vis_side,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id", "view"], errors="ignore")
            df_merged = df_merged.rename(
                columns={
                    c: f"{c}_p2"
                    for c in ["left_side", "width_side", "top_side", "height_side"]
                }
            )

            df_merged = pd.merge(
                df_merged,
                df_vis_end,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id", "view"], errors="ignore")
            df_merged = df_merged.rename(
                columns={
                    c: f"{c}_p2"
                    for c in ["left_end", "width_end", "top_end", "height_end"]
                }
            )

            # Calculate IoU
            def calc_iou(row, view):
                # p1 box
                l1, w1, t1, h1 = (
                    row.get(f"left_{view}_p1"),
                    row.get(f"width_{view}_p1"),
                    row.get(f"top_{view}_p1"),
                    row.get(f"height_{view}_p1"),
                )
                # p2 box
                l2, w2, t2, h2 = (
                    row.get(f"left_{view}_p2"),
                    row.get(f"width_{view}_p2"),
                    row.get(f"top_{view}_p2"),
                    row.get(f"height_{view}_p2"),
                )

                if pd.isna(l1) or pd.isna(l2):
                    return -1.0  # Sentinel for missing visual

                # Intersection
                x_left = max(l1, l2)
                y_top = max(t1, t2)
                x_right = min(l1 + w1, l2 + w2)
                y_bottom = min(t1 + h1, t2 + h2)

                if x_right < x_left or y_bottom < y_top:
                    return 0.0

                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                union_area = (w1 * h1) + (w2 * h2) - intersection_area

                if union_area <= 0:
                    return 0.0
                return intersection_area / union_area

            df_merged["iou_sideline"] = df_merged.apply(
                lambda r: calc_iou(r, "side"), axis=1
            )
            df_merged["iou_endzone"] = df_merged.apply(
                lambda r: calc_iou(r, "end"), axis=1
            )
            df_merged["iou_diff"] = (
                df_merged["iou_sideline"] - df_merged["iou_endzone"]
            ).abs()

            # Impute missing IoU with sentinel
            df_merged.fillna(
                {"iou_sideline": -999, "iou_endzone": -999, "iou_diff": -999},
                inplace=True,
            )

        else:
            # No visual data available
            df_merged["iou_sideline"] = -999
            df_merged["iou_endzone"] = -999
            df_merged["iou_diff"] = -999

        # 3. Base Physical Features
        # Distance
        df_merged["distance"] = np.sqrt(
            (df_merged["x_position_p1"] - df_merged["x_position_p2"]) ** 2
            + (df_merged["y_position_p1"] - df_merged["y_position_p2"]) ** 2
        )

        # Relative Speed (Scalar diff)
        df_merged["speed_rel"] = (df_merged["speed_p1"] - df_merged["speed_p2"]).abs()
        df_merged["accel_rel"] = (
            df_merged["acceleration_p1"] - df_merged["acceleration_p2"]
        ).abs()

        # 4. Cross-Modal Alignment (Temporal)
        # Sort for lag calculations
        df_merged.sort_values(
            by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
        )

        # We need to group by pair to calculate diffs
        # Create a pair ID for grouping
        df_merged["pair_id"] = (
            df_merged["game_play"]
            + "_"
            + df_merged["nfl_player_id_1"].astype(str)
            + "_"
            + df_merged["nfl_player_id_2"].astype(str)
        )

        # Calculate rates for specific lags (e.g., 4 steps = 0.4s)
        lag = 4
        grp = df_merged.groupby("pair_id")

        # Visual Looming Rate: d(IoU)/dt
        # Use max of sideline/endzone to be robust to occlusion in one view
        df_merged["max_iou"] = df_merged[["iou_sideline", "iou_endzone"]].max(axis=1)
        df_merged["visual_looming_rate"] = grp["max_iou"].diff(lag).fillna(0)

        # Physical Closure Rate: d(Dist)/dt
        # Note: If distance decreases, closure is positive. So -diff.
        df_merged["physical_closure_rate"] = -grp["distance"].diff(lag).fillna(0)

        # View Disagreement Trend
        df_merged["view_disagreement_trend"] = grp["iou_diff"].diff(lag).fillna(0)

        # Alignment Features
        df_merged["looming_closure_product"] = (
            df_merged["visual_looming_rate"] * df_merged["physical_closure_rate"]
        )
        # Ratio: Add epsilon to avoid div by zero
        df_merged["looming_closure_ratio"] = df_merged["visual_looming_rate"] / (
            df_merged["physical_closure_rate"].abs() + 1e-6
        )

        # 5. Select Features
        features = self.config["stream_a"]["features"]
        # Ensure all features exist (fill missing if logic skipped)
        for col in features:
            if col not in df_merged.columns:
                df_merged[col] = 0.0

        X = df_merged[features].copy()
        y = (
            df_merged["contact"].values
            if "contact" in df_merged.columns
            else np.zeros(len(df_merged))
        )
        ids = df_merged["contact_id"].values

        return X, ids, y

    def _compute_stream_b(self, df_labels, df_tracking):
        """
        Engineers features for Stream B (Impact Model).
        Focus: Rotational-Difference Dynamics (Sway vs Surge).
        """
        if df_labels.empty:
            return pd.DataFrame(), np.array([]), np.array([])

        # 1. Merge Tracking (Only P1 needed, P2 is Ground)
        # merge_tracking_data handles P1 and P2. P2 cols will be NaN.
        df_merged = merge_tracking_data(df_labels, df_tracking)

        # 2. Rotational Dynamics
        # Convert degrees to radians.
        # Tracking: 0 is usually Y-axis (North), 90 is X-axis (East).
        # Standard math: 0 is X, 90 is Y.
        # However, we only care about the *difference* between direction and orientation.
        # As long as both are in the same reference frame, subtraction works.

        # Fill NaNs in tracking
        df_merged.fillna(0, inplace=True)

        # Orientation and Direction are in degrees (0-360)
        theta_orient = np.radians(df_merged["orientation_p1"])
        theta_dir = np.radians(df_merged["direction_p1"])

        # Delta angle
        delta_theta = theta_dir - theta_orient

        # Velocity Projections
        # Surge: Component aligned with orientation (Forward/Backward)
        # Sway: Component orthogonal to orientation (Lateral/Strafing)
        speed = df_merged["speed_p1"]
        df_merged["v_surge"] = speed * np.cos(delta_theta)
        df_merged["v_sway"] = speed * np.sin(delta_theta)

        # Energy Terms
        df_merged["energy_surge"] = 0.5 * (df_merged["v_surge"] ** 2)
        df_merged["energy_sway"] = 0.5 * (df_merged["v_sway"] ** 2)

        # 3. Ego-Jerk (Derivative of Acceleration Magnitude)
        df_merged.sort_values(by=["game_play", "nfl_player_id_1", "step"], inplace=True)
        grp = df_merged.groupby(["game_play", "nfl_player_id_1"])

        # Calculate jerk (diff of acceleration)
        # Lag 1 step (0.1s)
        df_merged["ego_jerk"] = grp["acceleration_p1"].diff(1).fillna(0) / 0.1

        # 4. Rename base features to match config
        df_merged["speed"] = df_merged["speed_p1"]
        df_merged["acceleration"] = df_merged["acceleration_p1"]
        df_merged["sa"] = df_merged["sa_p1"]

        # 5. Select Features
        features = self.config["stream_b"]["features"]
        for col in features:
            if col not in df_merged.columns:
                df_merged[col] = 0.0

        X = df_merged[features].copy()
        y = (
            df_merged["contact"].values
            if "contact" in df_merged.columns
            else np.zeros(len(df_merged))
        )
        ids = df_merged["contact_id"].values

        return X, ids, y
