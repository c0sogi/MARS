import pandas as pd
import numpy as np
import os
import gc
from typing import Tuple, List, Dict, Optional

from library.config import Config
from library.utils import get_data_hash
from library.data_loader import load_tracking, load_helmets


class FeatureEngineer:
    """
    Implements the feature engineering pipeline for the Dual-Stream GBDT.
    Handles Kinematics, Visual Geometry, and Temporal Aggregation with Robust Imputation.
    """

    def __init__(self, config_dict: Optional[Dict] = None):
        self.config = config_dict if config_dict else Config.__dict__
        self.micro_window = Config.MICRO_WINDOW
        self.macro_window = Config.MACRO_WINDOW
        self.working_dir = Config.WORKING_DIR

    def _compute_kinematics(
        self, df_tracking: pd.DataFrame, compute_jerk: bool, compute_alignment: bool
    ) -> pd.DataFrame:
        """
        Computes advanced kinematic features: Jerk and Pose-Motion Alignment.
        """
        # Ensure data is sorted for temporal diffs
        df = df_tracking.sort_values(by=["game_play", "nfl_player_id", "step"]).copy()

        # 1. Jerk (Derivative of Acceleration)
        if compute_jerk:
            # Group by player/play to ensure boundaries
            # accel is in yards/s^2. dt is 0.1s.
            # Jerk = diff(accel) / 0.1 = diff(accel) * 10
            # We use shift(1) to get previous value: J_t = (A_t - A_{t-1}) / dt
            df["accel_prev"] = df.groupby(["game_play", "nfl_player_id"])[
                "acceleration"
            ].shift(1)
            df["jerk"] = (df["acceleration"] - df["accel_prev"]) * 10.0
            df["jerk"] = df["jerk"].fillna(0.0)
            df.drop(columns=["accel_prev"], inplace=True)

        # 2. Alignment (Cosine similarity between Orientation and Direction)
        if compute_alignment:
            # Orientation: Facing angle (degrees)
            # Direction: Motion angle (degrees)
            # Convert to radians
            ori_rad = np.radians(df["orientation"].fillna(0))
            dir_rad = np.radians(df["direction"].fillna(0))

            # Alignment: 1 = moving forward, -1 = moving backward, 0 = strafing
            df["alignment"] = np.cos(ori_rad - dir_rad)

        return df

    def _add_lags(
        self,
        df: pd.DataFrame,
        cols: List[str],
        group_cols: List[str],
        shifts: List[int],
    ) -> pd.DataFrame:
        """
        Adds flattened lag features to the dataframe using vectorized shifts.
        """
        df_res = df.copy()
        # Ensure sort order
        df_res = df_res.sort_values(by=group_cols + ["step"])

        grouped = df_res.groupby(group_cols)

        for col in cols:
            for s in shifts:
                col_name = f"{col}_lag_{s}"
                # shift(s): positive s shifts data down (t takes value of t-s).
                # We want lag_s to represent t+s.
                # If s=-1 (past), we want value at t-1. shift(1) gives value of t-1 at t.
                # If s=1 (future), we want value at t+1. shift(-1) gives value of t+1 at t.
                # So we use shift(-s).
                df_res[col_name] = grouped[col].shift(-s)

        return df_res

    def _compute_visual_features(
        self, df_features: pd.DataFrame, df_helmets: pd.DataFrame, impute_value: float
    ) -> pd.DataFrame:
        """
        Computes robust visual geometry features (IoU, Dist) and aggregates them.
        """
        # 1. Map Steps to Frames
        # Formula: Frame = Round(300 + Step * 5.994)
        df_features["frame"] = np.round(300 + df_features["step"] * 5.994).astype(int)

        # Helper to get boxes
        def get_boxes(ids_df, p_col, view_name):
            h_view = df_helmets[df_helmets["view"] == view_name]
            merged = pd.merge(
                ids_df,
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
                ],
                left_on=["game_play", "frame", p_col],
                right_on=["game_play", "frame", "nfl_player_id"],
                how="left",
            )
            return merged.rename(
                columns={
                    "left": f"{view_name}_left",
                    "width": f"{view_name}_width",
                    "top": f"{view_name}_top",
                    "height": f"{view_name}_height",
                }
            )

        visual_cols = []

        # 2. Compute Instantaneous Metrics for each View
        for view in ["Sideline", "Endzone"]:
            # Get P1 boxes
            p1_boxes = get_boxes(
                df_features[["game_play", "step", "frame", "nfl_player_id_1"]],
                "nfl_player_id_1",
                view,
            )
            # Get P2 boxes
            p2_boxes = get_boxes(
                df_features[["game_play", "step", "frame", "nfl_player_id_2"]],
                "nfl_player_id_2",
                view,
            )

            # Compute centroids
            p1_cx = p1_boxes[f"{view}_left"] + p1_boxes[f"{view}_width"] / 2
            p1_cy = p1_boxes[f"{view}_top"] + p1_boxes[f"{view}_height"] / 2
            p2_cx = p2_boxes[f"{view}_left"] + p2_boxes[f"{view}_width"] / 2
            p2_cy = p2_boxes[f"{view}_top"] + p2_boxes[f"{view}_height"] / 2

            # Distance
            dist_col = f"{view}_dist"
            df_features[dist_col] = np.sqrt((p1_cx - p2_cx) ** 2 + (p1_cy - p2_cy) ** 2)

            # IoU
            l1, t1, r1, b1 = (
                p1_boxes[f"{view}_left"],
                p1_boxes[f"{view}_top"],
                p1_boxes[f"{view}_left"] + p1_boxes[f"{view}_width"],
                p1_boxes[f"{view}_top"] + p1_boxes[f"{view}_height"],
            )
            l2, t2, r2, b2 = (
                p2_boxes[f"{view}_left"],
                p2_boxes[f"{view}_top"],
                p2_boxes[f"{view}_left"] + p2_boxes[f"{view}_width"],
                p2_boxes[f"{view}_top"] + p2_boxes[f"{view}_height"],
            )

            x_left = np.maximum(l1, l2)
            y_top = np.maximum(t1, t2)
            x_right = np.minimum(r1, r2)
            y_bottom = np.minimum(b1, b2)
            inter_area = np.maximum(0, x_right - x_left) * np.maximum(
                0, y_bottom - y_top
            )
            union_area = (
                (p1_boxes[f"{view}_width"] * p1_boxes[f"{view}_height"])
                + (p2_boxes[f"{view}_width"] * p2_boxes[f"{view}_height"])
                - inter_area
            )

            iou_col = f"{view}_iou"
            df_features[iou_col] = inter_area / (union_area + 1e-6)

            visual_cols.extend([dist_col, iou_col])

        # 3. Robust Imputation
        df_features[visual_cols] = df_features[visual_cols].fillna(impute_value)

        # Removed Multi-Resolution Aggregation (Rolling) for Visual Features
        # Cite solution_lesson_node_00041: Avoid Smoothing Sharp Signals

        df_features.drop(columns=["frame"], inplace=True)
        return df_features

    def create_features(
        self,
        metadata_df: pd.DataFrame,
        stream_config: Dict,
        dataset_type: str = "train",
        load_cached_data: bool = True,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Main driver function to generate features for a specific stream.
        """
        stream_name = stream_config["name"]

        # Generate Cache Hash
        # We include the first contact_id to differentiate splits/folds
        sample_id = (
            metadata_df["contact_id"].iloc[0] if not metadata_df.empty else "empty"
        )
        cache_hash = get_data_hash(
            {"config": stream_config, "sample_id": sample_id, "len": len(metadata_df)}
        )

        cache_file_X = os.path.join(
            self.working_dir, f"{stream_name}_{dataset_type}_{cache_hash}_X.parquet"
        )
        cache_file_y = os.path.join(
            self.working_dir, f"{stream_name}_{dataset_type}_{cache_hash}_y.npy"
        )
        cache_file_ids = os.path.join(
            self.working_dir, f"{stream_name}_{dataset_type}_{cache_hash}_ids.npy"
        )

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_file_X):
            print(f"[{stream_name}] Loading cached features...")
            X = pd.read_parquet(cache_file_X)
            y = np.load(cache_file_y, allow_pickle=True)
            ids = np.load(cache_file_ids, allow_pickle=True)
            return X, y, ids

        print(f"[{stream_name}] Generating features from scratch...")

        # 2. Filter Metadata for Stream Target
        df_meta = metadata_df.copy()
        if stream_config["target_type"] == "ground":
            df_meta = df_meta[df_meta["nfl_player_id_2"] == "G"].copy()
        else:
            df_meta = df_meta[df_meta["nfl_player_id_2"] != "G"].copy()

        if df_meta.empty:
            return pd.DataFrame(), np.array([]), np.array([])

        # 3. Load and Preprocess Tracking
        # Use 'train' tracking for both train and validation splits
        tracking_set = "train" if dataset_type == "validation" else dataset_type
        df_track = load_tracking(dataset_type=tracking_set, load_cached_data=True)

        # Filter to relevant games
        relevant_games = df_meta["game_play"].unique()
        df_track = df_track[df_track["game_play"].isin(relevant_games)].copy()

        # Compute Kinematics
        compute_jerk = stream_config.get("compute_jerk", False)
        compute_alignment = stream_config.get("compute_alignment", False)
        df_track = self._compute_kinematics(df_track, compute_jerk, compute_alignment)

        # Define features to flatten
        track_feats = Config.TRACKING_COLS.copy()
        if compute_jerk:
            track_feats.append("jerk")
        if compute_alignment:
            track_feats.append("alignment")

        # Create Lags (Flattening)
        # We do this on the full tracking set for relevant games to ensure continuity
        # Cite solution_lesson_node_00041: Use sparse, long-range lags
        df_track_lagged = self._add_lags(
            df_track, track_feats, ["game_play", "nfl_player_id"], Config.LAG_STEPS
        )

        # 4. Merge P1 Features
        # Prepare tracking columns for P1
        p1_cols = [
            c
            for c in df_track_lagged.columns
            if c
            not in [
                "game_play",
                "step",
                "nfl_player_id",
                "datetime",
                "position",
                "team",
                "jersey_number",
            ]
        ]
        df_track_p1 = df_track_lagged[
            ["game_play", "step", "nfl_player_id"] + p1_cols
        ].rename(columns={c: f"p1_{c}" for c in p1_cols})

        # Left join onto metadata
        df_features = pd.merge(
            df_meta,
            df_track_p1,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        ).drop(columns=["nfl_player_id"])

        # 5. Stream Specific Logic
        if stream_config["target_type"] == "player":
            # Merge P2 Features
            df_track_p2 = df_track_lagged[
                ["game_play", "step", "nfl_player_id"] + p1_cols
            ].rename(columns={c: f"p2_{c}" for c in p1_cols})
            df_features = pd.merge(
                df_features,
                df_track_p2,
                left_on=["game_play", "step", "nfl_player_id_2"],
                right_on=["game_play", "step", "nfl_player_id"],
                how="left",
            ).drop(columns=["nfl_player_id"])

            # Interaction Features (Instantaneous at lag 0)
            df_features["p1_p2_dist"] = np.sqrt(
                (
                    df_features["p1_x_position_lag_0"]
                    - df_features["p2_x_position_lag_0"]
                )
                ** 2
                + (
                    df_features["p1_y_position_lag_0"]
                    - df_features["p2_y_position_lag_0"]
                )
                ** 2
            )
            df_features["p1_p2_speed_diff"] = (
                df_features["p1_speed_lag_0"] - df_features["p2_speed_lag_0"]
            )

            # Visual Features
            if stream_config["use_visuals"]:
                df_helmets = load_helmets(
                    dataset_type=tracking_set, load_cached_data=True
                )
                df_features = self._compute_visual_features(
                    df_features, df_helmets, stream_config["impute_visuals"]
                )

        # 6. Finalize Output
        meta_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "datetime",
            "contact",
            "video_path_sideline",
            "video_path_endzone",
            "video_path_all29",
        ]
        feature_cols = [c for c in df_features.columns if c not in meta_cols]

        X = df_features[feature_cols].copy()
        # Handle target column: 'contact' exists in train/val but is placeholder in test
        y = (
            df_features["contact"].values.astype(int)
            if "contact" in df_features.columns
            else np.zeros(len(df_features))
        )
        ids = df_features["contact_id"].values

        # Save to cache
        try:
            X.to_parquet(cache_file_X)
            np.save(cache_file_y, y)
            np.save(cache_file_ids, ids)
        except Exception as e:
            print(f"Warning: Failed to save cache for {stream_name}: {e}")

        return X, y, ids
