import pandas as pd
import numpy as np
import os
import gc
import hashlib
from library.config import Config
from library.utils import get_dataframe_hash, seed_everything
from library.data_manager import DataManager


class FeatureEngineer:
    def __init__(self):
        self.config = Config
        self.data_manager = DataManager()
        seed_everything(self.config.SEED)

        # Ensure working directory exists for caching
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

    def _get_cache_paths(self, mode):
        """Generates file paths for caching feature sets."""
        base_dir = self.config.WORKING_DIR
        suffix = ""
        if mode == "train" and self.config.DEBUG_SAMPLE_SIZE is not None:
            suffix = f"_debug_{self.config.DEBUG_SAMPLE_SIZE}"

        paths = {
            "stream_a_X": os.path.join(
                base_dir, f"features_{mode}{suffix}_streamA_X.parquet"
            ),
            "stream_a_y": os.path.join(
                base_dir, f"features_{mode}{suffix}_streamA_y.npy"
            ),
            "stream_a_ids": os.path.join(
                base_dir, f"features_{mode}{suffix}_streamA_ids.npy"
            ),
            "stream_b_X": os.path.join(
                base_dir, f"features_{mode}{suffix}_streamB_X.parquet"
            ),
            "stream_b_y": os.path.join(
                base_dir, f"features_{mode}{suffix}_streamB_y.npy"
            ),
            "stream_b_ids": os.path.join(
                base_dir, f"features_{mode}{suffix}_streamB_ids.npy"
            ),
        }
        return paths

    def _load_helmets(self, mode, relevant_plays):
        """Loads and filters helmet data, mapping frames to steps."""
        path = (
            self.config.TRAIN_HELMETS_PATH
            if mode in ["train", "validation"]
            else self.config.TEST_HELMETS_PATH
        )

        # Read columns of interest to save memory
        cols = [
            "game_play",
            "view",
            "frame",
            "nfl_player_id",
            "left",
            "width",
            "top",
            "height",
        ]
        df_helmets = pd.read_csv(path, usecols=cols)

        # Filter by relevant plays
        df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)].copy()

        # Map Frame to Step
        # Step 0 is at 5.0s (approx frame 300). 59.94 fps -> ~6 frames per 0.1s step
        # frame = 300 + step * 5.994.
        # step = (frame - 300) / 5.994
        df_helmets["step"] = ((df_helmets["frame"] - 300) / 5.994).round().astype(int)

        # Calculate Box Area (optional, but standard IoU uses coords)
        # We need x1, y1, x2, y2 for IoU
        df_helmets["x1"] = df_helmets["left"]
        df_helmets["y1"] = df_helmets["top"]
        df_helmets["x2"] = df_helmets["left"] + df_helmets["width"]
        df_helmets["y2"] = df_helmets["top"] + df_helmets["height"]

        return df_helmets

    def _compute_iou(self, df_merged, df_helmets, view_name):
        """
        Computes IoU for a specific view (Sideline/Endzone).
        merges helmets for p1 and p2 onto the main dataframe.
        """
        # Filter helmets by view
        helmets_view = df_helmets[df_helmets["view"] == view_name].copy()

        # Prepare for merge
        # We need unique keys: game_play, step, nfl_player_id
        # Handle duplicates if any (take first detection)
        helmets_view = helmets_view.drop_duplicates(
            subset=["game_play", "step", "nfl_player_id"]
        )

        # Merge P1
        df_merged = df_merged.merge(
            helmets_view[
                ["game_play", "step", "nfl_player_id", "x1", "y1", "x2", "y2"]
            ].add_suffix("_p1"),
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play_p1", "step_p1", "nfl_player_id_p1"],
            how="left",
        )

        # Merge P2
        df_merged = df_merged.merge(
            helmets_view[
                ["game_play", "step", "nfl_player_id", "x1", "y1", "x2", "y2"]
            ].add_suffix("_p2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_p2", "step_p2", "nfl_player_id_p2"],
            how="left",
        )

        # Calculate IoU
        # Intersection
        xi1 = np.maximum(df_merged["x1_p1"], df_merged["x1_p2"])
        yi1 = np.maximum(df_merged["y1_p1"], df_merged["y1_p2"])
        xi2 = np.minimum(df_merged["x2_p1"], df_merged["x2_p2"])
        yi2 = np.minimum(df_merged["y2_p1"], df_merged["y2_p2"])

        inter_width = np.maximum(0, xi2 - xi1)
        inter_height = np.maximum(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        box1_area = (df_merged["x2_p1"] - df_merged["x1_p1"]) * (
            df_merged["y2_p1"] - df_merged["y1_p1"]
        )
        box2_area = (df_merged["x2_p2"] - df_merged["x1_p2"]) * (
            df_merged["y2_p2"] - df_merged["y1_p2"]
        )
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = np.where(union_area > 0, inter_area / union_area, 0.0)

        # Cleanup merge columns
        drop_cols = [
            c for c in df_merged.columns if c.endswith("_p1") or c.endswith("_p2")
        ]
        # Keep original p1/p2 tracking cols if they exist, only drop helmet cols
        # Helmet cols: x1, y1, x2, y2, nfl_player_id
        helmet_cols = [
            "x1_p1",
            "y1_p1",
            "x2_p1",
            "y2_p1",
            "nfl_player_id_p1",
            "x1_p2",
            "y1_p2",
            "x2_p2",
            "y2_p2",
            "nfl_player_id_p2",
        ]
        df_merged = df_merged.drop(
            columns=[c for c in helmet_cols if c in df_merged.columns]
        )

        return iou

    def _compute_stream_a(self, df, df_helmets):
        """
        Stream A: Interaction Model (Player-Player)
        Robust Consistency: Positional Differentials + Visual Consensus
        """
        # Filter for Player-Player interactions
        df_a = df[df["nfl_player_id_2"] != "G"].copy()

        if len(df_a) == 0:
            return pd.DataFrame(), np.array([]), np.array([])

        # 1. Positional Differentials
        # Calculate Distance
        df_a["distance"] = np.sqrt(
            (df_a["x_position_p1"] - df_a["x_position_p2"]) ** 2
            + (df_a["y_position_p1"] - df_a["y_position_p2"]) ** 2
        )

        # Calculate Closure Rate (Time Derivative of Distance)
        # Sort by play and step to ensure correct diff
        df_a = df_a.sort_values(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )

        # Group shift
        grp = df_a.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])
        df_a["prev_dist"] = grp["distance"].shift(1)
        # Closure rate: Positive means getting closer. -(d_t - d_{t-1}) / 0.1
        df_a["closure_rate"] = -(df_a["distance"] - df_a["prev_dist"]) / 0.1
        df_a["closure_rate"] = df_a["closure_rate"].fillna(0)

        # 2. Visual Features
        # Compute IoU for Sideline and Endzone
        iou_sideline = self._compute_iou(df_a, df_helmets, "Sideline")
        iou_endzone = self._compute_iou(df_a, df_helmets, "Endzone")

        df_a["iou_sideline"] = iou_sideline
        df_a["iou_endzone"] = iou_endzone

        # Visual Consensus
        df_a["max_iou_t0"] = np.maximum(df_a["iou_sideline"], df_a["iou_endzone"])
        df_a["min_iou_t0"] = np.minimum(df_a["iou_sideline"], df_a["iou_endzone"])
        df_a["iou_diff_t0"] = np.abs(df_a["iou_sideline"] - df_a["iou_endzone"])

        # 3. Visual Pyramids (Lags)
        # We need to re-group because we just added columns
        grp = df_a.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        for lag in self.config.VISUAL_LAGS:
            if lag == 0:
                continue

            # Future and Past lags? Config says just lags, usually implies past or centered.
            # Given "Pyramids" usually implies surrounding context.
            # Let's assume past lags for causality or symmetric if allowed.
            # Description mentions t, t+4, t+8... but standard tracking is usually causal or symmetric window.
            # Let's use symmetric shifts if possible, or just past.
            # "Pyramids at sparse lags t, t+/-4..." -> Symmetric.

            # Forward (Future)
            df_a[f"max_iou_t{lag}"] = (
                grp["max_iou_t0"].shift(-lag).fillna(-1)
            )  # Sentinel -1 or 0
            df_a[f"min_iou_t{lag}"] = grp["min_iou_t0"].shift(-lag).fillna(-1)
            df_a[f"iou_diff_t{lag}"] = grp["iou_diff_t0"].shift(-lag).fillna(-1)

            # We could also do backward, but Config just lists t4, t8. I'll map t4 to +4 (future) or just use one direction.
            # Given "Predict moments of contact", future info is available in post-processing/offline.
            # I will implement +lag (future) as it's most predictive of *impending* contact.

        # 4. Visual Looming & Consistency
        # Looming: Rate of change of Max IoU
        df_a["prev_max_iou"] = grp["max_iou_t0"].shift(1).fillna(0)
        df_a["visual_looming_rate"] = (df_a["max_iou_t0"] - df_a["prev_max_iou"]) / 0.1

        # Consistency Score: Normalized Closure Rate - Looming
        # Normalize closure rate roughly to 0-1 range?
        # Or just raw difference. "Large divergence indicates phantom".
        # Let's use raw difference.
        df_a["consistency_score"] = df_a["closure_rate"] - (
            df_a["visual_looming_rate"] * 10
        )  # Scale factor heuristic

        # 5. Kinematics (Renaming/Selecting)
        df_a["speed_p1"] = df_a["speed_p1"].fillna(0)
        df_a["speed_p2"] = df_a["speed_p2"].fillna(0)
        df_a["accel_p1"] = df_a["acceleration_p1"].fillna(0)
        df_a["accel_p2"] = df_a["acceleration_p2"].fillna(0)

        # Select Features
        features = self.config.FEATURES_STREAM_A
        # Ensure all features exist
        for col in features:
            if col not in df_a.columns:
                df_a[col] = 0.0

        X = df_a[features].copy()
        y = df_a["contact"].values.astype(int)
        ids = df_a["contact_id"].values

        return X, y, ids

    def _compute_stream_b(self, df):
        """
        Stream B: Impact Model (Player-Ground)
        Sensitive Dynamics: High-Order Physics / Rotational Differentials
        """
        # Filter for Player-Ground interactions
        df_b = df[df["nfl_player_id_2"] == "G"].copy()

        if len(df_b) == 0:
            return pd.DataFrame(), np.array([]), np.array([])

        # 1. Rotational Physics
        # Convert degrees to radians
        # direction: 0..360, orientation: 0..360
        # We need angle of motion relative to orientation
        # Note: In NFL data, 0 is usually Y-axis (North), 90 is X-axis (East).
        # But relative angle difference is invariant to the zero reference.

        rad_dir = np.radians(df_b["direction_p1"].fillna(0))
        rad_orient = np.radians(df_b["orientation_p1"].fillna(0))
        theta = rad_dir - rad_orient

        speed = df_b["speed_p1"].fillna(0)

        # Surge: Forward/Backward velocity relative to facing
        df_b["v_surge"] = speed * np.cos(theta)
        # Sway: Left/Right velocity relative to facing
        df_b["v_sway"] = speed * np.sin(theta)

        # 2. Differentials (Ego-Accel, Ego-Jerk)
        df_b = df_b.sort_values(["game_play", "nfl_player_id_1", "step"])
        grp = df_b.groupby(["game_play", "nfl_player_id_1"])

        # Ego-Acceleration
        df_b["prev_v_surge"] = grp["v_surge"].shift(1).fillna(0)
        df_b["prev_v_sway"] = grp["v_sway"].shift(1).fillna(0)

        df_b["ego_accel_surge"] = (df_b["v_surge"] - df_b["prev_v_surge"]) / 0.1
        df_b["ego_accel_sway"] = (df_b["v_sway"] - df_b["prev_v_sway"]) / 0.1

        # Ego-Jerk
        df_b["prev_acc_surge"] = grp["ego_accel_surge"].shift(1).fillna(0)
        df_b["prev_acc_sway"] = grp["ego_accel_sway"].shift(1).fillna(0)

        df_b["ego_jerk_surge"] = (
            df_b["ego_accel_surge"] - df_b["prev_acc_surge"]
        ) / 0.1
        df_b["ego_jerk_sway"] = (df_b["ego_accel_sway"] - df_b["prev_acc_sway"]) / 0.1

        # 3. Energy
        df_b["surge_energy"] = df_b["v_surge"] ** 2
        df_b["sway_energy"] = df_b["v_sway"] ** 2

        # 4. Invariant Baseline
        df_b["speed"] = speed
        df_b["acceleration"] = df_b["acceleration_p1"].fillna(0)

        # Select Features
        features = self.config.FEATURES_STREAM_B
        for col in features:
            if col not in df_b.columns:
                df_b[col] = 0.0

        X = df_b[features].copy()
        y = df_b["contact"].values.astype(int)
        ids = df_b["contact_id"].values

        return X, y, ids

    def process_features(self, mode: str, load_cached_data: bool = True):
        """
        Main pipeline to generate features for both streams.
        """
        paths = self._get_cache_paths(mode)

        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and cache_exists:
            print(f"Loading cached features for {mode}...")
            try:
                data = {
                    "stream_a": (
                        pd.read_parquet(paths["stream_a_X"]),
                        np.load(paths["stream_a_y"]),
                        np.load(paths["stream_a_ids"]),
                    ),
                    "stream_b": (
                        pd.read_parquet(paths["stream_b_X"]),
                        np.load(paths["stream_b_y"]),
                        np.load(paths["stream_b_ids"]),
                    ),
                }
                return data
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # Compute from scratch
        print(f"Computing features for {mode}...")

        # 1. Load Merged Data
        df_merged = self.data_manager.get_data(mode, load_cached_data=load_cached_data)

        # 2. Load Helmets (only needed for Stream A)
        relevant_plays = df_merged["game_play"].unique()
        df_helmets = self._load_helmets(mode, relevant_plays)

        # 3. Compute Stream A
        print("Computing Stream A (Interaction)...")
        X_a, y_a, ids_a = self._compute_stream_a(df_merged, df_helmets)

        # 4. Compute Stream B
        print("Computing Stream B (Impact)...")
        X_b, y_b, ids_b = self._compute_stream_b(df_merged)

        # 5. Save Cache
        print(f"Saving features to {self.config.WORKING_DIR}...")

        # Stream A
        if not X_a.empty:
            X_a.to_parquet(paths["stream_a_X"], index=False)
            np.save(paths["stream_a_y"], y_a)
            np.save(paths["stream_a_ids"], ids_a)
        else:
            # Handle empty case (unlikely but safe)
            pd.DataFrame(columns=self.config.FEATURES_STREAM_A).to_parquet(
                paths["stream_a_X"]
            )
            np.save(paths["stream_a_y"], np.array([]))
            np.save(paths["stream_a_ids"], np.array([]))

        # Stream B
        if not X_b.empty:
            X_b.to_parquet(paths["stream_b_X"], index=False)
            np.save(paths["stream_b_y"], y_b)
            np.save(paths["stream_b_ids"], ids_b)
        else:
            pd.DataFrame(columns=self.config.FEATURES_STREAM_B).to_parquet(
                paths["stream_b_X"]
            )
            np.save(paths["stream_b_y"], np.array([]))
            np.save(paths["stream_b_ids"], np.array([]))

        # Clean up
        del df_merged, df_helmets
        gc.collect()

        return {"stream_a": (X_a, y_a, ids_a), "stream_b": (X_b, y_b, ids_b)}
