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

        # Ensure nfl_player_id is string to match tracking data
        df_helmets["nfl_player_id"] = df_helmets["nfl_player_id"].astype(str)

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

    def _add_lags(self, df, cols, lags, groupby_cols):
        """Helper to add lag features."""
        grp = df.groupby(groupby_cols)
        for col in cols:
            for lag in lags:
                # Negative lag = past, Positive lag = future
                # shift(k) shifts data down k spots (t takes value from t-k)
                # We want feature at t-k. So we shift(k).
                # We want feature at t+k. So we shift(-k).
                # Our LAG_STEPS are [-15, ..., 15].
                # If lag is -15 (past), we want shift(15).
                # If lag is 15 (future), we want shift(-15).

                # Note: shift(1) puts t-1 value at t.
                shift_val = -lag
                # If lag is -1 (past), we want value from t-1. shift(1) brings t-1 to t.
                # So shift(-lag) is correct?
                # lag = -1. shift(1). Correct.
                # lag = 1 (future). shift(-1). Correct.

                df[f"{col}_lag{lag}"] = grp[col].shift(-lag).fillna(0)
        return df

    def _compute_stream_a(self, df, df_helmets):
        """
        Stream A: Interaction Model (Player-Player)
        Relational Dynamics + Visual Trajectory
        """
        df_a = df[df["nfl_player_id_2"] != "G"].copy()
        if len(df_a) == 0:
            return pd.DataFrame(), np.array([]), np.array([])

        # 1. Base Features
        df_a["distance"] = np.sqrt(
            (df_a["x_position_p1"] - df_a["x_position_p2"]) ** 2
            + (df_a["y_position_p1"] - df_a["y_position_p2"]) ** 2
        )

        # Sort for temporal ops
        df_a = df_a.sort_values(
            ["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"]
        )
        grp = df_a.groupby(["game_play", "nfl_player_id_1", "nfl_player_id_2"])

        # Finite Difference Closure Rate (Cite solution_lesson_node_00080)
        df_a["prev_dist"] = grp["distance"].shift(1)
        df_a["closure_rate"] = -(df_a["distance"] - df_a["prev_dist"]) / 0.1
        df_a["closure_rate"] = df_a["closure_rate"].fillna(0)

        # Kinematics
        df_a["speed_p1"] = df_a["speed_p1"].fillna(0)
        df_a["speed_p2"] = df_a["speed_p2"].fillna(0)
        df_a["accel_p1"] = df_a["acceleration_p1"].fillna(0)
        df_a["accel_p2"] = df_a["acceleration_p2"].fillna(0)

        # 2. Visual Features
        iou_sideline = self._compute_iou(df_a, df_helmets, "Sideline")
        iou_endzone = self._compute_iou(df_a, df_helmets, "Endzone")

        df_a["max_iou_t0"] = np.maximum(iou_sideline, iou_endzone)
        df_a["min_iou_t0"] = np.minimum(iou_sideline, iou_endzone)
        df_a["iou_diff_t0"] = np.abs(iou_sideline - iou_endzone)

        # 3. Add Lags (Cite solution_lesson_node_00086)
        # We add lags for Distance (interaction trajectory) and Max IoU (visual trajectory)
        lag_cols = ["distance", "max_iou_t0"]
        df_a = self._add_lags(
            df_a,
            lag_cols,
            self.config.LAG_STEPS,
            ["game_play", "nfl_player_id_1", "nfl_player_id_2"],
        )

        # Select Features
        features = self.config.FEATURES_STREAM_A
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
        Ego-Centric Invariance + Temporal Context
        """
        df_b = df[df["nfl_player_id_2"] == "G"].copy()
        if len(df_b) == 0:
            return pd.DataFrame(), np.array([]), np.array([])

        # 1. Ego-Centric Projection (Cite solution_lesson_node_00075)
        rad_dir = np.radians(df_b["direction_p1"].fillna(0))
        rad_orient = np.radians(df_b["orientation_p1"].fillna(0))
        theta = rad_dir - rad_orient
        speed = df_b["speed_p1"].fillna(0)

        df_b["v_surge"] = speed * np.cos(theta)
        df_b["v_sway"] = speed * np.sin(theta)
        df_b["speed"] = speed
        df_b["acceleration"] = df_b["acceleration_p1"].fillna(0)

        # 2. Add Lags (Cite solution_lesson_node_00086)
        # Instead of instantaneous jerk/energy, we provide the window of velocity/accel
        df_b = df_b.sort_values(["game_play", "nfl_player_id_1", "step"])
        lag_cols = ["v_surge", "v_sway", "speed", "acceleration"]
        df_b = self._add_lags(
            df_b, lag_cols, self.config.LAG_STEPS, ["game_play", "nfl_player_id_1"]
        )

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
