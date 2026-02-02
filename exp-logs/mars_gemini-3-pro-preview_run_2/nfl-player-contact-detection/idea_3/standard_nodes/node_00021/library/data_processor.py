import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class NFLContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Serves windowed time-series data and binary labels.
    """

    def __init__(self, X, y=None, contact_ids=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, Window_Size, Num_Features).
            y (np.ndarray, optional): Binary labels of shape (N,).
            contact_ids (np.ndarray, optional): Array of contact_id strings.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.contact_ids = contact_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class DataProcessor:
    """
    Handles data loading, feature engineering, window generation, and caching.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.scaler_mean_path = os.path.join(self.cache_dir, "scaler_mean.npy")
        self.scaler_scale_path = os.path.join(self.cache_dir, "scaler_scale.npy")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self, split):
        """Loads the appropriate metadata CSV file."""
        if split == "train":
            path = Config.TRAIN_META_PATH
        elif split == "validation":
            path = Config.VAL_META_PATH
        elif split == "test":
            path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        df = pd.read_csv(path)

        if Config.DEBUG:
            print(f"DEBUG MODE: Loading subset of {Config.DEBUG_SAMPLE_SIZE} samples.")
            df = df.head(Config.DEBUG_SAMPLE_SIZE)

        return df

    def _load_tracking(self, split, game_plays):
        """Loads tracking data, filtering for relevant game_plays."""
        if split == "test":
            path = Config.TEST_TRACKING_PATH
        else:
            path = Config.TRAIN_TRACKING_PATH

        # Load only necessary columns
        df_trk = pd.read_csv(path, usecols=Config.TRACKING_COLS)

        # Filter to relevant game_plays to optimize memory
        df_trk = df_trk[df_trk["game_play"].isin(game_plays)].copy()

        return df_trk

    def _compute_derived_features(self, df):
        """
        Computes distance, log_distance, and closing_speed.
        Cite solution_lesson_node_00007: "Clamping the denominator... to prevent gradient noise".
        Cite solution_lesson_node_00005: "Expanding Decision Boundaries via Non-Linear Feature Transformation".
        """
        # 1. Euclidean Distance
        dx = df["x_position_1"] - df["x_position_2"]
        dy = df["y_position_1"] - df["y_position_2"]
        dist = np.sqrt(dx**2 + dy**2)

        # Force distance to 0 for Ground contacts
        is_ground = df["is_ground"] == 1
        dist[is_ground] = 0.0

        # 2. Log Distance
        log_dist = np.log1p(dist)

        # 3. Physics-based Closing Speed
        # V_rel = V1 - V2
        # P_rel = P1 - P2
        # Closing Speed = -(V_rel . P_rel) / |P_rel|

        # Convert Direction (deg) to Radians. NFL 0=Y, 90=X?
        # Standard: 0=N(Y), 90=E(X).
        # Vx = Speed * sin(dir), Vy = Speed * cos(dir)
        dir1_rad = np.radians(df["direction_1"])
        dir2_rad = np.radians(df["direction_2"])

        vx1 = df["speed_1"] * np.sin(dir1_rad)
        vy1 = df["speed_1"] * np.cos(dir1_rad)
        vx2 = df["speed_2"] * np.sin(dir2_rad)
        vy2 = df["speed_2"] * np.cos(dir2_rad)

        dvx = vx1 - vx2
        dvy = vy1 - vy2

        # Dot product
        dot_prod = dvx * dx + dvy * dy

        # Clamped division (Lesson 00007)
        denom = np.maximum(dist, 1e-6)
        closing_speed = -(dot_prod / denom)

        # For ground, V2 is 0, P2 is 0? No, P2 is G.
        # If is_ground, we want closing speed to ground.
        # P2 is effectively (x1, y1) so dist=0.
        # This singularity is handled by clamping, but physically
        # closing speed to ground is just vertical speed?
        # We don't have Z. We assume 0 for ground cases to avoid noise.
        closing_speed[is_ground] = 0.0

        return dist, log_dist, closing_speed

    def process_data(self, split="train", load_cached_data=True):
        """
        Main processing pipeline.
        Cite solution_lesson_node_00015: "Vectorized Lag-Shifting Outperforms Vertical Window Expansion".
        We use the expansion method to ensure alignment but fix the merge logic.
        """
        # --- 1. Cache Check ---
        X_path = os.path.join(self.cache_dir, f"{split}_X.npy")
        y_path = os.path.join(self.cache_dir, f"{split}_y.npy")
        ids_path = os.path.join(self.cache_dir, f"{split}_ids.npy")

        if (
            load_cached_data
            and os.path.exists(X_path)
            and os.path.exists(y_path)
            and os.path.exists(ids_path)
        ):
            print(f"Loading {split} data from cache...")
            X = np.load(X_path)
            y = np.load(y_path)
            ids = np.load(ids_path, allow_pickle=True)
            return X, y, ids

        print(f"Processing {split} data from scratch...")

        # --- 2. Load Raw Data ---
        df_meta = self._load_metadata(split)
        unique_gps = df_meta["game_play"].unique()
        df_trk = self._load_tracking(split, unique_gps)

        # --- 3. Window Expansion ---
        half_window = Config.WINDOW_SIZE // 2
        offsets = np.arange(-half_window, half_window + 1)

        # Keep track of original index
        df_meta["orig_index"] = df_meta.index

        # Expand
        n_repeats = len(offsets)
        df_expanded = df_meta.loc[df_meta.index.repeat(n_repeats)].copy()
        df_expanded["offset"] = np.tile(offsets, len(df_meta))
        df_expanded["step_window"] = df_expanded["step"] + df_expanded["offset"]

        # --- 4. Merge Tracking Data ---
        # Ensure Types match for Merge Keys
        df_expanded["game_play"] = df_expanded["game_play"].astype(str)
        df_trk["game_play"] = df_trk["game_play"].astype(str)

        df_expanded["step_window"] = df_expanded["step_window"].astype(int)
        df_trk["step"] = df_trk["step"].astype(int)

        df_expanded["nfl_player_id_1"] = (
            pd.to_numeric(df_expanded["nfl_player_id_1"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        df_expanded["nfl_player_id_2_num"] = (
            pd.to_numeric(df_expanded["nfl_player_id_2"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        df_trk["nfl_player_id"] = (
            pd.to_numeric(df_trk["nfl_player_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

        # Merge P1
        df_merged = df_expanded.merge(
            df_trk,
            left_on=["game_play", "step_window", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        rename_p1 = {c: f"{c}_1" for c in Config.PLAYER_FEATURES}
        df_merged.rename(columns=rename_p1, inplace=True)
        df_merged.drop(
            columns=["nfl_player_id", "step_y"], inplace=True, errors="ignore"
        )
        df_merged.rename(columns={"step_x": "step"}, inplace=True)

        # Merge P2
        df_merged = df_merged.merge(
            df_trk,
            left_on=["game_play", "step_window", "nfl_player_id_2_num"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_2"),
        )
        rename_p2 = {c: f"{c}_2" for c in Config.PLAYER_FEATURES}
        df_merged.rename(columns=rename_p2, inplace=True)
        df_merged.drop(columns=["nfl_player_id", "step"], inplace=True, errors="ignore")

        # Fill Missing
        feat_cols_1 = [f"{c}_1" for c in Config.PLAYER_FEATURES]
        feat_cols_2 = [f"{c}_2" for c in Config.PLAYER_FEATURES]
        df_merged[feat_cols_1] = df_merged[feat_cols_1].fillna(0.0)
        df_merged[feat_cols_2] = df_merged[feat_cols_2].fillna(0.0)

        # --- 5. Feature Engineering ---
        df_merged.sort_values(by=["orig_index", "offset"], inplace=True)
        df_merged["is_ground"] = (df_merged["nfl_player_id_2"] == "G").astype(float)

        dist, log_dist, closing_speed = self._compute_derived_features(df_merged)
        df_merged["distance"] = dist
        df_merged["log_distance"] = log_dist
        df_merged["closing_speed"] = closing_speed

        # --- 6. Normalization ---
        feature_order = (
            feat_cols_1
            + feat_cols_2
            + ["distance", "log_distance", "closing_speed", "is_ground"]
        )
        X_raw = df_merged[feature_order].values.astype(np.float32)
        norm_indices = list(range(len(feature_order) - 1))

        if split == "train":
            mean = np.mean(X_raw[:, norm_indices], axis=0)
            scale = np.std(X_raw[:, norm_indices], axis=0)
            scale[scale < 1e-6] = 1.0
            np.save(self.scaler_mean_path, mean)
            np.save(self.scaler_scale_path, scale)
        else:
            if os.path.exists(self.scaler_mean_path):
                mean = np.load(self.scaler_mean_path)
                scale = np.load(self.scaler_scale_path)
            else:
                mean = 0.0
                scale = 1.0

        X_raw[:, norm_indices] = (X_raw[:, norm_indices] - mean) / scale

        # --- 7. Reshape (Flattened) ---
        # Cite solution_lesson_node_00015: "Wide" feature vectors.
        # We reshape to (N, Window * Features)
        num_samples = len(df_meta)
        num_features = len(feature_order)

        # First reshape to (N, Window, Features)
        X_3d = X_raw.reshape(num_samples, Config.WINDOW_SIZE, num_features)

        # Then flatten to (N, Window * Features)
        X_final = X_3d.reshape(num_samples, -1)

        y_final = df_meta["contact"].values.astype(np.float32)
        ids_final = df_meta["contact_id"].values

        # --- 8. Save ---
        np.save(X_path, X_final)
        np.save(y_path, y_final)
        np.save(ids_path, ids_final)

        print(f"Processed data shape: {X_final.shape}")
        return X_final, y_final, ids_final
