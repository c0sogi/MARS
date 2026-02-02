import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class NFLContactDataset(Dataset):
    """
    PyTorch Dataset for NFL Contact Detection.
    Serves flat feature vectors and binary labels.
    """

    def __init__(self, X, y=None, contact_ids=None):
        """
        Args:
            X (np.ndarray): Input features of shape (N, Input_Dim).
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
    Cite solution_lesson_node_00015: Uses vectorized lag-shifting for wide feature construction.
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
            df = df.head(Config.DEBUG_SAMPLE_SIZE)
        return df

    def _load_tracking(self, split, game_plays):
        """Loads tracking data, filtering for relevant game_plays."""
        if split == "test":
            path = Config.TEST_TRACKING_PATH
        else:
            path = Config.TRAIN_TRACKING_PATH

        df_trk = pd.read_csv(path, usecols=Config.TRACKING_COLS)
        df_trk = df_trk[df_trk["game_play"].isin(game_plays)].copy()
        return df_trk

    def _create_wide_tracking(self, df_trk):
        """
        Creates a wide dataframe with lagged features for each player.
        """
        # Sort for correct shifting
        df_trk.sort_values(by=["game_play", "nfl_player_id", "step"], inplace=True)

        # Base features to shift
        features = Config.PLAYER_FEATURES

        # Generate lags
        # Window: -5 to +5 (centered at 0)
        half_window = Config.WINDOW_SIZE // 2
        lags = range(-half_window, half_window + 1)

        # We want to create columns like x_lag_-5, x_lag_-4 ...
        # shift(k) shifts data DOWN by k.
        # To get t-5 at row t, we need to look 5 rows UP (if sorted asc).
        # Actually:
        # t-5 means 5 steps ago. In a sorted list [0, 1, 2...], step 0 is at index i.
        # We want data from step -5 (which doesn't exist) or step 5 (future).
        # df.shift(1) gives previous row.
        # To get data from t-k at row t, we use shift(k).
        # Example: lag=-1 (past). We want data from t-1. That is shift(1).
        # Example: lag=1 (future). We want data from t+1. That is shift(-1).

        # We perform groupby shift to respect player boundaries
        grouped = df_trk.groupby(["game_play", "nfl_player_id"])

        lagged_dfs = []
        for lag in lags:
            # lag is the time offset relative to current.
            # To get data at (t + lag), we need to shift by -lag.
            shift_amount = -lag

            # Select features and shift
            df_shifted = grouped[features].shift(shift_amount)

            # Rename columns
            df_shifted.columns = [f"{col}_lag_{lag}" for col in features]
            lagged_dfs.append(df_shifted)

        # Concatenate all lagged features horizontally
        df_wide = pd.concat(
            [df_trk[["game_play", "nfl_player_id", "step"]]] + lagged_dfs, axis=1
        )

        return df_wide

    def process_data(self, split="train", load_cached_data=True):
        """
        Main processing pipeline using Wide Format.
        """
        X_path = os.path.join(self.cache_dir, f"{split}_X.npy")
        y_path = os.path.join(self.cache_dir, f"{split}_y.npy")
        ids_path = os.path.join(self.cache_dir, f"{split}_ids.npy")

        if load_cached_data and os.path.exists(X_path):
            print(f"Loading {split} data from cache...")
            X = np.load(X_path)
            y = np.load(y_path)
            ids = np.load(ids_path, allow_pickle=True)
            return X, y, ids

        print(f"Processing {split} data from scratch (Wide Format)...")

        # 1. Load Metadata
        df_meta = self._load_metadata(split)

        # 2. Load and Widen Tracking
        unique_gps = df_meta["game_play"].unique()
        df_trk = self._load_tracking(split, unique_gps)
        df_wide = self._create_wide_tracking(df_trk)

        # 3. Merge P1
        df_meta["nfl_player_id_1"] = pd.to_numeric(
            df_meta["nfl_player_id_1"], errors="coerce"
        )
        df_merged = df_meta.merge(
            df_wide,
            left_on=["game_play", "step", "nfl_player_id_1"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
        )
        # Rename P1 columns (they have _lag_k suffixes)
        # We need to distinguish P1 vs P2 lags.
        # Current cols: x_position_lag_-5, etc.
        # We rename to x_position_1_lag_-5

        # Identify lag columns
        lag_cols = [c for c in df_wide.columns if "_lag_" in c]
        rename_p1 = {c: c.replace("_lag_", "_1_lag_") for c in lag_cols}
        df_merged.rename(columns=rename_p1, inplace=True)
        df_merged.drop(columns=["nfl_player_id"], inplace=True)

        # 4. Merge P2
        df_merged["nfl_player_id_2_num"] = pd.to_numeric(
            df_merged["nfl_player_id_2"], errors="coerce"
        )
        df_merged = df_merged.merge(
            df_wide,
            left_on=["game_play", "step", "nfl_player_id_2_num"],
            right_on=["game_play", "step", "nfl_player_id"],
            how="left",
            suffixes=("", "_drop"),
        )

        rename_p2 = {c: c.replace("_lag_", "_2_lag_") for c in lag_cols}
        df_merged.rename(columns=rename_p2, inplace=True)
        df_merged.drop(columns=["nfl_player_id"], inplace=True)

        # 5. Fill NaNs (Ground or Missing)
        p1_cols = list(rename_p1.values())
        p2_cols = list(rename_p2.values())
        df_merged[p1_cols] = df_merged[p1_cols].fillna(0.0)
        df_merged[p2_cols] = df_merged[p2_cols].fillna(0.0)

        # 6. Feature Engineering (Vectorized over lags)
        # We compute distance and log_distance for each lag
        half_window = Config.WINDOW_SIZE // 2
        lags = range(-half_window, half_window + 1)

        dist_cols = []
        log_dist_cols = []

        is_ground = (df_merged["nfl_player_id_2"] == "G").astype(float)

        for lag in lags:
            x1 = df_merged[f"x_position_1_lag_{lag}"]
            y1 = df_merged[f"y_position_1_lag_{lag}"]
            x2 = df_merged[f"x_position_2_lag_{lag}"]
            y2 = df_merged[f"y_position_2_lag_{lag}"]

            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx**2 + dy**2)

            # Ground logic: distance is 0
            dist[is_ground == 1.0] = 0.0

            # Log Distance (Cite solution_lesson_node_00005)
            log_dist = np.log1p(dist)

            d_col = f"distance_lag_{lag}"
            ld_col = f"log_distance_lag_{lag}"

            df_merged[d_col] = dist
            df_merged[ld_col] = log_dist

            dist_cols.append(d_col)
            log_dist_cols.append(ld_col)

        # 7. Final Feature Selection
        # P1 feats, P2 feats, Dist feats, LogDist feats, is_ground
        feature_cols = p1_cols + p2_cols + dist_cols + log_dist_cols + ["is_ground"]

        # Add is_ground column
        df_merged["is_ground"] = is_ground

        X_raw = df_merged[feature_cols].values.astype(np.float32)

        # 8. Normalization
        # Normalize all except is_ground
        norm_indices = list(range(len(feature_cols) - 1))

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

        # 9. Save
        y_final = df_meta["contact"].values.astype(np.float32)
        ids_final = df_meta["contact_id"].values

        np.save(X_path, X_raw)
        np.save(y_path, y_final)
        np.save(ids_path, ids_final)

        print(f"Processed data shape: {X_raw.shape}")
        return X_raw, y_final, ids_final
