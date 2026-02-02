import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


class PIRVDataset(Dataset):
    """
    PyTorch Dataset for the Pyramidal Invariant Residual-Visual Network.
    Serves dual-stream inputs: Kinematic (wide window) and Visual (shallow).
    """

    def __init__(self, features_kin, features_vis, labels=None):
        self.features_kin = torch.FloatTensor(features_kin)
        self.features_vis = torch.FloatTensor(features_vis)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features_kin)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features_kin[idx], self.features_vis[idx], self.labels[idx]
        else:
            return self.features_kin[idx], self.features_vis[idx]


def preprocess_tracking(df_tracking, load_cached_data=True):
    """
    Applies Entity-First feature generation: creating windowed lags for tracking data.
    Generates a 'wide' dataframe where each row contains features for t-5 to t+5.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "processed_tracking_wide.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Ensure sorting for correct shifting
    df = df_tracking.sort_values(["game_play", "nfl_player_id", "step"]).copy()

    # Base columns to shift (exclude keys)
    feature_cols = [
        c
        for c in Config.TRACKING_RAW_COLS
        if c not in ["game_play", "step", "nfl_player_id"]
    ]

    # Create windowed features using groupby and shift
    # We want window [t-5, ..., t+5].
    # shift(k) shifts data DOWN by k rows.
    # To get data from t+k (future) at row t, we need shift(-k).
    # To get data from t-k (past) at row t, we need shift(k).
    # We iterate lag from -5 to +5.
    # lag -5 means t-5. shift(5).

    shifted_dfs = []
    grouper = df.groupby(["game_play", "nfl_player_id"])[feature_cols]

    for lag in range(-Config.HALF_WINDOW, Config.HALF_WINDOW + 1):
        # lag is the time offset relative to t.
        # lag = -5 => want data from 5 steps ago. shift(5).
        # lag = 5 => want data from 5 steps future. shift(-5).
        shift_amount = -lag

        s_df = grouper.shift(shift_amount)
        s_df.columns = [f"{c}_lag_{lag}" for c in feature_cols]
        shifted_dfs.append(s_df)

    # Concatenate all lags. Index alignment is preserved.
    df_wide = pd.concat(
        [df[["game_play", "nfl_player_id", "step"]]] + shifted_dfs, axis=1
    )

    # Fill NaNs created at the edges of plays with 0
    df_wide = df_wide.fillna(0.0)

    # Save to cache
    df_wide.to_parquet(cache_path, index=False)

    return df_wide


def preprocess_helmets(df_helmets, load_cached_data=True):
    """
    Applies Max-Pooling Selection Strategy to helmet data.
    Selects the single largest bounding box per player per frame.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "processed_helmets_maxpool.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = df_helmets.copy()
    # Calculate area for max-pooling
    df["area"] = df["width"] * df["height"]

    # Sort by area descending
    df = df.sort_values("area", ascending=False)

    # Drop duplicates to keep only the largest box per player per frame
    # Keys: game_play, frame, nfl_player_id
    df_unique = df.drop_duplicates(
        subset=["game_play", "frame", "nfl_player_id"], keep="first"
    )

    # Select relevant columns
    cols = Config.HELMET_RAW_COLS + ["area"]
    # Ensure columns exist (handle potential missing cols in input)
    cols = [c for c in cols if c in df_unique.columns]
    df_unique = df_unique[cols]

    # Save to cache
    df_unique.to_parquet(cache_path, index=False)

    return df_unique


def generate_contact_features(
    metadata_path, tracking_path, helmets_path, mode="train", load_cached_data=True
):
    """
    Generates the final feature sets for PIRV-Net.
    Handles merging, Hybrid Ground Imputation, and Normalization.

    Args:
        mode: 'train', 'val', or 'test'.
    """
    # Define cache paths
    cache_prefix = os.path.join(Config.WORKING_DIR, f"features_{mode}")
    path_kin = f"{cache_prefix}_kin.npy"
    path_vis = f"{cache_prefix}_vis.npy"
    path_lbl = f"{cache_prefix}_lbl.npy"
    scaler_kin_path = os.path.join(Config.WORKING_DIR, "scaler_kin.joblib")
    scaler_vis_path = os.path.join(Config.WORKING_DIR, "scaler_vis.joblib")

    # Check cache
    if load_cached_data and os.path.exists(path_kin) and os.path.exists(path_vis):
        # For train/val, we expect labels. For test, we might not have them.
        if mode != "test" and not os.path.exists(path_lbl):
            pass  # Force recompute if labels missing
        else:
            X_kin = np.load(path_kin)
            X_vis = np.load(path_vis)
            y = np.load(path_lbl) if mode != "test" else None
            return X_kin, X_vis, y

    # 1. Load Data
    df_meta = pd.read_csv(metadata_path)
    # Ensure step is numeric (int) to prevent object-dtype issues in empty dataframes
    # Cite debug_lesson_17
    if "step" in df_meta.columns:
        df_meta["step"] = df_meta["step"].astype(int)

    df_tracking_raw = pd.read_csv(tracking_path)
    df_helmets_raw = pd.read_csv(helmets_path)

    # 2. Preprocess Inputs (with internal caching)
    df_track_wide = preprocess_tracking(
        df_tracking_raw, load_cached_data=load_cached_data
    )
    df_helmets_proc = preprocess_helmets(
        df_helmets_raw, load_cached_data=load_cached_data
    )

    # 3. Merge Tracking (Kinematics)
    # Ensure ID types match for merging
    df_meta["nfl_player_id_1"] = pd.to_numeric(
        df_meta["nfl_player_id_1"], errors="coerce"
    )

    # Merge Player 1
    df_merged = df_meta.merge(
        df_track_wide,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )

    # Rename P1 columns to {feat}_1_lag_{k}
    base_feats = [
        c
        for c in Config.TRACKING_RAW_COLS
        if c not in ["game_play", "step", "nfl_player_id"]
    ]
    rename_map_1 = {}
    for lag in range(-Config.HALF_WINDOW, Config.HALF_WINDOW + 1):
        for feat in base_feats:
            rename_map_1[f"{feat}_lag_{lag}"] = f"{feat}_1_lag_{lag}"
    df_merged = df_merged.rename(columns=rename_map_1)

    # Prepare P2 ID for merge (Handle 'G')
    df_merged["nfl_player_id_2_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    # Merge Player 2
    df_merged = df_merged.merge(
        df_track_wide,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_p2_temp"),
    )

    # Rename P2 columns to {feat}_2_lag_{k}
    rename_map_2 = {}
    for lag in range(-Config.HALF_WINDOW, Config.HALF_WINDOW + 1):
        for feat in base_feats:
            rename_map_2[f"{feat}_lag_{lag}"] = f"{feat}_2_lag_{lag}"
    df_merged = df_merged.rename(columns=rename_map_2)

    # 4. Hybrid Ground Imputation & Feature Calculation
    is_ground = df_merged["nfl_player_id_2"] == "G"
    kinematic_feature_cols = []

    for lag in range(-Config.HALF_WINDOW, Config.HALF_WINDOW + 1):
        suffix = f"_lag_{lag}"

        # P1 Columns
        x1 = df_merged[f"x_position_1{suffix}"]
        y1 = df_merged[f"y_position_1{suffix}"]

        # P2 Columns
        x2_col = f"x_position_2{suffix}"
        y2_col = f"y_position_2{suffix}"
        s2_col = f"speed_2{suffix}"
        a2_col = f"acceleration_2{suffix}"
        dir2_col = f"direction_2{suffix}"
        o2_col = f"orientation_2{suffix}"
        sa2_col = f"sa_2{suffix}"

        # Impute Ground: P2 Pos = P1 Pos, Vel/Accel = 0
        df_merged.loc[is_ground, x2_col] = x1[is_ground]
        df_merged.loc[is_ground, y2_col] = y1[is_ground]
        for col in [s2_col, a2_col, dir2_col, o2_col, sa2_col]:
            df_merged.loc[is_ground, col] = 0.0

        # Fill remaining NaNs (missing tracking for non-ground) with 0
        cols_to_fill = [x2_col, y2_col, s2_col, a2_col, dir2_col, o2_col, sa2_col]
        df_merged[cols_to_fill] = df_merged[cols_to_fill].fillna(0.0)

        # Compute Relative Features
        dx = df_merged[x1.name] - df_merged[x2_col]
        dy = df_merged[y1.name] - df_merged[y2_col]
        dist = np.sqrt(dx**2 + dy**2)

        df_merged[f"distance{suffix}"] = dist
        df_merged[f"x_rel{suffix}"] = dx
        df_merged[f"y_rel{suffix}"] = dy

        # Construct feature list for this lag based on Config.KINEMATIC_FEATURES
        # Config names are base names (e.g., 'x_position_1'). We map to lagged names.
        lag_cols = [f"{base}{suffix}" for base in Config.KINEMATIC_FEATURES]
        kinematic_feature_cols.extend(lag_cols)

    # 5. Visual Features
    # Convert step to approx frame (Snap is 300 frames in, step is 10Hz, video 59.94Hz)
    df_merged["frame_approx"] = (300 + df_merged["step"] * 5.994).round().astype(int)

    # Merge P1 Visuals
    df_merged = df_merged.merge(
        df_helmets_proc,
        left_on=["game_play", "frame_approx", "nfl_player_id_1"],
        right_on=["game_play", "frame", "nfl_player_id"],
        how="left",
        suffixes=("", "_vis1"),
    )
    vis_rename_1 = {
        "left": "left_1",
        "width": "width_1",
        "top": "top_1",
        "height": "height_1",
        "area": "view_area_1",
    }
    df_merged = df_merged.rename(columns=vis_rename_1)

    # Merge P2 Visuals
    df_merged = df_merged.merge(
        df_helmets_proc,
        left_on=["game_play", "frame_approx", "nfl_player_id_2_num"],
        right_on=["game_play", "frame", "nfl_player_id"],
        how="left",
        suffixes=("", "_vis2"),
    )
    vis_rename_2 = {
        "left": "left_2",
        "width": "width_2",
        "top": "top_2",
        "height": "height_2",
        "area": "view_area_2",
    }
    df_merged = df_merged.rename(columns=vis_rename_2)

    # Fill Visual NaNs (Ground or missing) with 0
    vis_cols = Config.VISUAL_FEATURES
    for col in vis_cols:
        if col not in df_merged.columns:
            df_merged[col] = 0.0
        else:
            df_merged[col] = df_merged[col].fillna(0.0)

    # 6. Normalization
    # Select feature matrices
    X_kin = df_merged[kinematic_feature_cols].values.astype(np.float32)
    X_vis = df_merged[vis_cols].values.astype(np.float32)

    if mode == "train":
        # Fit and save scalers
        scaler_kin = StandardScaler()
        X_kin = scaler_kin.fit_transform(X_kin)
        joblib.dump(scaler_kin, scaler_kin_path)

        scaler_vis = StandardScaler()
        X_vis = scaler_vis.fit_transform(X_vis)
        joblib.dump(scaler_vis, scaler_vis_path)
    else:
        # Load and apply scalers
        if os.path.exists(scaler_kin_path) and os.path.exists(scaler_vis_path):
            scaler_kin = joblib.load(scaler_kin_path)
            X_kin = scaler_kin.transform(X_kin)

            scaler_vis = joblib.load(scaler_vis_path)
            X_vis = scaler_vis.transform(X_vis)
        else:
            # Fallback if no scaler exists (e.g. testing without training first)
            pass

    # 7. Extract Labels
    y = None
    if "contact" in df_merged.columns:
        y = df_merged["contact"].values.astype(np.float32)

    # Cache final arrays
    np.save(path_kin, X_kin)
    np.save(path_vis, X_vis)
    if y is not None:
        np.save(path_lbl, y)

    return X_kin, X_vis, y
