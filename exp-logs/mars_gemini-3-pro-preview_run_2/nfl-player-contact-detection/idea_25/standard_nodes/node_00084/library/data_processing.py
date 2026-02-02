import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib

from library.config import Config
from library.utils import seed_everything

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)


class ContactDataset(Dataset):
    def __init__(self, X_kin, X_vis, X_gate, pos, team, y=None):
        self.X_kin = torch.FloatTensor(X_kin)
        self.X_vis = torch.FloatTensor(X_vis)
        self.X_gate = torch.FloatTensor(X_gate)
        self.pos = torch.LongTensor(pos)
        self.team = torch.LongTensor(team)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_kin)

    def __getitem__(self, idx):
        out = {
            "x_kin": self.X_kin[idx],
            "x_vis": self.X_vis[idx],
            "x_gate": self.X_gate[idx],
            "x_pos": self.pos[idx],
            "x_team": self.team[idx],
        }
        if self.y is not None:
            out["y"] = self.y[idx]
        return out


def get_shortest_arc(a, b):
    """Computes the shortest arc between two angles in degrees."""
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)


def process_tracking(df_tracking, load_cached=True):
    cache_path = os.path.join(Config.WORKING_DIR, "tracking_processed.parquet")
    if load_cached and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    print("Processing tracking data...")
    # Sort for windowing
    df = df_tracking.sort_values(["game_play", "nfl_player_id", "step"]).copy()

    # Keep static features separate
    static_cols = ["game_play", "nfl_player_id", "step", "position", "team"]
    feat_cols = Config.TRACKING_FEATS

    # Group for shifting
    grouped = df.groupby(["game_play", "nfl_player_id"])[feat_cols]

    lagged_dfs = []
    # Lags: negative is future, positive is past. Range -5 to 5.
    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        # shift(lag): positive lag shifts data down (t gets t-lag), negative shifts up (t gets t+lag)
        # We want lag k to represent t-k. So shift(k) is correct.
        shifted = grouped.shift(lag)
        shifted.columns = [f"{c}_lag_{lag}" for c in feat_cols]
        lagged_dfs.append(shifted)

    df_wide = pd.concat([df[static_cols]] + lagged_dfs, axis=1)

    # Fill NaNs at edges of play with 0 (padding)
    df_wide = df_wide.fillna(0)

    # Optimize types
    for c in df_wide.columns:
        if df_wide[c].dtype == "float64":
            df_wide[c] = df_wide[c].astype("float32")

    df_wide.to_parquet(cache_path)
    return df_wide


def process_visuals(df_helmets, load_cached=True):
    cache_path = os.path.join(Config.WORKING_DIR, "visuals_processed.parquet")
    if load_cached and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    print("Processing visual data...")
    df = df_helmets.copy()

    # Map frame to step (approximate 59.94Hz to 10Hz)
    # Snap is frame 300, step 0.
    df["step"] = ((df["frame"] - 300) / 6).round().astype(int)

    # Calculate box area
    df["area"] = df["width"] * df["height"]

    # Max-Pooling: Select best view per player per step based on area
    # Sort by area descending
    df = df.sort_values("area", ascending=False)
    # Drop duplicates to keep largest area
    df = df.drop_duplicates(subset=["game_play", "nfl_player_id", "step"])

    # Sort for windowing
    df = df.sort_values(["game_play", "nfl_player_id", "step"])

    # Windowing
    feat_cols = Config.VISUAL_FEATS + ["area"]
    grouped = df.groupby(["game_play", "nfl_player_id"])[feat_cols]

    lagged_dfs = []
    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        shifted = grouped.shift(lag)
        shifted.columns = [f"{c}_lag_{lag}" for c in feat_cols]
        lagged_dfs.append(shifted)

    df_wide = pd.concat(
        [df[["game_play", "nfl_player_id", "step"]]] + lagged_dfs, axis=1
    )
    df_wide = df_wide.fillna(0)

    for c in df_wide.columns:
        if df_wide[c].dtype == "float64":
            df_wide[c] = df_wide[c].astype("float32")

    df_wide.to_parquet(cache_path)
    return df_wide


def merge_and_engineer(df_labels, df_track, df_vis, is_train=True):
    print("Merging and engineering features...")

    # Ensure IDs are strings for merging
    df_labels["nfl_player_id_1"] = df_labels["nfl_player_id_1"].astype(str)
    df_labels["nfl_player_id_2"] = df_labels["nfl_player_id_2"].astype(str)
    df_track["nfl_player_id"] = df_track["nfl_player_id"].astype(str)
    df_vis["nfl_player_id"] = df_vis["nfl_player_id"].astype(str)

    # --- Merge Player 1 ---
    # Tracking
    df_merged = df_labels.merge(
        df_track.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )
    # Visuals
    df_merged = df_merged.merge(
        df_vis.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
    )

    # --- Merge Player 2 ---
    # Handle Ground separately later, first merge standard players
    # We merge blindly, then fix Ground rows
    df_merged = df_merged.merge(
        df_track.add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
    )
    df_merged = df_merged.merge(
        df_vis.add_suffix("_2"),
        left_on=["game_play", "step", "nfl_player_id_2"],
        right_on=["game_play_2", "step_2", "nfl_player_id_2"],
        how="left",
    )

    # --- Ground Imputation ---
    is_ground = df_merged["nfl_player_id_2"] == "G"

    # For Ground, P2 pos = P1 pos (dist=0), P2 vel = 0
    # We need to iterate over lags to do this correctly for the windowed features
    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # Tracking cols
        p1_x = f"x_position{suffix}_1"
        p1_y = f"y_position{suffix}_1"
        p2_x = f"x_position{suffix}_2"
        p2_y = f"y_position{suffix}_2"

        # Impute Position
        df_merged.loc[is_ground, p2_x] = df_merged.loc[is_ground, p1_x]
        df_merged.loc[is_ground, p2_y] = df_merged.loc[is_ground, p1_y]

        # Impute Velocity/Accel/Angles (Zero)
        for feat in ["speed", "acceleration", "orientation", "direction", "sa"]:
            col = f"{feat}{suffix}_2"
            df_merged.loc[is_ground, col] = 0.0

        # Impute Visuals (Zero)
        for feat in Config.VISUAL_FEATS + ["area"]:
            col = f"{feat}{suffix}_2"
            df_merged.loc[is_ground, col] = 0.0

    # Fill remaining NaNs (missing tracking for real players) with 0
    df_merged = df_merged.fillna(0)

    # --- Feature Engineering ---
    # Derived features for each lag
    derived_cols = []

    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"

        # 1. Distance
        dx = df_merged[f"x_position{suffix}_1"] - df_merged[f"x_position{suffix}_2"]
        dy = df_merged[f"y_position{suffix}_1"] - df_merged[f"y_position{suffix}_2"]
        dist_col = f"distance{suffix}"
        df_merged[dist_col] = np.sqrt(dx**2 + dy**2)
        derived_cols.append(dist_col)

        # 2. Relative Speed
        s1 = df_merged[f"speed{suffix}_1"]
        s2 = df_merged[f"speed{suffix}_2"]
        rs_col = f"rel_speed{suffix}"
        df_merged[rs_col] = np.abs(s1 - s2)
        derived_cols.append(rs_col)

        # 3. Relative Accel
        a1 = df_merged[f"acceleration{suffix}_1"]
        a2 = df_merged[f"acceleration{suffix}_2"]
        ra_col = f"rel_accel{suffix}"
        df_merged[ra_col] = np.abs(a1 - a2)
        derived_cols.append(ra_col)

        # 4. Relative Orientation
        o1 = df_merged[f"orientation{suffix}_1"]
        o2 = df_merged[f"orientation{suffix}_2"]
        ro_col = f"rel_angle_o{suffix}"
        df_merged[ro_col] = get_shortest_arc(o1, o2)
        derived_cols.append(ro_col)

        # 5. Relative Direction
        d1 = df_merged[f"direction{suffix}_1"]
        d2 = df_merged[f"direction{suffix}_2"]
        rd_col = f"rel_angle_d{suffix}"
        df_merged[rd_col] = get_shortest_arc(d1, d2)
        derived_cols.append(rd_col)

    # --- Clamping ---
    # Clamp derived features and raw kinematic features
    kin_cols_to_clamp = []
    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"
        # Add P1/P2 raw tracking cols
        for feat in Config.TRACKING_FEATS:
            kin_cols_to_clamp.append(f"{feat}{suffix}_1")
            kin_cols_to_clamp.append(f"{feat}{suffix}_2")

    all_clamp_cols = kin_cols_to_clamp + derived_cols
    df_merged[all_clamp_cols] = df_merged[all_clamp_cols].clip(
        Config.CLAMP_MIN, Config.CLAMP_MAX
    )

    # --- Collect Feature Names ---
    # Kinematic: P1 raw + P2 raw + Derived
    kin_feats = kin_cols_to_clamp + derived_cols

    # Visual: P1 raw + P2 raw (area excluded from main vis input? Config says VISUAL_FEATS=['left'..])
    vis_feats = []
    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        suffix = f"_lag_{lag}"
        for feat in Config.VISUAL_FEATS:
            vis_feats.append(f"{feat}{suffix}_1")
            vis_feats.append(f"{feat}{suffix}_2")

    # Gate: P1 area + P2 area (current frame, lag 0)
    gate_feats = [f"area_lag_0_1", f"area_lag_0_2"]

    return df_merged, kin_feats, vis_feats, gate_feats


def get_data(load_cached=True):
    # 1. Load Metadata
    print("Loading metadata...")
    df_train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    df_val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))
    df_test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # 2. Load Raw Data
    print("Loading raw tracking and helmets...")
    # We load full training tracking/helmets for both train and val
    # Note: In a real scenario with huge data, we might filter. Here we load all.
    df_tracking = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
    )
    df_helmets = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "train_baseline_helmets.csv")
    )

    # 3. Process Raw Data
    df_track_proc = process_tracking(df_tracking, load_cached=load_cached)
    df_vis_proc = process_visuals(df_helmets, load_cached=load_cached)

    # 4. Merge & Engineer
    # Train
    print("Preparing Train Set...")
    df_train, kin_cols, vis_cols, gate_cols = merge_and_engineer(
        df_train_meta, df_track_proc, df_vis_proc
    )

    # Val
    print("Preparing Validation Set...")
    df_val, _, _, _ = merge_and_engineer(df_val_meta, df_track_proc, df_vis_proc)

    # Test
    print("Preparing Test Set...")
    df_test_tracking = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
    )
    df_test_helmets = pd.read_csv(
        os.path.join(Config.INPUT_DIR, "test_baseline_helmets.csv")
    )

    # Process test raw data (no caching for test usually, or separate cache)
    # We reuse functions but don't overwrite train cache.
    # Actually, process_tracking caches based on filename. We should handle test separately or disable cache.
    # For simplicity, we process test in memory without caching to file to avoid conflicts.
    df_track_test_proc = process_tracking(df_test_tracking, load_cached=False)
    df_vis_test_proc = process_visuals(df_test_helmets, load_cached=False)

    df_test, _, _, _ = merge_and_engineer(
        df_test_meta, df_track_test_proc, df_vis_test_proc
    )

    # 5. Encoding & Scaling
    print("Encoding and Scaling...")

    # Categoricals (Position, Team) - Fit on Tracking data (contains all players)
    # P1 is always a player.
    le_pos = LabelEncoder()
    le_team = LabelEncoder()

    # Fit on all unique values in tracking
    all_pos = df_tracking["position"].unique()
    all_team = df_tracking["team"].unique()
    le_pos.fit(all_pos)
    le_team.fit(all_team)

    # Handle unknown in test?
    # Helper to safe transform
    def safe_transform(le, series):
        # Map unknown to first class (or mode)
        return series.map(lambda x: x if x in le.classes_ else le.classes_[0]).apply(
            lambda x: np.where(le.classes_ == x)[0][0]
        )

    df_train["pos_idx"] = safe_transform(le_pos, df_train["position_1"])
    df_train["team_idx"] = safe_transform(le_team, df_train["team_1"])

    df_val["pos_idx"] = safe_transform(le_pos, df_val["position_1"])
    df_val["team_idx"] = safe_transform(le_team, df_val["team_1"])

    df_test["pos_idx"] = safe_transform(le_pos, df_test["position_1"])
    df_test["team_idx"] = safe_transform(le_team, df_test["team_1"])

    # Scaler
    scaler_kin = RobustScaler()
    scaler_vis = RobustScaler()
    scaler_gate = RobustScaler()

    # Fit on Train
    X_kin_train = df_train[kin_cols].values.astype(np.float32)
    X_vis_train = df_train[vis_cols].values.astype(np.float32)
    X_gate_train = df_train[gate_cols].values.astype(np.float32)

    scaler_kin.fit(X_kin_train)
    scaler_vis.fit(X_vis_train)
    scaler_gate.fit(X_gate_train)

    # Save scalers
    joblib.dump(scaler_kin, os.path.join(Config.WORKING_DIR, "scaler_kin.joblib"))
    joblib.dump(scaler_vis, os.path.join(Config.WORKING_DIR, "scaler_vis.joblib"))
    joblib.dump(scaler_gate, os.path.join(Config.WORKING_DIR, "scaler_gate.joblib"))
    joblib.dump(le_pos, os.path.join(Config.WORKING_DIR, "le_pos.joblib"))
    joblib.dump(le_team, os.path.join(Config.WORKING_DIR, "le_team.joblib"))

    # Transform all
    def transform_and_create_dataset(df, X_kin_raw, X_vis_raw, X_gate_raw, y=None):
        X_kin = scaler_kin.transform(X_kin_raw)
        X_vis = scaler_vis.transform(X_vis_raw)
        X_gate = scaler_gate.transform(X_gate_raw)

        pos = df["pos_idx"].values
        team = df["team_idx"].values

        return ContactDataset(X_kin, X_vis, X_gate, pos, team, y)

    train_dataset = transform_and_create_dataset(
        df_train, X_kin_train, X_vis_train, X_gate_train, df_train["contact"].values
    )

    val_dataset = transform_and_create_dataset(
        df_val,
        df_val[kin_cols].values.astype(np.float32),
        df_val[vis_cols].values.astype(np.float32),
        df_val[gate_cols].values.astype(np.float32),
        df_val["contact"].values,
    )

    test_dataset = transform_and_create_dataset(
        df_test,
        df_test[kin_cols].values.astype(np.float32),
        df_test[vis_cols].values.astype(np.float32),
        df_test[gate_cols].values.astype(np.float32),
        None,
    )

    # Metadata for model init
    dims = {
        "kin_input_dim": X_kin_train.shape[1],
        "vis_input_dim": X_vis_train.shape[1],
        "gate_input_dim": X_gate_train.shape[1],
        "num_pos": len(le_pos.classes_),
        "num_team": len(le_team.classes_),
    }

    # Return datasets and test metadata (for submission ID mapping)
    return train_dataset, val_dataset, test_dataset, dims, df_test["contact_id"].values
