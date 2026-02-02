import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Dataset Class
# =========================================================================


class ContactDataset(Dataset):
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


# =========================================================================
# Feature Engineering Functions
# =========================================================================


def engineer_tracking_features(df_tracking):
    """
    Applies Entity-First processing: Lags, Windows, Clamping, Angular conversion.
    Returns a wide DataFrame with features for window t-5 to t+5.
    """
    # Sort for shifting
    df = df_tracking.sort_values(["game_play", "nfl_player_id", "step"]).copy()

    # Angular components
    df["dir_sin"] = np.sin(np.deg2rad(df["direction"].fillna(0)))
    df["dir_cos"] = np.cos(np.deg2rad(df["direction"].fillna(0)))
    df["o_sin"] = np.sin(np.deg2rad(df["orientation"].fillna(0)))
    df["o_cos"] = np.cos(np.deg2rad(df["orientation"].fillna(0)))

    # Clamping (Explicit Numerical Stability)
    # Clamp speed/accel/sa to reasonable physical limits to prevent outliers
    df["speed"] = df["speed"].clip(0, 50)
    df["acceleration"] = df["acceleration"].clip(0, 50)
    df["sa"] = df["sa"].clip(-50, 50)

    # Features to window
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "sa",
        "dir_sin",
        "dir_cos",
        "o_sin",
        "o_cos",
    ]

    # Generate Windowed Features (t-5 to t+5)
    # We create a wide dataframe where each row 'step' contains info from step-5 to step+5
    # Since data is sorted by step, we can use shift.
    # Window size 5 means lags -5, -4, ..., 0, ..., 4, 5

    wide_features = {}

    # Group by player to respect boundaries
    # Using groupby().shift() is slow.
    # Since it's sorted, we can just shift and mask boundaries.
    # However, game_play/player boundaries must be respected.
    # Fast approach: Use groupby object
    grp = df.groupby(["game_play", "nfl_player_id"])

    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)  # -5 to 5

    for col in feature_cols:
        for lag in lags:
            # lag > 0 is past (shift positive), lag < 0 is future (shift negative)
            # Standard shift: shift(1) gets prev row.
            # We want t-5 (past) to t+5 (future).
            # Let's denote lag k: value at t-k.
            # So lag=5 means t-5. shift(5).
            # lag=-5 means t+5. shift(-5).
            col_name = f"{col}_lag_{lag}"
            wide_features[col_name] = grp[col].shift(lag)

    # Combine
    df_wide = pd.concat(wide_features, axis=1)

    # Add keys back for merging
    df_wide["game_play"] = df["game_play"]
    df_wide["nfl_player_id"] = df["nfl_player_id"]
    df_wide["step"] = df["step"]

    # Fill NaNs (edges of play) with nearest valid observation or 0
    # Forward fill then backward fill within groups is best, but slow.
    # Simple fill 0 for now as specified "Ground Velocity = 0" logic extends to missing.
    # But for positions, 0 is bad.
    # Given the density, we'll fill with 0 for motion and keep NaNs for pos to handle later or ffill.
    # Let's use ffill/bfill on the whole DF since it's sorted by play/player
    # Note: This might bleed across players if not careful, but we have keys.
    # Actually, let's just fillna(0) for motion and handle position carefully?
    # Better: The model should handle 0s if we use relative coords.
    df_wide = df_wide.fillna(0)

    return df_wide


def engineer_visual_features(df_helmets):
    """
    Applies Max-Pooling Selection Strategy to select best helmet box.
    """
    # Calculate Area
    df = df_helmets.copy()
    df["area"] = df["width"] * df["height"]

    # Sort by area descending to prioritize largest boxes
    df = df.sort_values(
        ["game_play", "frame", "nfl_player_id", "area"],
        ascending=[True, True, True, False],
    )

    # Drop duplicates to keep only the largest box per player-frame
    df_best = df.drop_duplicates(
        subset=["game_play", "frame", "nfl_player_id"], keep="first"
    )

    # Select features
    keep_cols = [
        "game_play",
        "frame",
        "nfl_player_id",
        "left",
        "width",
        "top",
        "height",
    ]
    return df_best[keep_cols]


# =========================================================================
# Main Processing Logic
# =========================================================================


def process_data(
    metadata_path,
    tracking_path,
    helmets_path,
    cache_path,
    load_cached_data=True,
    is_train=True,
):
    """
    Loads raw data, merges, computes features, and caches the result.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            # Separate features and labels
            if "contact" in df.columns:
                labels = df["contact"].values
                df = df.drop(columns=["contact"])
            else:
                labels = None

            # Identify columns
            vis_cols = [c for c in df.columns if c.startswith("vis_")]
            kin_cols = [c for c in df.columns if c not in vis_cols]

            return (
                df[kin_cols].values.astype(np.float32),
                df[vis_cols].values.astype(np.float32),
                labels,
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing data from {metadata_path}...")

    # 2. Load Data
    df_meta = pd.read_csv(metadata_path)

    # Optimization: Filter tracking/helmets to only relevant game_plays
    relevant_plays = df_meta["game_play"].unique()

    # Load Tracking
    df_tracking = pd.read_csv(tracking_path)
    df_tracking = df_tracking[df_tracking["game_play"].isin(relevant_plays)]

    # Load Helmets
    df_helmets = pd.read_csv(helmets_path)
    df_helmets = df_helmets[df_helmets["game_play"].isin(relevant_plays)]

    # 3. Preprocess Inputs
    print("Engineering tracking features...")
    track_wide = engineer_tracking_features(df_tracking)

    print("Engineering visual features...")
    vis_best = engineer_visual_features(df_helmets)

    # 4. Merge
    print("Merging datasets...")
    # Prepare metadata
    # Ensure nfl_player_id_2 is string for 'G' handling
    df_meta["nfl_player_id_2"] = df_meta["nfl_player_id_2"].astype(str)

    # Merge P1 Tracking
    # track_wide has [game_play, nfl_player_id, step]
    df_merged = df_meta.merge(
        track_wide.add_suffix("_p1"),
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play_p1", "nfl_player_id_p1", "step_p1"],
        how="left",
    )

    # Merge P2 Tracking
    # We need to handle 'G'. If 'G', merge will fail (NaN), which we fill later.
    # Convert P2 ID to numeric where possible for merge, 'G' becomes NaN
    df_merged["p2_id_num"] = pd.to_numeric(
        df_merged["nfl_player_id_2"], errors="coerce"
    )

    df_merged = df_merged.merge(
        track_wide.add_suffix("_p2"),
        left_on=["game_play", "p2_id_num", "step"],
        right_on=["game_play_p2", "nfl_player_id_p2", "step_p2"],
        how="left",
    )

    # Merge Visuals
    # Map step to frame: frame = 300 + step * 5.994
    df_merged["frame_approx"] = (300 + df_merged["step"] * 5.994).round().astype(int)

    # Merge P1 Visual
    df_merged = df_merged.merge(
        vis_best.add_suffix("_p1"),
        left_on=["game_play", "frame_approx", "nfl_player_id_1"],
        right_on=["game_play_p1", "frame_p1", "nfl_player_id_p1"],
        how="left",
    )

    # Merge P2 Visual
    df_merged = df_merged.merge(
        vis_best.add_suffix("_p2"),
        left_on=["game_play", "frame_approx", "p2_id_num"],
        right_on=["game_play_p2", "frame_p2", "nfl_player_id_p2"],
        how="left",
    )

    # 5. Imputation & Relative Features
    print("Computing relative features...")

    # Identify Lag Columns
    lags = range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1)

    kinematic_features = {}

    # Ground Imputation Logic:
    # If P2 is Ground (NaN in tracking), set P2 pos = P1 pos, P2 vel = 0
    is_ground = df_merged["nfl_player_id_2"] == "G"

    for lag in lags:
        suffix = f"_lag_{lag}"

        # Extract raw columns
        x1 = df_merged[f"x_position{suffix}_p1"].fillna(0)
        y1 = df_merged[f"y_position{suffix}_p1"].fillna(0)
        s1 = df_merged[f"speed{suffix}_p1"].fillna(0)
        a1 = df_merged[f"acceleration{suffix}_p1"].fillna(0)
        sa1 = df_merged[f"sa{suffix}_p1"].fillna(0)
        ds1 = df_merged[f"dir_sin{suffix}_p1"].fillna(0)
        dc1 = df_merged[f"dir_cos{suffix}_p1"].fillna(0)
        os1 = df_merged[f"o_sin{suffix}_p1"].fillna(0)
        oc1 = df_merged[f"o_cos{suffix}_p1"].fillna(0)

        x2 = df_merged[f"x_position{suffix}_p2"]
        y2 = df_merged[f"y_position{suffix}_p2"]
        s2 = df_merged[f"speed{suffix}_p2"]
        a2 = df_merged[f"acceleration{suffix}_p2"]
        sa2 = df_merged[f"sa{suffix}_p2"]
        ds2 = df_merged[f"dir_sin{suffix}_p2"]
        dc2 = df_merged[f"dir_cos{suffix}_p2"]
        os2 = df_merged[f"o_sin{suffix}_p2"]
        oc2 = df_merged[f"o_cos{suffix}_p2"]

        # Impute P2 for Ground
        # Pos = P1, Vel/Accel = 0, Angles = 0
        x2 = np.where(is_ground, x1, x2.fillna(0))
        y2 = np.where(is_ground, y1, y2.fillna(0))
        s2 = np.where(is_ground, 0, s2.fillna(0))
        a2 = np.where(is_ground, 0, a2.fillna(0))
        sa2 = np.where(is_ground, 0, sa2.fillna(0))
        ds2 = np.where(is_ground, 0, ds2.fillna(0))
        dc2 = np.where(is_ground, 0, dc2.fillna(0))
        os2 = np.where(is_ground, 0, os2.fillna(0))
        oc2 = np.where(is_ground, 0, oc2.fillna(0))

        # Compute Relative Features (Strict Invariance)
        kinematic_features[f"x_diff{suffix}"] = x1 - x2
        kinematic_features[f"y_diff{suffix}"] = y1 - y2

        # Absolute Motion Features (Invariant to global position, so okay)
        kinematic_features[f"s1{suffix}"] = s1
        kinematic_features[f"s2{suffix}"] = s2
        kinematic_features[f"a1{suffix}"] = a1
        kinematic_features[f"a2{suffix}"] = a2
        kinematic_features[f"sa1{suffix}"] = sa1
        kinematic_features[f"sa2{suffix}"] = sa2

        # Relative Angles via Sin/Cos
        # We pass raw components; MLP learns relation.
        # Or explicitly: sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
        # Let's pass components to allow flexibility
        kinematic_features[f"ds1{suffix}"] = ds1
        kinematic_features[f"dc1{suffix}"] = dc1
        kinematic_features[f"ds2{suffix}"] = ds2
        kinematic_features[f"dc2{suffix}"] = dc2
        kinematic_features[f"os1{suffix}"] = os1
        kinematic_features[f"oc1{suffix}"] = oc1
        kinematic_features[f"os2{suffix}"] = os2
        kinematic_features[f"oc2{suffix}"] = oc2

    # Current Step Distance (Resolution Enhancement)
    # lag 0 is suffix "_lag_0"
    dx_0 = kinematic_features["x_diff_lag_0"]
    dy_0 = kinematic_features["y_diff_lag_0"]
    dist = np.sqrt(dx_0**2 + dy_0**2)
    kinematic_features["log_dist"] = np.log1p(dist)

    # Visual Features
    visual_features = {}
    vis_cols = ["left", "width", "top", "height"]
    for c in vis_cols:
        # P1
        visual_features[f"vis_{c}_p1"] = df_merged[f"{c}_p1"].fillna(0)
        # P2 (Ground -> 0)
        visual_features[f"vis_{c}_p2"] = np.where(
            is_ground, 0, df_merged[f"{c}_p2"].fillna(0)
        )

    # Construct Final DataFrames
    df_kin = pd.DataFrame(kinematic_features)
    df_vis = pd.DataFrame(visual_features)

    # Combine for saving
    df_final = pd.concat([df_kin, df_vis], axis=1)

    if is_train and "contact" in df_merged.columns:
        df_final["contact"] = df_merged["contact"]
        labels = df_merged["contact"].values
    else:
        labels = None

    # Save to Parquet
    print(f"Saving features to {cache_path}...")
    df_final.to_parquet(cache_path)

    # Return numpy arrays
    return df_kin.values.astype(np.float32), df_vis.values.astype(np.float32), labels


# =========================================================================
# Data Loaders
# =========================================================================


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates data loading, scaling, and DataLoader creation.
    """
    seed_everything(Config.SEED)

    # 1. Process/Load Data
    print("\n--- Preparing Training Data ---")
    X_kin_train, X_vis_train, y_train = process_data(
        Config.METADATA_TRAIN,
        Config.TRAIN_TRACKING_PATH,
        Config.TRAIN_HELMETS_PATH,
        Config.CACHE_TRAIN_FEATURES,
        load_cached_data,
        is_train=True,
    )

    print("\n--- Preparing Validation Data ---")
    X_kin_val, X_vis_val, y_val = process_data(
        Config.METADATA_VAL,
        Config.TRAIN_TRACKING_PATH,
        Config.TRAIN_HELMETS_PATH,
        Config.CACHE_VAL_FEATURES,
        load_cached_data,
        is_train=True,
    )

    print("\n--- Preparing Test Data ---")
    X_kin_test, X_vis_test, _ = process_data(
        Config.METADATA_TEST,
        Config.TEST_TRACKING_PATH,
        Config.TEST_HELMETS_PATH,
        Config.CACHE_TEST_FEATURES,
        load_cached_data,
        is_train=False,
    )

    # 2. Scaling
    # We scale Kinematic and Visual features separately or together?
    # Usually separately is safer if distributions differ wildly.
    # Let's use one scaler for Kinematics and one for Visuals.

    print("\nFitting Scalers...")
    scaler_kin = StandardScaler()
    scaler_vis = StandardScaler()

    # Fit on Train
    X_kin_train = scaler_kin.fit_transform(X_kin_train)
    X_vis_train = scaler_vis.fit_transform(X_vis_train)

    # Transform Val/Test
    X_kin_val = scaler_kin.transform(X_kin_val)
    X_vis_val = scaler_vis.transform(X_vis_val)

    X_kin_test = scaler_kin.transform(X_kin_test)
    X_vis_test = scaler_vis.transform(X_vis_test)

    # Save Scalers
    joblib.dump({"kin": scaler_kin, "vis": scaler_vis}, Config.SCALER_PATH)

    # 3. Create Datasets
    train_dataset = ContactDataset(X_kin_train, X_vis_train, y_train)
    val_dataset = ContactDataset(X_kin_val, X_vis_val, y_val)
    test_dataset = ContactDataset(X_kin_test, X_vis_test, labels=None)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
