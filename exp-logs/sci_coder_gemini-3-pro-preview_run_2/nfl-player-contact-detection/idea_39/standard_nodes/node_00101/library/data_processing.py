import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class NFLDataset(Dataset):
    def __init__(self, features, labels=None, training=False):
        """
        Args:
            features (np.ndarray): Standardized feature matrix.
            labels (np.ndarray, optional): Binary labels.
            training (bool): Whether to apply training augmentations (noise injection).
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.training = training
        self.noise_sigma = Config.NOISE_SIGMA

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]

        # Stochastic Noise Injection during training
        if self.training and self.noise_sigma > 0:
            noise = torch.randn_like(x) * self.noise_sigma
            x = x + noise

        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


# -----------------------------------------------------------------------------
# Preprocessing Functions
# -----------------------------------------------------------------------------
def preprocess_tracking(df_tracking):
    """
    Applies Entity-First Feature Generation.
    Generates lags and windowed features for each player BEFORE merging with labels.
    """
    # Ensure data is sorted for shifting
    df_tracking = df_tracking.sort_values(
        by=["game_play", "nfl_player_id", "step"]
    ).copy()

    # Features to window
    feature_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Generate lags
    # Window: t-WINDOW_PRE to t+WINDOW_POST
    # shift(k) shifts data down by k. To get t-1 (past) at t, we look at previous row -> shift(1)
    # To get t+1 (future) at t, we look at next row -> shift(-1)

    # We want a wide format. The original columns represent t=0.
    # We add columns for t-5...t-1 and t+1...t+5

    shifts = range(-Config.WINDOW_PRE, Config.WINDOW_POST + 1)

    # We process by group to ensure boundaries are respected, but for speed on large data,
    # we can use global shifts if we verify game_play/player_id match.
    # However, groupby shift is safer and reasonably fast.

    grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

    result_dfs = []

    for lag in shifts:
        if lag == 0:
            # Keep original columns, maybe rename to _0 for consistency or keep as is
            # We keep original names for t=0 to simplify downstream logic, or rename all.
            # Let's rename all to suffix _t{lag}
            suffix = f"_t{lag}"
            df_shifted = df_tracking[feature_cols].add_suffix(suffix)
        else:
            # lag > 0 (e.g. 5) means Future in our config context?
            # Config: WINDOW_PRE=5 (Past), WINDOW_POST=5 (Future)
            # Usually lag k means t-k.
            # Let's be explicit:
            # offset -5 means t-5. shift(5).
            # offset +5 means t+5. shift(-5).

            # We iterate offset from -5 to +5
            offset = lag
            shift_amount = -offset  # shift(1) gives previous row (t-1)

            suffix = f"_t{offset}"
            df_shifted = grouped[feature_cols].shift(shift_amount).add_suffix(suffix)

        result_dfs.append(df_shifted)

    # Concatenate all shifted features
    # We rely on the index being preserved
    df_wide = pd.concat(result_dfs, axis=1)

    # Add keys back
    df_wide["game_play"] = df_tracking["game_play"]
    df_wide["nfl_player_id"] = df_tracking["nfl_player_id"]
    df_wide["step"] = df_tracking["step"]

    # Fill NaNs created by shifting (edges of plays) with appropriate values (e.g. forward/back fill or 0)
    # For physics continuity, ffill/bfill is better than 0.
    # Since we grouped, we can fill within groups.
    # However, concat lost the group structure.
    # A simple ffill/bfill on the whole DF might bleed across plays if not careful.
    # Given the strict windowing, we can fill with 0 or nearest.
    # Let's use 0 for missing steps outside of play boundaries as simplest safe default.
    df_wide = df_wide.fillna(0)

    return df_wide


def preprocess_helmets(df_helmets):
    """
    Applies Max-Pooling Selection Strategy to select the best view for each helmet.
    """
    # Calculate area
    df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

    # Sort by game_play, nfl_player_id, frame, and area (descending)
    df_helmets = df_helmets.sort_values(
        by=["game_play", "nfl_player_id", "frame", "area"],
        ascending=[True, True, True, False],
    )

    # Drop duplicates to keep the one with max area
    df_best_view = df_helmets.drop_duplicates(
        subset=["game_play", "nfl_player_id", "frame"], keep="first"
    )

    # Select relevant columns
    cols = ["game_play", "nfl_player_id", "frame", "left", "width", "top", "height"]
    return df_best_view[cols]


def feature_engineering(df_labels, df_tracking, df_helmets, is_train=True):
    """
    Merges data sources, handles ground imputation, applies clamping and physics calculations.
    """
    # 1. Prepare Labels
    # Map step to frame for helmet merging: frame approx 300 + step * 5.994
    df_labels["frame"] = (300 + df_labels["step"] * 5.994).round().astype(int)

    # Ensure nfl_player_id_2 is numeric (G becomes NaN)
    df_labels["nfl_player_id_2_num"] = pd.to_numeric(
        df_labels["nfl_player_id_2"], errors="coerce"
    )

    # 2. Merge Tracking Data (Entity-First: already windowed)
    # Merge Player 1
    df_merged = df_labels.merge(
        df_tracking,
        left_on=["game_play", "nfl_player_id_1", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
    )

    # Rename P1 columns (they currently have suffixes like _t0, _t-1 etc.)
    # We need to distinguish P1 vs P2.
    # The columns in df_tracking are like x_position_t0, x_position_t-1...
    # We will rename them to p1_x_position_t0...

    # Identify tracking columns (exclude keys)
    track_cols = [
        c
        for c in df_tracking.columns
        if c not in ["game_play", "nfl_player_id", "step"]
    ]
    rename_dict_p1 = {c: f"p1_{c}" for c in track_cols}
    df_merged = df_merged.rename(columns=rename_dict_p1)

    # Merge Player 2
    # We merge on the numeric ID. 'G' will result in NaNs.
    df_merged = df_merged.merge(
        df_tracking,
        left_on=["game_play", "nfl_player_id_2_num", "step"],
        right_on=["game_play", "nfl_player_id", "step"],
        how="left",
        suffixes=("", "_p2_temp"),
    )

    # Rename P2 columns
    rename_dict_p2 = {c: f"p2_{c}" for c in track_cols}
    df_merged = df_merged.rename(columns=rename_dict_p2)

    # 3. Hybrid Ground Imputation
    # Where nfl_player_id_2 is 'G', impute P2 features from P1 (pos) or 0 (vel)
    is_ground = df_merged["nfl_player_id_2"] == "G"

    # For every window offset t
    shifts = range(-Config.WINDOW_PRE, Config.WINDOW_POST + 1)
    for t in shifts:
        suffix = f"_t{t}"

        # Position: P2 = P1
        df_merged.loc[is_ground, f"p2_x_position{suffix}"] = df_merged.loc[
            is_ground, f"p1_x_position{suffix}"
        ]
        df_merged.loc[is_ground, f"p2_y_position{suffix}"] = df_merged.loc[
            is_ground, f"p1_y_position{suffix}"
        ]

        # Velocity/Accel: P2 = 0
        for col in ["speed", "acceleration", "sa"]:
            df_merged.loc[is_ground, f"p2_{col}{suffix}"] = 0.0

        # Orientation/Direction: P2 = 0 (arbitrary, but consistent)
        for col in ["orientation", "direction"]:
            df_merged.loc[is_ground, f"p2_{col}{suffix}"] = 0.0

    # 4. Merge Helmets
    # Merge P1 Helmets
    df_merged = df_merged.merge(
        df_helmets,
        left_on=["game_play", "nfl_player_id_1", "frame"],
        right_on=["game_play", "nfl_player_id", "frame"],
        how="left",
    )
    df_merged = df_merged.rename(
        columns={c: f"p1_vis_{c}" for c in ["left", "width", "top", "height"]}
    )

    # Merge P2 Helmets
    df_merged = df_merged.merge(
        df_helmets,
        left_on=["game_play", "nfl_player_id_2_num", "frame"],
        right_on=["game_play", "nfl_player_id", "frame"],
        how="left",
        suffixes=("", "_p2_vis"),
    )
    df_merged = df_merged.rename(
        columns={c: f"p2_vis_{c}" for c in ["left", "width", "top", "height"]}
    )

    # Ground Visuals: Set to 0 (already NaN for G usually, but explicit 0 is safer for scaler)
    vis_cols = ["left", "width", "top", "height"]
    for c in vis_cols:
        df_merged.loc[is_ground, f"p2_vis_{c}"] = 0.0

    # Fill remaining NaNs (missing tracking/helmets) with 0
    df_merged = df_merged.fillna(0)

    # 5. Explicit Physics & Clamping
    # Calculate Distances and Relative Angles for t=0 (and potentially others, but t=0 is most critical)
    # We can compute distance for all timesteps to feed the deep network

    for t in shifts:
        suffix = f"_t{t}"
        dx = df_merged[f"p1_x_position{suffix}"] - df_merged[f"p2_x_position{suffix}"]
        dy = df_merged[f"p1_y_position{suffix}"] - df_merged[f"p2_y_position{suffix}"]

        dist_col = f"distance{suffix}"
        df_merged[dist_col] = np.sqrt(dx**2 + dy**2)

        # Log distance (Resolution Enhancement)
        df_merged[f"log_distance{suffix}"] = np.log1p(df_merged[dist_col])

        # Clamping
        # Clamp speed, accel, distance
        for col in ["speed", "acceleration"]:
            limit = Config.PHYSICAL_RANGES[col]
            df_merged[f"p1_{col}{suffix}"] = df_merged[f"p1_{col}{suffix}"].clip(
                upper=limit
            )
            df_merged[f"p2_{col}{suffix}"] = df_merged[f"p2_{col}{suffix}"].clip(
                upper=limit
            )

        df_merged[dist_col] = df_merged[dist_col].clip(
            upper=Config.PHYSICAL_RANGES["distance"]
        )

    # Angular Continuity (Shortest Arc) for relative orientation at t=0
    # We'll add a feature: relative_orientation
    p1_o = df_merged["p1_orientation_t0"]
    p2_o = df_merged["p2_orientation_t0"]
    diff = (p1_o - p2_o).abs()
    df_merged["rel_orientation_t0"] = np.minimum(diff, 360 - diff)

    # 6. Select Final Feature Columns
    # Exclude metadata
    meta_cols = [
        "contact_id",
        "game_play",
        "step",
        "datetime",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "nfl_player_id_2_num",
        "contact",
        "path_endzone",
        "path_sideline",
        "path_all29",
        "frame",
        "nfl_player_id",
        "nfl_player_id_p2_temp",
        "nfl_player_id_p2_vis",
    ]

    feature_cols = [c for c in df_merged.columns if c not in meta_cols]

    # Return features and target
    if "contact" in df_merged.columns:
        return df_merged[feature_cols], df_merged["contact"]
    else:
        return df_merged[feature_cols], None


# -----------------------------------------------------------------------------
# Main Data Loading & Orchestration
# -----------------------------------------------------------------------------
def get_data(load_cached_data=True, debug=False):
    """
    Orchestrates the data pipeline.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed Parquet files.
        debug (bool): If True, subsamples the data for rapid debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, feature_dim)
    """
    seed_everything(Config.SEED)

    # Paths
    train_pq = Config.CACHE_TRAIN_PARQUET
    val_pq = Config.CACHE_VAL_PARQUET
    test_pq = Config.CACHE_TEST_PARQUET
    scaler_path = Config.CACHE_SCALER

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_pq)
        and os.path.exists(val_pq)
        and os.path.exists(test_pq)
        and os.path.exists(scaler_path)
    ):
        print("Loading cached features...")
        X_train = pd.read_parquet(train_pq)
        y_train = np.load(train_pq.replace(".parquet", "_labels.npy"))

        X_val = pd.read_parquet(val_pq)
        y_val = np.load(val_pq.replace(".parquet", "_labels.npy"))

        X_test = pd.read_parquet(test_pq)

        # Load scaler just to ensure it exists, though data is likely already scaled if cached?
        # Usually cache stores raw features or scaled features.
        # To be safe and allow re-scaling if needed, let's assume cache is PRE-SCALING or POST-SCALING.
        # Let's assume cache is POST-SCALING for efficiency.

        # If debug, sample
        if debug:
            X_train = X_train.iloc[:10000]
            y_train = y_train[:10000]
            X_val = X_val.iloc[:2000]
            y_val = y_val[:2000]
            X_test = X_test.iloc[:2000]

        return (
            NFLDataset(X_train.values, y_train, training=True),
            NFLDataset(X_val.values, y_val, training=False),
            NFLDataset(X_test.values, None, training=False),
            X_train.shape[1],
        )

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Metadata
    df_train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    df_val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))
    df_test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    if debug:
        df_train_meta = df_train_meta.sample(10000, random_state=Config.SEED)
        df_val_meta = df_val_meta.sample(2000, random_state=Config.SEED)
        df_test_meta = df_test_meta.sample(2000, random_state=Config.SEED)

    # Load Raw Data
    # We load full tracking/helmets and filter by game_plays in metadata to save memory
    print("Loading raw tracking/helmets...")

    # Get all relevant game_plays
    all_gps = pd.concat(
        [
            df_train_meta["game_play"],
            df_val_meta["game_play"],
            df_test_meta["game_play"],
        ]
    ).unique()

    # Helper to load and filter
    def load_filtered(path, gp_col="game_play"):
        df = pd.read_csv(path)
        return df[df[gp_col].isin(all_gps)].copy()

    df_track_train = load_filtered(
        os.path.join(Config.INPUT_DIR, "train_player_tracking.csv")
    )
    df_helm_train = load_filtered(
        os.path.join(Config.INPUT_DIR, "train_baseline_helmets.csv")
    )

    df_track_test = load_filtered(
        os.path.join(Config.INPUT_DIR, "test_player_tracking.csv")
    )
    df_helm_test = load_filtered(
        os.path.join(Config.INPUT_DIR, "test_baseline_helmets.csv")
    )

    # Combine tracking/helmets for processing (split by train/test source)
    # Note: Train and Val come from train source. Test from test source.

    # 3. Preprocess Inputs
    print("Preprocessing Tracking & Helmets...")
    track_train_proc = preprocess_tracking(df_track_train)
    helm_train_proc = preprocess_helmets(df_helm_train)

    track_test_proc = preprocess_tracking(df_track_test)
    helm_test_proc = preprocess_helmets(df_helm_test)

    # Clear raw data
    del df_track_train, df_helm_train, df_track_test, df_helm_test
    gc.collect()

    # 4. Feature Engineering & Merge
    print("Feature Engineering (Train)...")
    X_train, y_train = feature_engineering(
        df_train_meta, track_train_proc, helm_train_proc
    )

    print("Feature Engineering (Val)...")
    X_val, y_val = feature_engineering(df_val_meta, track_train_proc, helm_train_proc)

    print("Feature Engineering (Test)...")
    X_test, _ = feature_engineering(df_test_meta, track_test_proc, helm_test_proc)

    # 5. Scaling
    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # 6. Caching
    print("Saving cache...")
    X_train_scaled.to_parquet(train_pq)
    np.save(train_pq.replace(".parquet", "_labels.npy"), y_train.values)

    X_val_scaled.to_parquet(val_pq)
    np.save(val_pq.replace(".parquet", "_labels.npy"), y_val.values)

    X_test_scaled.to_parquet(test_pq)
    dump(scaler, scaler_path)

    return (
        NFLDataset(X_train_scaled.values, y_train.values, training=True),
        NFLDataset(X_val_scaled.values, y_val.values, training=False),
        NFLDataset(X_test_scaled.values, None, training=False),
        X_train_scaled.shape[1],
    )
