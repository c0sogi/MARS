import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library import config, utils


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class NFLContactDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


# -----------------------------------------------------------------------------
# Data Processing Functions
# -----------------------------------------------------------------------------


def load_metadata(split):
    """Loads metadata CSVs from the metadata directory."""
    path = os.path.join(config.METADATA_DIR, f"{split}.csv")
    return pd.read_csv(path)


def load_raw_tracking(split):
    """Loads raw tracking data."""
    if split in ["train", "validation"]:
        path = os.path.join(config.INPUT_DIR, "train_player_tracking.csv")
    else:
        path = os.path.join(config.INPUT_DIR, "test_player_tracking.csv")

    df = pd.read_csv(path)

    # Cast types to save memory
    df["nfl_player_id"] = df["nfl_player_id"].astype("int32")
    df["step"] = df["step"].astype("int16")
    return df


def load_raw_helmets(split):
    """Loads raw helmet data."""
    if split in ["train", "validation"]:
        path = os.path.join(config.INPUT_DIR, "train_baseline_helmets.csv")
    else:
        path = os.path.join(config.INPUT_DIR, "test_baseline_helmets.csv")

    df = pd.read_csv(path)
    return df


def create_windowed_features(df, group_cols, feature_cols, lags):
    """
    Creates windowed features using groupby and shift.
    """
    out_df = df[group_cols + ["step"]].copy()
    grouped = df.groupby(group_cols)

    for lag in lags:
        # shift(k) shifts data down by k.
        # To get t-5 (past) at t, we need to shift data from t-5 down to t. That is shift(5).
        # To get t+5 (future) at t, we need to shift data from t+5 up to t. That is shift(-5).
        shift_amount = lag
        suffix = f"_lag_{lag}"

        for col in feature_cols:
            out_df[f"{col}{suffix}"] = grouped[col].shift(shift_amount)

    return out_df


def process_tracking_stream(df_tracking, game_plays=None):
    """
    Processes tracking data: filters, sorts, and creates windows.
    """
    if game_plays is not None:
        df_tracking = df_tracking[df_tracking["game_play"].isin(game_plays)].copy()

    # Sort
    df_tracking = df_tracking.sort_values(["game_play", "nfl_player_id", "step"])

    # Base features to window
    raw_cols = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Generate windows: t-5 to t+5
    # Lags represent the offset. We want features from t-5, t-4... t+5.
    # In create_windowed_features logic:
    # If we want value at t-5, we use lag=5.
    lags = list(range(config.WINDOW_PRE, -config.WINDOW_POST - 1, -1))
    # Rename lags to be intuitive: -5 means 5 steps back.
    # My helper uses shift(lag). shift(5) gives t-5.
    # So if I want t-5, I pass lag=5.
    # Let's use standard integer notation: -5 to +5.
    # And adjust shift inside helper?
    # Let's adjust helper call.
    # We want lags: -5, -4, ..., 0, ..., 5.
    # shift(5) -> t-5. shift(-5) -> t+5.
    # So lag k corresponds to shift(-k)? No.
    # t_new = t_old + shift.
    # We want row t to contain data from t+k.
    # shift(-k) moves data from t+k to t.
    # So for offset k (e.g. -5), we want shift(-(-5)) = shift(5).
    # For offset k (e.g. +5), we want shift(-5).

    lag_range = list(range(-config.WINDOW_PRE, config.WINDOW_POST + 1))

    out_df = df_tracking[["game_play", "nfl_player_id", "step"]].copy()
    grouped = df_tracking.groupby(["game_play", "nfl_player_id"])

    for lag in lag_range:
        # lag is -5 (past) to +5 (future)
        # To get past data (t-5) at t, we shift(5).
        # To get future data (t+5) at t, we shift(-5).
        shift_val = -lag
        suffix = f"_lag_{lag}"

        for col in raw_cols:
            out_df[f"{col}{suffix}"] = grouped[col].shift(shift_val)

    out_df = out_df.fillna(0.0)
    return out_df


def process_visual_stream(df_helmets, game_plays=None):
    """
    Processes visual data: filters, maps step, max-pools, and creates windows.
    """
    if game_plays is not None:
        df_helmets = df_helmets[df_helmets["game_play"].isin(game_plays)].copy()

    # Map frame to step (Frame 300 is step 0. 59.94 fps)
    df_helmets["step"] = ((df_helmets["frame"] - 300) / 5.994).round().astype(int)

    # Calculate Area
    df_helmets["area"] = df_helmets["width"] * df_helmets["height"]

    # Max Pooling: Select box with largest area per (game_play, step, player)
    df_helmets = df_helmets.sort_values("area", ascending=False)
    df_helmets = df_helmets.drop_duplicates(
        subset=["game_play", "step", "nfl_player_id"]
    )
    df_helmets = df_helmets.sort_values(["game_play", "nfl_player_id", "step"])

    vis_cols = ["left", "top", "width", "height", "area"]
    lag_range = list(range(-config.WINDOW_PRE, config.WINDOW_POST + 1))

    out_df = df_helmets[["game_play", "nfl_player_id", "step"]].copy()
    grouped = df_helmets.groupby(["game_play", "nfl_player_id"])

    for lag in lag_range:
        shift_val = -lag
        suffix = f"_lag_{lag}"
        for col in vis_cols:
            out_df[f"{col}{suffix}"] = grouped[col].shift(shift_val)

    out_df = out_df.fillna(0.0)
    return out_df


def engineer_pair_features(df):
    """
    Computes relative features and clamps values.
    """
    lags = list(range(-config.WINDOW_PRE, config.WINDOW_POST + 1))

    for lag in lags:
        suffix = f"_lag_{lag}"

        # 1. Distance & Log Distance
        dx = df[f"x_position{suffix}_1"] - df[f"x_position{suffix}_2"]
        dy = df[f"y_position{suffix}_1"] - df[f"y_position{suffix}_2"]
        dist = np.sqrt(dx**2 + dy**2)

        df[f"distance{suffix}"] = dist
        df[f"distance_log1p{suffix}"] = np.log1p(dist)

        # 2. Relative Speed (Scalar Diff)
        df[f"relative_speed{suffix}"] = df[f"speed{suffix}_1"] - df[f"speed{suffix}_2"]

        # 3. Relative Angle (Shortest Arc)
        df[f"relative_angle{suffix}"] = utils.calculate_shortest_arc(
            df[f"orientation{suffix}_1"], df[f"orientation{suffix}_2"]
        )

        # 4. Clamping
        clamp_map = {
            "speed": config.CLAMP_RANGES["speed"],
            "acceleration": config.CLAMP_RANGES["acceleration"],
            "sa": config.CLAMP_RANGES["sa"],
            "relative_speed": config.CLAMP_RANGES["relative_speed"],
            "distance": config.CLAMP_RANGES["distance"],
            "distance_log1p": config.CLAMP_RANGES["distance_log1p"],
            "relative_angle": config.CLAMP_RANGES["relative_angle"],
            "orientation": config.CLAMP_RANGES["orientation"],
            "direction": config.CLAMP_RANGES["direction"],
            "left": config.CLAMP_RANGES["left"],
            "top": config.CLAMP_RANGES["top"],
            "width": config.CLAMP_RANGES["width"],
            "height": config.CLAMP_RANGES["height"],
            "area": config.CLAMP_RANGES["area"],
        }

        for base_feat, (min_v, max_v) in clamp_map.items():
            # Check and clamp all variations
            for entity in ["_1", "_2", ""]:
                if entity == "":
                    col = f"{base_feat}{suffix}"
                else:
                    col = f"{base_feat}{suffix}{entity}"

                if col in df.columns:
                    df[col] = df[col].clip(min_v, max_v)

    return df


def build_features(metadata_df, tracking_df, helmets_df):
    """
    Merges metadata with processed tracking and visual streams.
    """
    game_plays = metadata_df["game_play"].unique()

    # Process streams
    track_windows = process_tracking_stream(tracking_df, game_plays)
    vis_windows = process_visual_stream(helmets_df, game_plays)

    # Ensure ID types
    metadata_df["nfl_player_id_1"] = (
        pd.to_numeric(metadata_df["nfl_player_id_1"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    # Merge P1
    merged = metadata_df.merge(
        track_windows.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
        suffixes=(None, "_dup"),
    ).drop(columns=["game_play_1", "step_1", "nfl_player_id_1_dup"], errors="ignore")

    merged = merged.merge(
        vis_windows.add_suffix("_1"),
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play_1", "step_1", "nfl_player_id_1"],
        how="left",
        suffixes=(None, "_dup"),
    ).drop(columns=["game_play_1", "step_1", "nfl_player_id_1_dup"], errors="ignore")

    # Split Player vs Ground
    mask_ground = merged["nfl_player_id_2"] == "G"
    df_ground = merged[mask_ground].copy()
    df_player = merged[~mask_ground].copy()

    # --- Process Player-Player ---
    if not df_player.empty:
        df_player["nfl_player_id_2"] = pd.to_numeric(
            df_player["nfl_player_id_2"]
        ).astype(int)
        df_player = df_player.merge(
            track_windows.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
            suffixes=(None, "_dup"),
        ).drop(
            columns=["game_play_2", "step_2", "nfl_player_id_2_dup"], errors="ignore"
        )

        df_player = df_player.merge(
            vis_windows.add_suffix("_2"),
            left_on=["game_play", "step", "nfl_player_id_2"],
            right_on=["game_play_2", "step_2", "nfl_player_id_2"],
            how="left",
            suffixes=(None, "_dup"),
        ).drop(
            columns=["game_play_2", "step_2", "nfl_player_id_2_dup"], errors="ignore"
        )
        df_player["is_ground"] = 0

    # --- Process Player-Ground ---
    if not df_ground.empty:
        lags = list(range(-config.WINDOW_PRE, config.WINDOW_POST + 1))
        for lag in lags:
            suffix = f"_lag_{lag}"
            # Impute P2 Tracking from P1
            df_ground[f"x_position{suffix}_2"] = df_ground[f"x_position{suffix}_1"]
            df_ground[f"y_position{suffix}_2"] = df_ground[f"y_position{suffix}_1"]

            # Zero out kinematics and visuals
            for col in ["speed", "acceleration", "orientation", "direction", "sa"]:
                df_ground[f"{col}{suffix}_2"] = 0.0
            for col in ["left", "top", "width", "height", "area"]:
                df_ground[f"{col}{suffix}_2"] = 0.0

        df_ground["is_ground"] = 1

    # Concatenate and Engineer
    final_df = pd.concat([df_player, df_ground], axis=0).sort_index()
    final_df = engineer_pair_features(final_df)
    final_df = final_df.fillna(0.0)

    return final_df


def get_feature_columns(df_columns):
    """Identifies feature columns from dataframe columns."""
    lags = list(range(-config.WINDOW_PRE, config.WINDOW_POST + 1))
    feature_cols = []

    # Definitions
    kin_base = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]
    vis_base = ["left", "top", "width", "height", "area"]
    derived_base = ["distance", "distance_log1p", "relative_speed", "relative_angle"]

    for lag in lags:
        suffix = f"_lag_{lag}"
        # Add P1/P2 Kinematics
        feature_cols.extend([f"{c}{suffix}_1" for c in kin_base])
        feature_cols.extend([f"{c}{suffix}_2" for c in kin_base])
        # Add Derived
        feature_cols.extend([f"{c}{suffix}" for c in derived_base])
        # Add P1/P2 Visuals
        feature_cols.extend([f"{c}{suffix}_1" for c in vis_base])
        feature_cols.extend([f"{c}{suffix}_2" for c in vis_base])

    feature_cols.append("is_ground")

    # Filter to what actually exists in the DF (safety)
    valid_cols = [c for c in feature_cols if c in df_columns]
    return valid_cols


def prepare_datasets(load_cached_data=True, debug_sample=config.DEBUG_SAMPLE_SIZE):
    """
    Main function to prepare DataLoaders.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_features.parquet")
    val_cache = os.path.join(cache_dir, "val_features.parquet")
    test_cache = os.path.join(cache_dir, "test_features.parquet")

    # 1. Load or Compute
    if load_cached_data and os.path.exists(train_cache) and os.path.exists(val_cache):
        print("Loading cached features...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache) if os.path.exists(test_cache) else None
    else:
        print("Computing features from scratch...")
        meta_train = load_metadata("train")
        meta_val = load_metadata("validation")
        meta_test = load_metadata("test")

        if debug_sample is not None:
            meta_train = meta_train.head(debug_sample)
            meta_val = meta_val.head(debug_sample)

        raw_track_train = load_raw_tracking("train")
        raw_helm_train = load_raw_helmets("train")
        raw_track_test = load_raw_tracking("test")
        raw_helm_test = load_raw_helmets("test")

        print("Building Train features...")
        df_train = build_features(meta_train, raw_track_train, raw_helm_train)
        print("Building Validation features...")
        df_val = build_features(meta_val, raw_track_train, raw_helm_train)
        print("Building Test features...")
        df_test = build_features(meta_test, raw_track_test, raw_helm_test)

        print("Saving to cache...")
        df_train.to_parquet(train_cache)
        df_val.to_parquet(val_cache)
        df_test.to_parquet(test_cache)

        del raw_track_train, raw_helm_train, raw_track_test, raw_helm_test
        gc.collect()

    # 2. Scaling
    feature_cols = get_feature_columns(df_train.columns)
    print(f"Feature count: {len(feature_cols)}")

    scaler = StandardScaler()
    print("Fitting scaler...")
    X_train = scaler.fit_transform(df_train[feature_cols].astype(np.float32))
    y_train = df_train["contact"].values.astype(np.float32)

    X_val = scaler.transform(df_val[feature_cols].astype(np.float32))
    y_val = df_val["contact"].values.astype(np.float32)

    X_test = None
    if df_test is not None:
        X_test = scaler.transform(df_test[feature_cols].astype(np.float32))

    # 3. DataLoaders
    train_loader = DataLoader(
        NFLContactDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        NFLContactDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    test_loader = None
    if X_test is not None:
        test_loader = DataLoader(
            NFLContactDataset(X_test, None),
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=(config.DEVICE == "cuda"),
        )

    return train_loader, val_loader, test_loader, df_test
