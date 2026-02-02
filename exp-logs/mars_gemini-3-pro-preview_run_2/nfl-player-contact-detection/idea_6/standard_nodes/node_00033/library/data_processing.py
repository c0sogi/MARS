import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from library.config import Config


def load_data():
    """
    Loads raw metadata and tracking data from CSV files.
    """
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load tracking data
    # We load the full files; pandas handles memory efficiently enough for this size
    train_tracking = pd.read_csv(Config.TRAIN_TRACKING_PATH)
    test_tracking = pd.read_csv(Config.TEST_TRACKING_PATH)

    return train_meta, val_meta, test_meta, train_tracking, test_tracking


def preprocess_tracking(df_tracking):
    """
    Cleans tracking data by selecting relevant columns and removing duplicates.
    """
    cols = ["game_play", "step", "nfl_player_id"] + Config.TRACKING_COLS

    # Filter columns
    df = df_tracking[cols].copy()

    # Drop duplicates to ensure unique mapping
    df = df.drop_duplicates(subset=["game_play", "step", "nfl_player_id"])

    return df


def compute_instantaneous_features(df):
    """
    Computes interaction features for a merged dataframe (P1 and P2 columns present).
    Handles Geometric Consistency for Ground contacts.
    """
    # 1. Geometric Consistency for Ground
    # If is_ground is 1, P2 features should match P1 to represent "contact with environment at self position"
    is_ground = df["is_ground"] == 1

    for col in Config.TRACKING_COLS:
        p1_vals = df.loc[is_ground, f"{col}_1"]
        df.loc[is_ground, f"{col}_2"] = p1_vals

    # 2. Distance Features
    dx = df["x_position_1"] - df["x_position_2"]
    dy = df["y_position_1"] - df["y_position_2"]
    dist = np.sqrt(dx**2 + dy**2)

    df["distance"] = dist
    df["log_distance"] = np.log1p(dist)

    # 3. Kinematic Features (Closing Speed)
    # Convert speed/direction to velocity components
    # Assumption: direction is in degrees, 0=N (Y+), 90=E (X+) standard NFL coords
    dir_1_rad = np.radians(df["direction_1"].fillna(0))
    vx_1 = df["speed_1"].fillna(0) * np.sin(dir_1_rad)
    vy_1 = df["speed_1"].fillna(0) * np.cos(dir_1_rad)

    dir_2_rad = np.radians(df["direction_2"].fillna(0))
    vx_2 = df["speed_2"].fillna(0) * np.sin(dir_2_rad)
    vy_2 = df["speed_2"].fillna(0) * np.cos(dir_2_rad)

    dvx = vx_1 - vx_2
    dvy = vy_1 - vy_2

    # Closing speed: projection of relative velocity onto relative position vector
    # closing_speed = -(v_rel . r_rel) / |r_rel|
    dot_prod = dx * dvx + dy * dvy

    # Clamped denominator to avoid division by zero at contact
    denom = np.maximum(dist, 1e-6)

    # Positive closing speed implies objects are moving closer
    df["closing_speed"] = -(dot_prod / denom)

    return df


def create_wide_features(df_meta, df_tracking):
    """
    Constructs the wide temporal feature set (t-5 to t+5).
    """
    # --- Merge Tracking Data ---

    # Merge Player 1
    df = df_meta.merge(
        df_tracking,
        left_on=["game_play", "step", "nfl_player_id_1"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
    )
    df = df.rename(columns={c: f"{c}_1" for c in Config.TRACKING_COLS})
    df = df.drop(columns=["nfl_player_id"], errors="ignore")

    # Merge Player 2
    # Handle 'G' for P2 merge by creating a temporary numeric ID column
    df["p2_merge_id"] = pd.to_numeric(df["nfl_player_id_2"], errors="coerce")

    df = df.merge(
        df_tracking,
        left_on=["game_play", "step", "p2_merge_id"],
        right_on=["game_play", "step", "nfl_player_id"],
        how="left",
        suffixes=("", "_2"),
    )
    df = df.rename(columns={c: f"{c}_2" for c in Config.TRACKING_COLS})
    df = df.drop(columns=["nfl_player_id", "p2_merge_id"], errors="ignore")

    # Create conditioning feature
    df["is_ground"] = (df["nfl_player_id_2"] == "G").astype(int)

    # --- Compute Instantaneous Features ---
    df = compute_instantaneous_features(df)

    # --- Vectorized Lag Construction ---

    # Define base features to shift
    base_feats = (
        [f"{c}_1" for c in Config.TRACKING_COLS]
        + [f"{c}_2" for c in Config.TRACKING_COLS]
        + ["distance", "log_distance", "closing_speed"]
    )

    # Fill NaNs (missing tracking) with 0 before shifting
    df[base_feats] = df[base_feats].fillna(0)

    # Sort to ensure temporal continuity for shifting
    df = df.sort_values(["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"])

    # Create unique group ID to validate shift boundaries
    df["group_id"] = (
        df["game_play"].astype(str)
        + "_"
        + df["nfl_player_id_1"].astype(str)
        + "_"
        + df["nfl_player_id_2"].astype(str)
    )

    shifted_features = []

    # Loop from -WINDOW to +WINDOW (e.g., -5 to 5)
    # lag k means accessing t-k.
    # pandas shift(k) moves data down k rows.
    # To get data from 5 steps back (t-5), we use shift(5).
    # To get data from 5 steps ahead (t+5), we use shift(-5).
    for lag in range(-Config.WINDOW_SIZE, Config.WINDOW_SIZE + 1):
        shift_amount = lag

        # Shift data
        shifted = df[base_feats].shift(shift_amount)

        # Validation Mask:
        # 1. Group ID must match (didn't shift into another play/pair)
        # 2. Step must be consistent (step[i] == step[i-k] + k)
        mask = (df["group_id"] == df["group_id"].shift(shift_amount)) & (
            df["step"] == df["step"].shift(shift_amount) + shift_amount
        )

        # Apply mask (zero out invalid shifts)
        shifted[~mask] = 0.0

        # Rename columns
        suffix = f"_lag_{lag}"
        shifted.columns = [c + suffix for c in shifted.columns]

        shifted_features.append(shifted)

    # Concatenate all wide features
    df_wide = pd.concat(shifted_features, axis=1)

    # Combine with metadata
    meta_cols = ["contact_id", "contact", "is_ground"]
    df_final = pd.concat([df[meta_cols], df_wide], axis=1)

    return df_final


def prepare_data(load_cached_data=True, debug=False, force_save=False):
    """
    Main pipeline function. Loads, processes, caches, and returns tensor data.
    """
    Config.setup_directories()

    # 1. Load Raw Data
    train_meta, val_meta, test_meta, train_tracking, test_tracking = load_data()

    if debug:
        print("Debug mode: Subsampling data...")
        train_meta = train_meta.sample(10000, random_state=Config.SEED)
        val_meta = val_meta.sample(2000, random_state=Config.SEED)
        test_meta = test_meta.sample(2000, random_state=Config.SEED)

    # Preprocess Tracking
    train_tracking = preprocess_tracking(train_tracking)
    test_tracking = preprocess_tracking(test_tracking)

    # 2. Process & Cache Train
    if load_cached_data and os.path.exists(Config.TRAIN_FEATURES_CACHE) and not debug:
        print("Loading cached train features...")
        df_train = pd.read_parquet(Config.TRAIN_FEATURES_CACHE)
    else:
        print("Processing train features...")
        df_train = create_wide_features(train_meta, train_tracking)
        if not debug or force_save:
            df_train.to_parquet(Config.TRAIN_FEATURES_CACHE)

    # 3. Process & Cache Validation
    if load_cached_data and os.path.exists(Config.VAL_FEATURES_CACHE) and not debug:
        print("Loading cached val features...")
        df_val = pd.read_parquet(Config.VAL_FEATURES_CACHE)
    else:
        print("Processing val features...")
        # Validation set uses train tracking data
        df_val = create_wide_features(val_meta, train_tracking)
        if not debug or force_save:
            df_val.to_parquet(Config.VAL_FEATURES_CACHE)

    # 4. Process & Cache Test
    if load_cached_data and os.path.exists(Config.TEST_FEATURES_CACHE) and not debug:
        print("Loading cached test features...")
        df_test = pd.read_parquet(Config.TEST_FEATURES_CACHE)
    else:
        print("Processing test features...")
        df_test = create_wide_features(test_meta, test_tracking)
        if not debug or force_save:
            df_test.to_parquet(Config.TEST_FEATURES_CACHE)

    # 5. Scaling
    feature_cols = [
        c for c in df_train.columns if c not in ["contact_id", "contact", "is_ground"]
    ]

    scaler = StandardScaler()

    # Check if scaler exists
    if load_cached_data and os.path.exists(Config.SCALER_PATH) and not debug:
        print("Loading cached scaler...")
        scaler = joblib.load(Config.SCALER_PATH)
    else:
        print("Fitting scaler on training data...")
        X_train_raw = df_train[feature_cols].values.astype(np.float32)
        scaler.fit(X_train_raw)
        if not debug or force_save:
            joblib.dump(scaler, Config.SCALER_PATH)

    # 6. Transform and Convert to Tensors

    def process_split(df, is_test=False):
        # Transform features
        X_raw = df[feature_cols].values.astype(np.float32)
        X_scaled = scaler.transform(X_raw)

        # Extract Condition
        condition = df["is_ground"].values.astype(np.float32).reshape(-1, 1)

        # Identify Center Features (lag_0)
        center_suffix = "_lag_0"
        center_indices = [
            i for i, c in enumerate(feature_cols) if c.endswith(center_suffix)
        ]

        # Convert to Tensors
        X_wide_t = torch.tensor(X_scaled, dtype=torch.float32)
        X_center_t = torch.tensor(X_scaled[:, center_indices], dtype=torch.float32)
        condition_t = torch.tensor(condition, dtype=torch.float32)

        # Inputs tuple
        inputs = (X_wide_t, X_center_t, condition_t)

        if is_test:
            # For test, return IDs
            targets = df["contact_id"].values
        else:
            # For train/val, return targets
            targets = torch.tensor(
                df["contact"].values.astype(np.float32), dtype=torch.float32
            )

        return inputs, targets

    print("Converting to tensors...")
    train_data = process_split(df_train)
    val_data = process_split(df_val)
    test_data = process_split(df_test, is_test=True)

    return train_data, val_data, test_data
