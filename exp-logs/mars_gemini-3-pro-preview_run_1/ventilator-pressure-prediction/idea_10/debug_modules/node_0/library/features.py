import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import get_config_hash


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-based features including time deltas, volume integration,
    lags, differences, and interaction terms.

    Args:
        df (pd.DataFrame): Dataframe containing raw columns.

    Returns:
        pd.DataFrame: Dataframe with added feature columns.
    """
    # Ensure data is sorted by breath_id and time_step
    # (Metadata generation guarantees this, but we rely on it for vectorization)

    # 1. Calculate dt (time delta)
    # We can use vectorized operations faster than groupby by detecting breath boundaries
    # dt = current_time - prev_time. If breath_id changes, dt = 0.
    df["dt"] = df["time_step"].diff()
    # Mask out the first time step of each breath (where breath_id changes)
    # The first row of the entire DF is also NaN from diff(), fill with 0
    first_breath_mask = df["breath_id"] != df["breath_id"].shift(1)
    df.loc[first_breath_mask, "dt"] = 0.0
    df["dt"] = df["dt"].fillna(0.0)

    # 2. Calculate Volume (Integral of u_in * dt)
    # u_in is 0-100, dt is seconds.
    df["vol_incr"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby("breath_id")["vol_incr"].cumsum()

    # 3. Lag Features (1-4)
    # Using groupby().shift() handles the boundaries correctly
    for lag in [1, 2, 3, 4]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0.0)

    # 4. Difference Features
    # diff1 = u_in[t] - u_in[t-1]
    # diff2 = u_in[t] - u_in[t-2] (captures wider trend/acceleration proxy)
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff(1).fillna(0.0)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in"].diff(2).fillna(0.0)

    # 5. Interaction Terms (Physics Injection)
    # R * u_in: Pressure drop is proportional to Flow * Resistance
    df["R_u_in"] = df["R"] * df["u_in"]

    # Volume / C: Pressure is proportional to Volume / Compliance
    # Add epsilon to C to avoid potential division by zero (though C is 10, 20, 50)
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    return df


def prepare_datasets(load_cached_data: bool = True):
    """
    Main entry point for data loading and preprocessing.
    Handles caching, feature engineering, scaling, and reshaping.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        Tuple containing:
        train_x, train_y, val_x, val_y, test_x, test_ids, scaler_center, scaler_scale
    """
    config_hash = get_config_hash(Config)
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_x": os.path.join(cache_dir, f"train_x_{config_hash}.npy"),
        "train_y": os.path.join(cache_dir, f"train_y_{config_hash}.npy"),
        "val_x": os.path.join(cache_dir, f"val_x_{config_hash}.npy"),
        "val_y": os.path.join(cache_dir, f"val_y_{config_hash}.npy"),
        "test_x": os.path.join(cache_dir, f"test_x_{config_hash}.npy"),
        "test_ids": os.path.join(cache_dir, f"test_ids_{config_hash}.npy"),
        "scaler_center": os.path.join(cache_dir, f"scaler_center_{config_hash}.npy"),
        "scaler_scale": os.path.join(cache_dir, f"scaler_scale_{config_hash}.npy"),
    }

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(p) for p in files.values()):
        print(f"Loading cached dataset with hash {config_hash}...")
        return (
            np.load(files["train_x"]),
            np.load(files["train_y"]),
            np.load(files["val_x"]),
            np.load(files["val_y"]),
            np.load(files["test_x"]),
            np.load(files["test_ids"]),
            np.load(files["scaler_center"]),
            np.load(files["scaler_scale"]),
        )

    print(
        f"Cache miss or force reload. Processing data from scratch (Hash: {config_hash})..."
    )

    # 1. Load Raw Data
    # Use float32 to save memory
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths...")
        train_breaths = train_df[Config.BREATH_ID_COL].unique()[
            : Config.DEBUG_SAMPLE_SIZE
        ]
        val_breaths = val_df[Config.BREATH_ID_COL].unique()[: Config.DEBUG_SAMPLE_SIZE]
        test_breaths = test_df[Config.BREATH_ID_COL].unique()[
            : Config.DEBUG_SAMPLE_SIZE
        ]

        train_df = train_df[train_df[Config.BREATH_ID_COL].isin(train_breaths)].copy()
        val_df = val_df[val_df[Config.BREATH_ID_COL].isin(val_breaths)].copy()
        test_df = test_df[test_df[Config.BREATH_ID_COL].isin(test_breaths)].copy()

    # 2. Feature Engineering
    print("Generating physics features...")
    train_df = add_physics_features(train_df)
    val_df = add_physics_features(val_df)
    test_df = add_physics_features(test_df)

    # 3. Scaling
    # We use RobustScaler. We must fit on Train and transform Train, Val, Test.
    # We manually implement scaling to save params as .npy (avoiding pickle)
    print("Fitting RobustScaler...")
    feature_cols = Config.FEATURE_COLS

    # Extract training features for fitting
    X_train_raw = train_df[feature_cols].values

    scaler = RobustScaler()
    scaler.fit(X_train_raw)

    center = scaler.center_
    scale = scaler.scale_

    # Handle cases where scale is 0 (constant feature) to avoid NaN
    scale[scale == 0.0] = 1.0

    # Apply scaling manually: (X - center) / scale
    print("Applying scaling...")

    def apply_scaling(df, cols, center, scale):
        X = df[cols].values
        X_scaled = (X - center) / scale
        return X_scaled.astype(np.float32)

    train_x_flat = apply_scaling(train_df, feature_cols, center, scale)
    val_x_flat = apply_scaling(val_df, feature_cols, center, scale)
    test_x_flat = apply_scaling(test_df, feature_cols, center, scale)

    # 4. Reshaping to (N_breaths, SEQ_LEN, N_features)
    # We assume SEQ_LEN = 80 and data is sorted.
    seq_len = Config.SEQ_LEN
    num_features = len(feature_cols)

    print(f"Reshaping to (-1, {seq_len}, {num_features})...")

    train_x = train_x_flat.reshape(-1, seq_len, num_features)
    val_x = val_x_flat.reshape(-1, seq_len, num_features)
    test_x = test_x_flat.reshape(-1, seq_len, num_features)

    # 5. Extract Targets and IDs
    train_y = train_df[Config.TARGET_COL].values.reshape(-1, seq_len).astype(np.float32)
    val_y = val_df[Config.TARGET_COL].values.reshape(-1, seq_len).astype(np.float32)
    test_ids = test_df[Config.ID_COL].values.astype(
        np.int32
    )  # Keep IDs flat for submission mapping

    # 6. Save to Cache
    print("Saving to cache...")
    np.save(files["train_x"], train_x)
    np.save(files["train_y"], train_y)
    np.save(files["val_x"], val_x)
    np.save(files["val_y"], val_y)
    np.save(files["test_x"], test_x)
    np.save(files["test_ids"], test_ids)
    np.save(files["scaler_center"], center)
    np.save(files["scaler_scale"], scale)

    print("Data preparation complete.")

    return train_x, train_y, val_x, val_y, test_x, test_ids, center, scale
