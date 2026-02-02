import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import get_hash_filename


def save_scaler(scaler: RobustScaler, save_dir: str, prefix: str):
    """
    Saves the RobustScaler parameters (center and scale) to npy files.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{prefix}_center.npy"), scaler.center_)
    np.save(os.path.join(save_dir, f"{prefix}_scale.npy"), scaler.scale_)


def load_scaler(save_dir: str, prefix: str) -> RobustScaler:
    """
    Loads a RobustScaler from saved npy files.
    """
    center_path = os.path.join(save_dir, f"{prefix}_center.npy")
    scale_path = os.path.join(save_dir, f"{prefix}_scale.npy")

    if not os.path.exists(center_path) or not os.path.exists(scale_path):
        raise FileNotFoundError(
            f"Scaler files not found with prefix {prefix} in {save_dir}"
        )

    scaler = RobustScaler()
    scaler.center_ = np.load(center_path)
    scaler.scale_ = np.load(scale_path)
    return scaler


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds physics-inspired features: dt, area, volume, and interaction terms.
    Uses vectorized operations for efficiency.
    """
    # Ensure data is sorted
    df = df.sort_values(by=[Config.BREATH_ID, Config.TIME_STEP]).reset_index(drop=True)

    # 1. Time Delta (dt)
    # Calculate difference, then mask out the first time step of each breath
    df[Config.DT] = df[Config.TIME_STEP].diff()
    # Identify start of new breaths
    breath_change_mask = df[Config.BREATH_ID] != df[Config.BREATH_ID].shift(1)
    df.loc[breath_change_mask, Config.DT] = 0.0
    # Fill any remaining NaNs (e.g., very first row)
    df[Config.DT] = df[Config.DT].fillna(0.0)

    # 2. Area (Flow * dt) -> Volume (Integral of Area)
    df[Config.AREA] = df[Config.U_IN] * df[Config.DT]
    # Groupby cumsum is reasonably fast
    df[Config.VOLUME] = df.groupby(Config.BREATH_ID)[Config.AREA].cumsum()

    # 3. Interaction Terms
    df[Config.R_U_IN] = df[Config.U_IN] * df[Config.R]
    df[Config.VOL_DIV_C] = df[Config.VOLUME] / df[Config.C]

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag and difference features for u_in.
    """
    # Identify start of new breaths to prevent leaking data across breaths
    breath_change_mask = df[Config.BREATH_ID] != df[Config.BREATH_ID].shift(1)

    # Lags 1-4
    for lag in range(1, 5):
        col_name = f"u_in_lag{lag}"
        df[col_name] = df[Config.U_IN].shift(lag).fillna(0.0)
        # Mask out values that shifted from previous breath
        # We need to mask the first 'lag' rows of each breath
        # A simpler vectorized approach:
        # If breath_id[t] != breath_id[t-lag], then it's invalid.
        # This check is expensive to do for every row.
        # Since seq_len is fixed at 80, we can rely on the structure or just use groupby shift for safety.
        # Given the constraints, groupby shift is safer and acceptable.
        df[col_name] = df.groupby(Config.BREATH_ID)[Config.U_IN].shift(lag).fillna(0.0)

    # Diffs
    # diff1: u_in(t) - u_in(t-1)
    # diff2: diff1(t) - diff1(t-1)
    # We can use the already computed lag1
    df[Config.U_IN_DIFF1] = df[Config.U_IN] - df[Config.U_IN_LAG1]
    df[Config.U_IN_DIFF2] = df[Config.U_IN_DIFF1] - df.groupby(Config.BREATH_ID)[
        Config.U_IN_DIFF1
    ].shift(1).fillna(0.0)

    return df


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the feature engineering pipeline.
    """
    df = add_physics_features(df)
    df = add_lag_features(df)
    return df


def reshape_to_sequences(df: pd.DataFrame, columns: list) -> np.ndarray:
    """
    Reshapes a flat DataFrame (N_rows, N_cols) into (N_breaths, 80, N_cols).
    Assumes dataframe is sorted and length is multiple of 80.
    """
    values = df[columns].values
    num_rows = values.shape[0]
    num_features = values.shape[1]

    if num_rows % Config.SEQ_LEN != 0:
        raise ValueError(
            f"Total rows {num_rows} is not divisible by sequence length {Config.SEQ_LEN}"
        )

    num_breaths = num_rows // Config.SEQ_LEN
    return values.reshape(num_breaths, Config.SEQ_LEN, num_features)


def prepare_train_data(load_cached_data: bool = True):
    """
    Loads, processes, scales, and caches the training data.

    Returns:
        train_x (np.ndarray): (N, 80, F)
        train_y (np.ndarray): (N, 80)
        train_uout (np.ndarray): (N, 80)
        train_ids (np.ndarray): (N, 80)
        scaler (RobustScaler): Fitted scaler
    """
    # config dict for hashing
    conf = {
        "features": Config.INPUT_FEATURES,
        "seq_len": Config.SEQ_LEN,
        "split": "train",
    }

    # Generate cache filenames
    cache_x = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "train_x"))
    cache_y = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "train_y"))
    cache_uout = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "train_uout"))
    cache_ids = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "train_ids"))
    scaler_prefix = get_hash_filename(conf, "scaler").replace(".npy", "")

    # Check cache
    if load_cached_data and os.path.exists(cache_x) and os.path.exists(cache_y):
        print("Loading cached training data...")
        X = np.load(cache_x)
        y = np.load(cache_y)
        u_out = np.load(cache_uout)
        ids = np.load(cache_ids)
        scaler = load_scaler(Config.WORKING_DIR, scaler_prefix)
        return X, y, u_out, ids, scaler

    print("Processing training data from scratch...")
    df = pd.read_csv(Config.TRAIN_CSV)

    # Feature Engineering
    df = transform_dataframe(df)

    # Extract Targets and Aux
    y_flat = df[Config.PRESSURE].values
    u_out_flat = df[Config.U_OUT].values
    ids_flat = df[Config.ID].values

    # Scaling
    # We scale the input features. We do NOT scale the target pressure.
    # Fit scaler on flat data
    scaler = RobustScaler()
    X_flat = df[Config.INPUT_FEATURES].values
    X_scaled_flat = scaler.fit_transform(X_flat)

    # Reshape
    num_breaths = len(df) // Config.SEQ_LEN
    num_features = len(Config.INPUT_FEATURES)

    X = X_scaled_flat.reshape(num_breaths, Config.SEQ_LEN, num_features)
    y = y_flat.reshape(num_breaths, Config.SEQ_LEN)
    u_out = u_out_flat.reshape(num_breaths, Config.SEQ_LEN)
    ids = ids_flat.reshape(num_breaths, Config.SEQ_LEN)

    # Save Cache
    np.save(cache_x, X)
    np.save(cache_y, y)
    np.save(cache_uout, u_out)
    np.save(cache_ids, ids)
    save_scaler(scaler, Config.WORKING_DIR, scaler_prefix)

    return X, y, u_out, ids, scaler


def prepare_val_data(scaler: RobustScaler, load_cached_data: bool = True):
    """
    Loads, processes, scales (using provided scaler), and caches the validation data.
    """
    conf = {
        "features": Config.INPUT_FEATURES,
        "seq_len": Config.SEQ_LEN,
        "split": "val",
    }

    cache_x = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "val_x"))
    cache_y = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "val_y"))
    cache_uout = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "val_uout"))
    cache_ids = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "val_ids"))

    if load_cached_data and os.path.exists(cache_x) and os.path.exists(cache_y):
        print("Loading cached validation data...")
        X = np.load(cache_x)
        y = np.load(cache_y)
        u_out = np.load(cache_uout)
        ids = np.load(cache_ids)
        return X, y, u_out, ids

    print("Processing validation data from scratch...")
    df = pd.read_csv(Config.VAL_CSV)

    df = transform_dataframe(df)

    y_flat = df[Config.PRESSURE].values
    u_out_flat = df[Config.U_OUT].values
    ids_flat = df[Config.ID].values
    X_flat = df[Config.INPUT_FEATURES].values

    # Transform using provided scaler
    X_scaled_flat = scaler.transform(X_flat)

    num_breaths = len(df) // Config.SEQ_LEN
    num_features = len(Config.INPUT_FEATURES)

    X = X_scaled_flat.reshape(num_breaths, Config.SEQ_LEN, num_features)
    y = y_flat.reshape(num_breaths, Config.SEQ_LEN)
    u_out = u_out_flat.reshape(num_breaths, Config.SEQ_LEN)
    ids = ids_flat.reshape(num_breaths, Config.SEQ_LEN)

    np.save(cache_x, X)
    np.save(cache_y, y)
    np.save(cache_uout, u_out)
    np.save(cache_ids, ids)

    return X, y, u_out, ids


def prepare_test_data(scaler: RobustScaler, load_cached_data: bool = True):
    """
    Loads, processes, scales (using provided scaler), and caches the test data.
    Returns X, u_out, ids (no y).
    """
    conf = {
        "features": Config.INPUT_FEATURES,
        "seq_len": Config.SEQ_LEN,
        "split": "test",
    }

    cache_x = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "test_x"))
    cache_uout = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "test_uout"))
    cache_ids = os.path.join(Config.WORKING_DIR, get_hash_filename(conf, "test_ids"))

    if load_cached_data and os.path.exists(cache_x):
        print("Loading cached test data...")
        X = np.load(cache_x)
        u_out = np.load(cache_uout)
        ids = np.load(cache_ids)
        return X, u_out, ids

    print("Processing test data from scratch...")
    df = pd.read_csv(Config.TEST_CSV)

    df = transform_dataframe(df)

    u_out_flat = df[Config.U_OUT].values
    ids_flat = df[Config.ID].values
    X_flat = df[Config.INPUT_FEATURES].values

    X_scaled_flat = scaler.transform(X_flat)

    num_breaths = len(df) // Config.SEQ_LEN
    num_features = len(Config.INPUT_FEATURES)

    X = X_scaled_flat.reshape(num_breaths, Config.SEQ_LEN, num_features)
    u_out = u_out_flat.reshape(num_breaths, Config.SEQ_LEN)
    ids = ids_flat.reshape(num_breaths, Config.SEQ_LEN)

    np.save(cache_x, X)
    np.save(cache_uout, u_out)
    np.save(cache_ids, ids)

    return X, u_out, ids
