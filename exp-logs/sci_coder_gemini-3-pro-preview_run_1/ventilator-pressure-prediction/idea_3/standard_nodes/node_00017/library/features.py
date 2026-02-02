import pandas as pd
import numpy as np
import os
import hashlib
import json
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


def get_feature_hash():
    """
    Generates a unique MD5 hash based on the current feature configuration.
    This ensures that if features or scaler settings change, the cache is invalidated.
    """
    config_dict = {
        "cont_features": sorted(Config.CONT_FEATURES),
        "cat_features": sorted(Config.CAT_FEATURES),
        "seq_len": Config.SEQ_LEN,
        "scaler": "RobustScaler",
        "target": Config.TARGET,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def add_physics_features(df):
    """
    Adds equation-driven physics features based on the Equation of Motion.
    """
    # Calculate time delta (dt)
    # Group by breath_id to ensure diff doesn't cross breaths
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # Volume approximation: u_in * dt
    # u_in is a control valve setting (0-100), acting as a proxy for flow rate
    df["volume"] = df["u_in"] * df["dt"]

    # Cumulative Volume (Area)
    df["area"] = df.groupby("breath_id")["volume"].cumsum()

    # Cumulative Sum of u_in (alternative volume proxy without time scaling)
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # Theoretical Pressure Approximation
    # Equation of Motion: P(t) = Volume(t)/C + Flow(t)*R
    df["pressure_approx"] = (df["area"] / df["C"]) + (df["u_in"] * df["R"])

    # Interaction Features
    df["u_in_R"] = df["u_in"] * df["R"]
    df["u_in_C"] = df["u_in"] * df["C"]

    return df


def add_lag_diff_features(df):
    """
    Adds lag and difference features to capture temporal dynamics.
    """
    # Lags for u_in and u_out
    for lag in [1, 2]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)
        df[f"u_out_lag{lag}"] = df.groupby("breath_id")["u_out"].shift(lag).fillna(0)

    # First difference of u_in
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]

    # Second difference of u_in (acceleration)
    # diff2 = (u_in[t] - u_in[t-1]) - (u_in[t-1] - u_in[t-2])
    # We can approximate this by diffing the diff1 column
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    return df


def engineer_features(df):
    """
    Orchestrates the feature engineering process.
    """
    # Ensure data is sorted correctly before time-series operations
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    df = add_physics_features(df)
    df = add_lag_diff_features(df)

    # Preserve raw u_out for masking (before it potentially gets scaled)
    df["u_out_raw"] = df["u_out"]

    return df


def prepare_datasets(load_cached_data=True):
    """
    Main entry point to prepare Train, Validation, and Test datasets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
                                 If False or cache missing, processes from scratch.

    Returns:
        tuple: (train_data, val_data, test_data)
               Each is a dictionary containing 'x', 'y' (except test), and 'u_out'.
    """
    seed_everything(Config.SEED)

    # 1. Setup Cache Paths
    feature_hash = get_feature_hash()
    cache_dir = Config.OUTPUT_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_x": f"train_x_{feature_hash}.npy",
        "train_y": f"train_y_{feature_hash}.npy",
        "train_uout": f"train_uout_{feature_hash}.npy",
        "val_x": f"val_x_{feature_hash}.npy",
        "val_y": f"val_y_{feature_hash}.npy",
        "val_uout": f"val_uout_{feature_hash}.npy",
        "test_x": f"test_x_{feature_hash}.npy",
        "test_ids": f"test_ids_{feature_hash}.npy",
        "test_uout": f"test_uout_{feature_hash}.npy",
    }

    # 2. Check Cache
    cache_exists = all(
        os.path.exists(os.path.join(cache_dir, f)) for f in files.values()
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached datasets with hash {feature_hash}...")
        data = {}
        for key, fname in files.items():
            data[key] = np.load(os.path.join(cache_dir, fname))

        return (
            {"x": data["train_x"], "y": data["train_y"], "u_out": data["train_uout"]},
            {"x": data["val_x"], "y": data["val_y"], "u_out": data["val_uout"]},
            {"x": data["test_x"], "ids": data["test_ids"], "u_out": data["test_uout"]},
        )

    # 3. Process Data from Scratch
    print("Cache miss or reload requested. Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print("DEBUG Mode: Using small subset of data.")
        train_df = train_df.iloc[: 80 * 100]  # 100 breaths
        val_df = val_df.iloc[: 80 * 50]
        test_df = test_df.iloc[: 80 * 50]

    # Feature Engineering
    print("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Normalization (Continuous Features)
    print("Normalizing continuous features...")
    scaler = RobustScaler()

    # Fit on Train, Transform All
    train_df[Config.CONT_FEATURES] = scaler.fit_transform(
        train_df[Config.CONT_FEATURES]
    )
    val_df[Config.CONT_FEATURES] = scaler.transform(val_df[Config.CONT_FEATURES])
    test_df[Config.CONT_FEATURES] = scaler.transform(test_df[Config.CONT_FEATURES])

    # Categorical Encoding (R and C)
    # Map physical values to 0-based indices for Embedding layers
    r_map = {5: 0, 20: 1, 50: 2}
    c_map = {10: 0, 20: 1, 50: 2}

    for df in [train_df, val_df, test_df]:
        df["R_cat"] = df["R"].map(r_map)
        df["C_cat"] = df["C"].map(c_map)

    # Reshaping Logic
    feature_cols = Config.CONT_FEATURES + ["R_cat", "C_cat"]

    def reshape_dataset(df, is_test=False):
        # Ensure sorting is maintained
        df = df.sort_values(["breath_id", "time_step"])

        num_breaths = df["breath_id"].nunique()
        seq_len = Config.SEQ_LEN

        # Extract X (Features)
        # Shape: (N, 80, Num_Features)
        x = df[feature_cols].values.astype(np.float32)
        x = x.reshape(num_breaths, seq_len, len(feature_cols))

        # Extract u_out (Mask) - Use raw unscaled version
        u_out = df["u_out_raw"].values.astype(np.float32)
        u_out = u_out.reshape(num_breaths, seq_len)

        if is_test:
            ids = df["id"].values.astype(np.int32)  # Flattened IDs for submission
            return x, ids, u_out
        else:
            y = df[Config.TARGET].values.astype(np.float32)
            y = y.reshape(num_breaths, seq_len)
            return x, y, u_out

    print("Reshaping datasets...")
    train_x, train_y, train_uout = reshape_dataset(train_df)
    val_x, val_y, val_uout = reshape_dataset(val_df)
    test_x, test_ids, test_uout = reshape_dataset(test_df, is_test=True)

    # Save to Cache
    print(f"Saving processed datasets to {cache_dir}...")
    np.save(os.path.join(cache_dir, files["train_x"]), train_x)
    np.save(os.path.join(cache_dir, files["train_y"]), train_y)
    np.save(os.path.join(cache_dir, files["train_uout"]), train_uout)

    np.save(os.path.join(cache_dir, files["val_x"]), val_x)
    np.save(os.path.join(cache_dir, files["val_y"]), val_y)
    np.save(os.path.join(cache_dir, files["val_uout"]), val_uout)

    np.save(os.path.join(cache_dir, files["test_x"]), test_x)
    np.save(os.path.join(cache_dir, files["test_ids"]), test_ids)
    np.save(os.path.join(cache_dir, files["test_uout"]), test_uout)

    return (
        {"x": train_x, "y": train_y, "u_out": train_uout},
        {"x": val_x, "y": val_y, "u_out": val_uout},
        {"x": test_x, "ids": test_ids, "u_out": test_uout},
    )
