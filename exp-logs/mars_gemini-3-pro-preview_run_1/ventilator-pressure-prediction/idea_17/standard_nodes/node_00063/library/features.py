import os
import hashlib
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from library.config import Config


def get_config_hash():
    """
    Generates a hash based on the feature engineering configuration.
    This ensures that if feature parameters change, the cache is invalidated.
    """
    config_dict = {
        "n_lags": Config.N_LAGS,
        "use_differences": Config.USE_DIFFERENCES,
        "physics_features": Config.PHYSICS_FEATURES,
        "experiment": Config.EXPERIMENT_NAME,
        "scaler": "RobustScaler",
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def add_physics_features(df):
    """
    Adds physics-based features, lags, and differences to the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing raw ventilator data.

    Returns:
        pd.DataFrame: Dataframe with added features.
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Calculate dt (time delta)
    # We group by breath_id to ensure the first step of a new breath doesn't diff with previous breath
    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0)

    # --- Physics Integration ---
    if Config.PHYSICS_FEATURES:
        # Volume = Integral of flow (u_in) over time
        # Note: u_in is 0-100, we treat it as flow rate
        df["volume"] = (
            df.groupby("breath_id")
            .apply(lambda x: (x["u_in"] * x["dt"]).cumsum())
            .reset_index(level=0, drop=True)
        )

        # Interaction Terms (Soft Physics)
        df["u_in_R"] = df["u_in"] * df["R"]
        df["vol_C"] = df["volume"] / df["C"]

        # Additional explicit physics interactions
        df["R_C"] = df["R"] * df["C"]

    # --- Time Series Dynamics ---
    # Lags
    for lag in range(1, Config.N_LAGS + 1):
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # Differences (Derivatives)
    if Config.USE_DIFFERENCES:
        df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)
        df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

        # Derivative of pressure is not available for test, so we don't compute it here.
        # We focus on input dynamics.

    # Fill any remaining NaNs resulting from shifts/diffs
    df = df.fillna(0)

    return df


def reshape_to_sequences(df, feature_cols, target_col=None):
    """
    Reshapes the tabular dataframe into (N_breaths, 80, N_features).

    Args:
        df (pd.DataFrame): Tabular data.
        feature_cols (list): List of feature column names.
        target_col (str, optional): Name of target column.

    Returns:
        tuple: (X, y, u_out, ids)
            X: (N, 80, F)
            y: (N, 80) or None
            u_out: (N, 80)
            ids: (N, 80)
    """
    # Ventilator data has exactly 80 steps per breath
    n_steps = 80
    n_breaths = len(df) // n_steps

    # Ensure strict ordering
    if len(df) % n_steps != 0:
        raise ValueError(
            f"Data length {len(df)} is not divisible by {n_steps} steps per breath."
        )

    # Extract arrays
    X = df[feature_cols].values.reshape(n_breaths, n_steps, len(feature_cols))
    u_out = df["u_out"].values.reshape(n_breaths, n_steps)
    ids = df["id"].values.reshape(n_breaths, n_steps)

    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].values.reshape(n_breaths, n_steps)

    return X, y, u_out, ids


def prepare_datasets(load_cached_data=True):
    """
    Main entry point to load, process, scale, and reshape data.
    Handles caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing:
            'train_X', 'train_y', 'train_uout',
            'val_X', 'val_y', 'val_uout',
            'test_X', 'test_uout', 'test_ids',
            'feature_names'
    """
    config_hash = get_config_hash()
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(cache_dir, f"train_X_{config_hash}.npy"),
        "train_y": os.path.join(cache_dir, f"train_y_{config_hash}.npy"),
        "train_uout": os.path.join(cache_dir, f"train_uout_{config_hash}.npy"),
        "val_X": os.path.join(cache_dir, f"val_X_{config_hash}.npy"),
        "val_y": os.path.join(cache_dir, f"val_y_{config_hash}.npy"),
        "val_uout": os.path.join(cache_dir, f"val_uout_{config_hash}.npy"),
        "test_X": os.path.join(cache_dir, f"test_X_{config_hash}.npy"),
        "test_uout": os.path.join(cache_dir, f"test_uout_{config_hash}.npy"),
        "test_ids": os.path.join(cache_dir, f"test_ids_{config_hash}.npy"),
        "meta": os.path.join(cache_dir, f"meta_{config_hash}.json"),
    }

    scaler_path = os.path.join(cache_dir, f"scaler_{config_hash}.joblib")

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print(f"Loading cached datasets from {cache_dir} (Hash: {config_hash})...")
        data = {}
        for k, v in cache_files.items():
            if k == "meta":
                with open(v, "r") as f:
                    data.update(json.load(f))
            else:
                data[k] = np.load(v)
        return data

    print("Cache not found or invalid. Processing data from scratch...")

    # Load Metadata CSVs
    print("Loading raw CSVs...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print("DEBUG Mode: Subsampling data...")
        # Keep full breaths (80 steps)
        train_df = train_df.iloc[: 80 * 100]
        val_df = val_df.iloc[: 80 * 50]
        test_df = test_df.iloc[: 80 * 50]

    # Feature Engineering
    print("Applying Physics and Time-Series Feature Engineering...")
    train_df = add_physics_features(train_df)
    val_df = add_physics_features(val_df)
    test_df = add_physics_features(test_df)

    # Define Feature Columns
    # Exclude: id, breath_id, pressure, u_out (handled separately)
    # Include: R, C, time_step, u_in, dt, volume, u_in_R, vol_C, lags, diffs
    exclude_cols = ["id", "breath_id", "pressure", "u_out"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Selected Features ({len(feature_cols)}): {feature_cols}")

    # Scaling
    # We segregate u_out (binary) which is already excluded from feature_cols
    # We apply RobustScaler to all continuous features
    print("Fitting RobustScaler on Training Data...")
    scaler = RobustScaler()

    # Fit on Train, Transform All
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Save Scaler
    joblib.dump(scaler, scaler_path)

    # Reshape to Sequences
    print("Reshaping to 3D Sequences...")
    train_X, train_y, train_uout, _ = reshape_to_sequences(
        train_df, feature_cols, "pressure"
    )
    val_X, val_y, val_uout, _ = reshape_to_sequences(val_df, feature_cols, "pressure")
    test_X, _, test_uout, test_ids = reshape_to_sequences(test_df, feature_cols, None)

    # Save to Cache
    print("Saving processed data to cache...")
    np.save(cache_files["train_X"], train_X)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["train_uout"], train_uout)
    np.save(cache_files["val_X"], val_X)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["val_uout"], val_uout)
    np.save(cache_files["test_X"], test_X)
    np.save(cache_files["test_uout"], test_uout)
    np.save(cache_files["test_ids"], test_ids)

    meta_data = {"feature_names": feature_cols}
    with open(cache_files["meta"], "w") as f:
        json.dump(meta_data, f)

    data = {
        "train_X": train_X,
        "train_y": train_y,
        "train_uout": train_uout,
        "val_X": val_X,
        "val_y": val_y,
        "val_uout": val_uout,
        "test_X": test_X,
        "test_uout": test_uout,
        "test_ids": test_ids,
        "feature_names": feature_cols,
    }

    print("Data processing complete.")
    return data
