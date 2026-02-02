import os
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config


def get_config_hash():
    """
    Generates a unique hash based on the feature engineering configuration.
    """
    config_dict = {
        "USE_PHYSICS_INTERACTIONS": Config.USE_PHYSICS_INTERACTIONS,
        "USE_LAGS": Config.USE_LAGS,
        "LAG_STEPS": Config.LAG_STEPS,
        "USE_DIFFS": Config.USE_DIFFS,
        "CONTINUOUS_COLS": Config.CONTINUOUS_COLS,
        "SEED": Config.SEED,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def compute_features(df):
    """
    Computes physics-fidelity features and dynamics.
    Assumes df is sorted by breath_id and time_step.
    """
    # Ensure sorted
    df = df.sort_values(["breath_id", "time_step"])

    # 1. Time Delta
    # Calculate dt per breath.
    # We can shift globally and mask where breath_id changes.
    df["dt"] = df["time_step"].diff().fillna(0)
    # Mask invalid dt (first step of each breath should be 0 or small, but diff across breaths is wrong)
    # Since time_step starts at 0 for each breath, diff is correct except at boundary if we didn't group.
    # However, time_step is absolute in some datasets, but here usually relative.
    # Let's verify boundary:
    mask_new_breath = df["breath_id"] != df["breath_id"].shift(1)
    df.loc[mask_new_breath, "dt"] = 0

    # 2. Volume Integration (Physics)
    # volume = cumsum(u_in * dt)
    # We use groupby for cumsum as it's harder to vectorize purely with shift/mask without leakage
    # Optimization: Since data is sorted, we can use simple cumsum and reset at boundaries?
    # Actually, groupby transform is reasonably fast for cumsum.
    df["volume"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()

    # 3. Physics Interactions
    if Config.USE_PHYSICS_INTERACTIONS:
        df["R_u_in"] = df["R"] * df["u_in"]
        # Avoid division by zero if C is 0 (unlikely in this dataset, but safe practice)
        df["vol_C"] = df["volume"] / df["C"]

    # 4. Dynamics (Lags and Diffs)
    # We can use global shifts and mask out boundaries
    if Config.USE_LAGS:
        for lag in range(1, Config.LAG_STEPS + 1):
            col_name = f"u_in_lag{lag}"
            df[col_name] = df["u_in"].shift(lag).fillna(0)
            # Mask rows where the lag crosses breath boundary
            # condition: breath_id[i] != breath_id[i-lag]
            mask_boundary = df["breath_id"] != df["breath_id"].shift(lag)
            df.loc[mask_boundary, col_name] = 0

    if Config.USE_DIFFS:
        # 1st Diff
        df["u_in_diff1"] = df["u_in"].diff().fillna(0)
        df.loc[mask_new_breath, "u_in_diff1"] = 0

        # 2nd Diff
        df["u_in_diff2"] = df["u_in_diff1"].diff().fillna(0)
        df.loc[mask_new_breath, "u_in_diff2"] = 0

    return df


def reshape_sequences(df, feature_cols, target_col=None, is_test=False):
    """
    Reshapes the dataframe into (N_breaths, 80, N_features).
    """
    # Verify sequence length
    seq_len = 80
    num_breaths = len(df) // seq_len

    if len(df) % seq_len != 0:
        raise ValueError(
            f"Total rows {len(df)} is not divisible by sequence length {seq_len}"
        )

    # Extract arrays
    # Features
    X = df[feature_cols].values.astype(np.float32)
    X = X.reshape(num_breaths, seq_len, -1)

    # u_out (Control Segregation)
    u_out = df["u_out"].values.astype(np.float32)
    u_out = u_out.reshape(num_breaths, seq_len)

    outputs = {"X": X, "u_out": u_out}

    if not is_test and target_col:
        y = df[target_col].values.astype(np.float32)
        y = y.reshape(num_breaths, seq_len)
        outputs["y"] = y

    if is_test:
        ids = df["id"].values.astype(np.int32)
        # Don't reshape IDs to (N, 80) if we want to flatten later easily,
        # but for consistency with X, let's reshape.
        ids = ids.reshape(num_breaths, seq_len)
        outputs["ids"] = ids

    return outputs


def prepare_data(load_cached_data=True):
    """
    Main function to load, process, and return data.
    Implements caching and segregates processing logic.
    """
    # 1. Setup Cache Paths
    config_hash = get_config_hash()
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train_X": os.path.join(cache_dir, f"train_X_{config_hash}.npy"),
        "train_y": os.path.join(cache_dir, f"train_y_{config_hash}.npy"),
        "train_uout": os.path.join(cache_dir, f"train_uout_{config_hash}.npy"),
        "val_X": os.path.join(cache_dir, f"val_X_{config_hash}.npy"),
        "val_y": os.path.join(cache_dir, f"val_y_{config_hash}.npy"),
        "val_uout": os.path.join(cache_dir, f"val_uout_{config_hash}.npy"),
        "test_X": os.path.join(cache_dir, f"test_X_{config_hash}.npy"),
        "test_ids": os.path.join(cache_dir, f"test_ids_{config_hash}.npy"),
        "test_uout": os.path.join(cache_dir, f"test_uout_{config_hash}.npy"),
    }

    # 2. Check Cache
    all_exist = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_exist:
        print(f"Loading cached features from {cache_dir} (Hash: {config_hash})")
        train_data = (
            np.load(files["train_X"]),
            np.load(files["train_y"]),
            np.load(files["train_uout"]),
        )
        val_data = (
            np.load(files["val_X"]),
            np.load(files["val_y"]),
            np.load(files["val_uout"]),
        )
        test_data = (
            np.load(files["test_X"]),
            np.load(files["test_ids"]),
            np.load(files["test_uout"]),
        )
        return train_data, val_data, test_data

    # 3. Compute from Scratch
    print("Computing features from scratch...")

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Apply Feature Engineering
    print("Feature Engineering: Train...")
    train_df = compute_features(train_df)
    print("Feature Engineering: Val...")
    val_df = compute_features(val_df)
    print("Feature Engineering: Test...")
    test_df = compute_features(test_df)

    # Identify Feature Columns
    # Base continuous cols + derived
    feature_cols = [c for c in Config.CONTINUOUS_COLS if c in train_df.columns]

    # Add derived features
    if "volume" in train_df.columns:
        feature_cols.append("volume")
    if Config.USE_PHYSICS_INTERACTIONS:
        feature_cols.extend(["R_u_in", "vol_C"])
    if Config.USE_LAGS:
        feature_cols.extend([f"u_in_lag{i}" for i in range(1, Config.LAG_STEPS + 1)])
    if Config.USE_DIFFS:
        feature_cols.extend(["u_in_diff1", "u_in_diff2"])

    # Ensure u_out is NOT in feature_cols (it is handled separately)
    feature_cols = [c for c in feature_cols if c != "u_out"]

    print(f"Selected Features ({len(feature_cols)}): {feature_cols}")

    # Scaling
    # Fit RobustScaler ONLY on Train
    print("Fitting Scaler...")
    scaler = RobustScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

    # Transform Val and Test
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Reshape
    print("Reshaping Data...")
    train_out = reshape_sequences(train_df, feature_cols, target_col="pressure")
    val_out = reshape_sequences(val_df, feature_cols, target_col="pressure")
    test_out = reshape_sequences(test_df, feature_cols, is_test=True)

    # Save to Cache
    print("Saving to Cache...")
    np.save(files["train_X"], train_out["X"])
    np.save(files["train_y"], train_out["y"])
    np.save(files["train_uout"], train_out["u_out"])

    np.save(files["val_X"], val_out["X"])
    np.save(files["val_y"], val_out["y"])
    np.save(files["val_uout"], val_out["u_out"])

    np.save(files["test_X"], test_out["X"])
    np.save(files["test_ids"], test_out["ids"])
    np.save(files["test_uout"], test_out["u_out"])

    return (
        (train_out["X"], train_out["y"], train_out["u_out"]),
        (val_out["X"], val_out["y"], val_out["u_out"]),
        (test_out["X"], test_out["ids"], test_out["u_out"]),
    )
