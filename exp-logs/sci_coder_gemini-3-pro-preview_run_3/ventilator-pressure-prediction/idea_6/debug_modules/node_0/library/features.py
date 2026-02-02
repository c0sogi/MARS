import pandas as pd
import numpy as np
import os
from library.config import Config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes PID and physics-based features for the ventilator dataset.

    Args:
        df: Input dataframe containing raw columns (id, breath_id, R, C, u_in, u_out, etc.)

    Returns:
        df: Dataframe with added feature columns.
    """
    # Ensure data is sorted by breath_id and time_step
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # --- PID Features ---
    # Integral (Volume proxy)
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # Derivative (Velocity/Acceleration proxies)
    # Using shift to compute differences within groups
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_diff1"] - (df["u_in_lag1"] - df["u_in_lag2"])

    # --- Physics Features ---
    # Resistive Pressure Component: Flow * Resistance
    df["R_u_in"] = df["R"] * df["u_in"]

    # Elastic Pressure Component: Volume / Compliance
    # Note: u_in_cumsum is a proxy for volume
    df["vol_C"] = df["u_in_cumsum"] / df["C"]

    # Cleanup temporary columns
    df = df.drop(columns=["u_in_lag1", "u_in_lag2"])

    return df


def normalize_features(train_df, val_df, test_df, features):
    """
    Normalizes continuous features using Mean/Std calculated on Training set.
    Skips binary features like 'u_out'.
    """
    # Identify continuous features to normalize
    # We skip u_out as it is binary (0/1)
    cont_features = [f for f in features if f != "u_out"]

    # Compute stats on training set
    train_stats = train_df[cont_features].agg(["mean", "std"]).to_dict()

    for col in cont_features:
        mean = train_stats[col]["mean"]
        std = train_stats[col]["std"]

        # Avoid division by zero (unlikely here but good practice)
        if std == 0:
            std = 1.0

        train_df[col] = (train_df[col] - mean) / std
        val_df[col] = (val_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std

    return train_df, val_df, test_df


def reshape_to_sequence(df, features, target_col=None, seq_len=80):
    """
    Reshapes tabular data (N_samples, N_features) to (N_breaths, Seq_Len, N_features).
    """
    num_breaths = len(df) // seq_len

    # Extract features matrix
    x_data = df[features].values.astype(np.float32)
    # Reshape: (N_breaths, Seq_Len, Features)
    x_reshaped = x_data.reshape(num_breaths, seq_len, len(features))

    # Extract u_out separately for masking
    u_out_data = df["u_out"].values.astype(np.float32)
    u_out_reshaped = u_out_data.reshape(num_breaths, seq_len)

    y_reshaped = None
    if target_col and target_col in df.columns:
        y_data = df[target_col].values.astype(np.float32)
        y_reshaped = y_data.reshape(num_breaths, seq_len)

    return x_reshaped, u_out_reshaped, y_reshaped


def prepare_datasets(load_cached_data: bool = True):
    """
    Main function to load, process, cache, and return datasets.

    Args:
        load_cached_data: If True, attempts to load pre-processed .npy files.

    Returns:
        Dictionary containing:
        train_x, train_y, train_u_out
        val_x, val_y, val_u_out
        test_x, test_u_out
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_x": os.path.join(cache_dir, "train_x.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "train_u_out": os.path.join(cache_dir, "train_u_out.npy"),
        "val_x": os.path.join(cache_dir, "val_x.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "val_u_out": os.path.join(cache_dir, "val_u_out.npy"),
        "test_x": os.path.join(cache_dir, "test_x.npy"),
        "test_u_out": os.path.join(cache_dir, "test_u_out.npy"),
    }

    # Check if all files exist
    all_cached = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_cached:
        print(f"Loading cached datasets from {cache_dir}...")
        data = {k: np.load(v) for k, v in files.items()}
        return data

    print("Processing datasets from scratch...")

    # 1. Load Raw Data
    print(f"Loading CSVs from {Config.INPUT_DIR}...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Feature Engineering
    print("Generating physics and PID features...")
    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # 3. Normalization
    print("Normalizing features...")
    train_df, val_df, test_df = normalize_features(
        train_df, val_df, test_df, Config.FEATURES
    )

    # 4. Reshaping
    print("Reshaping to sequences...")
    train_x, train_u_out, train_y = reshape_to_sequence(
        train_df, Config.FEATURES, Config.COL_TARGET, Config.SEQ_LEN
    )
    val_x, val_u_out, val_y = reshape_to_sequence(
        val_df, Config.FEATURES, Config.COL_TARGET, Config.SEQ_LEN
    )
    test_x, test_u_out, _ = reshape_to_sequence(
        test_df, Config.FEATURES, None, Config.SEQ_LEN
    )

    # 5. Saving to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["train_x"], train_x)
    np.save(files["train_y"], train_y)
    np.save(files["train_u_out"], train_u_out)

    np.save(files["val_x"], val_x)
    np.save(files["val_y"], val_y)
    np.save(files["val_u_out"], val_u_out)

    np.save(files["test_x"], test_x)
    np.save(files["test_u_out"], test_u_out)

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "train_u_out": train_u_out,
        "val_x": val_x,
        "val_y": val_y,
        "val_u_out": val_u_out,
        "test_x": test_x,
        "test_u_out": test_u_out,
    }

    print("Data processing complete.")
    return data
