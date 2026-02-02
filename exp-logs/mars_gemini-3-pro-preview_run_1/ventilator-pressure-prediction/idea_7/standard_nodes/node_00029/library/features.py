import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import seed_everything


def add_features(df):
    """
    Adds features to the dataframe based on Config.
    Calculates lags, diffs, cumulative sums, and physical interaction terms.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values(["breath_id", "time_step"])

    # --- Physics / Interaction Features ---
    # R * u_in (Flow-Resistive Pressure approximation)
    df["R_u_in"] = df["R"] * df["u_in"]

    # Cumulative u_in (Volume proxy)
    # Group by breath_id to ensure cumsum resets for each breath
    df["u_in_cum"] = df.groupby("breath_id")["u_in"].cumsum()

    # Volume / C (Elastic Pressure approximation)
    df["vol_C"] = df["u_in_cum"] / df["C"]

    # --- Lag Features ---
    if Config.USE_LAG_FEATURES:
        for lag in Config.LAG_STEPS:
            # Shift u_in by 'lag' steps within each breath group
            df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)

    # --- Diff Features ---
    if Config.USE_DIFF_FEATURES:
        for diff in Config.DIFF_STEPS:
            # Calculate difference of u_in within each breath group
            df[f"u_in_diff{diff}"] = (
                df.groupby("breath_id")["u_in"].diff(diff).fillna(0)
            )

    # Fill any remaining NaNs (e.g. from lags/diffs at start of breath) with 0
    df = df.fillna(0)

    return df


def get_data(load_cached_data=True):
    """
    Loads, processes, and returns the ventilator dataset.
    Implements caching to avoid re-processing.

    Returns:
        tuple: (train_data, val_data, test_data)
        Each element is a dictionary containing tensors (e.g., 'x', 'y', 'u_out').
    """
    seed_everything(Config.SEED)

    # Requirement: Ensure specific directory exists (as per task instructions)
    os.makedirs("./working/idea_7", exist_ok=True)

    # Define cache file paths using Config.CACHE_DIR
    cache_files = {
        "train_x": os.path.join(Config.CACHE_DIR, "train_x.npy"),
        "train_y": os.path.join(Config.CACHE_DIR, "train_y.npy"),
        "train_u_out": os.path.join(Config.CACHE_DIR, "train_u_out.npy"),
        "val_x": os.path.join(Config.CACHE_DIR, "val_x.npy"),
        "val_y": os.path.join(Config.CACHE_DIR, "val_y.npy"),
        "val_u_out": os.path.join(Config.CACHE_DIR, "val_u_out.npy"),
        "test_x": os.path.join(Config.CACHE_DIR, "test_x.npy"),
        "test_u_out": os.path.join(Config.CACHE_DIR, "test_u_out.npy"),
        "test_ids": os.path.join(Config.CACHE_DIR, "test_ids.npy"),
        "scaler_center": os.path.join(Config.CACHE_DIR, "scaler_center.npy"),
        "scaler_scale": os.path.join(Config.CACHE_DIR, "scaler_scale.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading cached data...")
        data = {}
        for k, v in cache_files.items():
            data[k] = np.load(v)

        # Convert numpy arrays to torch tensors
        train_data = {
            "x": torch.tensor(data["train_x"], dtype=torch.float32),
            "y": torch.tensor(data["train_y"], dtype=torch.float32),
            "u_out": torch.tensor(data["train_u_out"], dtype=torch.float32),
        }
        val_data = {
            "x": torch.tensor(data["val_x"], dtype=torch.float32),
            "y": torch.tensor(data["val_y"], dtype=torch.float32),
            "u_out": torch.tensor(data["val_u_out"], dtype=torch.float32),
        }
        test_data = {
            "x": torch.tensor(data["test_x"], dtype=torch.float32),
            "u_out": torch.tensor(data["test_u_out"], dtype=torch.float32),
            "ids": torch.tensor(data["test_ids"], dtype=torch.long),
        }
        return train_data, val_data, test_data

    print("Cache not found or reload requested. Processing data from scratch...")

    # Load raw metadata CSVs
    print(f"Loading train from {Config.TRAIN_CSV}")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print(f"Loading val from {Config.VAL_CSV}")
    val_df = pd.read_csv(Config.VAL_CSV)
    print(f"Loading test from {Config.TEST_CSV}")
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug mode: sample data to speed up development
    if Config.DEBUG:
        print("DEBUG mode enabled: Using small subset of data.")
        train_df = train_df.iloc[: Config.SEQ_LEN * 100]  # 100 breaths
        val_df = val_df.iloc[: Config.SEQ_LEN * 20]
        test_df = test_df.iloc[: Config.SEQ_LEN * 20]

    # Apply Feature Engineering
    print("Generating features for Train...")
    train_df = add_features(train_df)
    print("Generating features for Val...")
    val_df = add_features(val_df)
    print("Generating features for Test...")
    test_df = add_features(test_df)

    # Define columns to be used as input features
    feature_cols = Config.CONT_FEATURES

    # Scaling
    print("Fitting Scaler...")
    scaler = RobustScaler()

    # Fit scaler only on training data
    train_values = train_df[feature_cols].values
    scaler.fit(train_values)

    # Save scaler parameters manually (No pickle allowed)
    np.save(cache_files["scaler_center"], scaler.center_)
    np.save(cache_files["scaler_scale"], scaler.scale_)

    # Transform all splits
    print("Transforming data...")
    train_scaled = scaler.transform(train_values)
    val_scaled = scaler.transform(val_df[feature_cols].values)
    test_scaled = scaler.transform(test_df[feature_cols].values)

    # Reshape helper
    def reshape_dataset(scaled_flat, df, is_test=False):
        # Calculate number of breaths
        n_breaths = len(df) // Config.SEQ_LEN
        n_feats = scaled_flat.shape[1]

        # Reshape to (N_breaths, 80, N_features)
        x = scaled_flat.reshape(n_breaths, Config.SEQ_LEN, n_feats)

        # Reshape u_out for masking (keep as raw binary 0/1)
        u_out = df["u_out"].values.reshape(n_breaths, Config.SEQ_LEN)

        if not is_test:
            # Reshape target
            y = df[Config.TARGET_COL].values.reshape(n_breaths, Config.SEQ_LEN)
            return x, y, u_out, None
        else:
            # Reshape IDs for submission
            ids = df["id"].values.reshape(n_breaths, Config.SEQ_LEN)
            return x, None, u_out, ids

    # Reshape all datasets
    print("Reshaping datasets...")
    train_x, train_y, train_u_out, _ = reshape_dataset(train_scaled, train_df)
    val_x, val_y, val_u_out, _ = reshape_dataset(val_scaled, val_df)
    test_x, _, test_u_out, test_ids = reshape_dataset(
        test_scaled, test_df, is_test=True
    )

    # Save processed arrays to cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    np.save(cache_files["train_x"], train_x)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["train_u_out"], train_u_out)

    np.save(cache_files["val_x"], val_x)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["val_u_out"], val_u_out)

    np.save(cache_files["test_x"], test_x)
    np.save(cache_files["test_u_out"], test_u_out)
    np.save(cache_files["test_ids"], test_ids)

    # Return tensors
    train_data = {
        "x": torch.tensor(train_x, dtype=torch.float32),
        "y": torch.tensor(train_y, dtype=torch.float32),
        "u_out": torch.tensor(train_u_out, dtype=torch.float32),
    }
    val_data = {
        "x": torch.tensor(val_x, dtype=torch.float32),
        "y": torch.tensor(val_y, dtype=torch.float32),
        "u_out": torch.tensor(val_u_out, dtype=torch.float32),
    }
    test_data = {
        "x": torch.tensor(test_x, dtype=torch.float32),
        "u_out": torch.tensor(test_u_out, dtype=torch.float32),
        "ids": torch.tensor(test_ids, dtype=torch.long),
    }

    print("Data processing complete.")
    return train_data, val_data, test_data
