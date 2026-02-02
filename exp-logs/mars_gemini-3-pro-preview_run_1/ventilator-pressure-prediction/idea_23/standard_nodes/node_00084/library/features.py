import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config

# Constants
STEPS_PER_BREATH = 80


def get_scaler_paths():
    """Returns paths for scaler center and scale parameters."""
    center_path = os.path.join(Config.WORKING_DIR, "scaler_center.npy")
    scale_path = os.path.join(Config.WORKING_DIR, "scaler_scale.npy")
    return center_path, scale_path


def engineer_features(df):
    """
    Applies physics-based feature engineering and lag generation.
    """
    # Ensure data is sorted by breath_id and time_step just in case
    df = df.sort_values([Config.BREATH_ID_COL, Config.TIME_COL])

    # --- 1. Physics Features ---
    # Calculate dt: time difference between steps
    # Group by breath_id to ensure diff doesn't cross breaths. Fillna(0) for the first step.
    df["dt"] = df.groupby(Config.BREATH_ID_COL)[Config.TIME_COL].diff().fillna(0)

    # Calculate Volume = Cumulative Sum of (u_in * dt)
    # We use the 'u_in' column directly.
    df["volume"] = (
        (df[Config.U_IN_COL] * df["dt"]).groupby(df[Config.BREATH_ID_COL]).cumsum()
    )

    # Interaction Terms
    if Config.USE_INTERACTIONS:
        # R * u_in: Resistance interaction
        df["R_u_in"] = df[Config.R_COL] * df[Config.U_IN_COL]

        # Volume / C: Compliance interaction (Pressure ~ Vol/C)
        # C values are typically 10, 20, 50, so no division by zero risk in this dataset
        df["vol_C"] = df["volume"] / df[Config.C_COL]

    # --- 2. Lag Features ---
    if Config.USE_LAGS:
        for lag in Config.LAG_STEPS:
            # Shift u_in by lag steps, filling with 0
            df[f"u_in_lag{lag}"] = (
                df.groupby(Config.BREATH_ID_COL)[Config.U_IN_COL].shift(lag).fillna(0)
            )

    if Config.USE_DIFFS:
        # First difference of u_in
        df["u_in_diff1"] = (
            df.groupby(Config.BREATH_ID_COL)[Config.U_IN_COL].diff().fillna(0)
        )
        # Second difference of u_in
        df["u_in_diff2"] = (
            df.groupby(Config.BREATH_ID_COL)["u_in_diff1"].diff().fillna(0)
        )

    return df


def prepare_dataset(split="train", load_cached_data=True):
    """
    Loads data, engineers features, scales continuous features, and reshapes for LSTM.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        dict: {
            'X': np.ndarray (N_breaths, 80, F),
            'y': np.ndarray (N_breaths, 80) or None,
            'u_out': np.ndarray (N_breaths, 80),
            'ids': np.ndarray (N_breaths, 80)
        }
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    # Append debug status to filename to prevent stale cache collision
    suffix = "_debug" if Config.DEBUG else "_full"
    cache_prefix = os.path.join(Config.WORKING_DIR, f"{split}_processed{suffix}")
    path_X = f"{cache_prefix}_X.npy"
    path_y = f"{cache_prefix}_y.npy"
    path_uout = f"{cache_prefix}_uout.npy"
    path_ids = f"{cache_prefix}_ids.npy"

    # Check cache
    if load_cached_data:
        files_exist = (
            os.path.exists(path_X)
            and os.path.exists(path_uout)
            and os.path.exists(path_ids)
        )
        if split != "test":
            files_exist = files_exist and os.path.exists(path_y)

        if files_exist:
            print(f"Loading cached {split} data from {Config.WORKING_DIR}...")
            X = np.load(path_X)
            u_out = np.load(path_uout)
            ids = np.load(path_ids)
            y = np.load(path_y) if split != "test" else None
            return {"X": X, "y": y, "u_out": u_out, "ids": ids}

    print(f"Processing {split} data from scratch...")

    # Load raw data
    if split == "train":
        file_path = Config.TRAIN_FILE
    elif split == "val":
        file_path = Config.VAL_FILE
    elif split == "test":
        file_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    df = pd.read_csv(file_path)

    # Debug mode: subset data
    if Config.DEBUG:
        print("DEBUG MODE: Using subset of data.")
        unique_breaths = df[Config.BREATH_ID_COL].unique()[:200]
        df = df[df[Config.BREATH_ID_COL].isin(unique_breaths)].copy()

    # Engineer Features
    df = engineer_features(df)

    # Define Feature Columns
    # Exclude IDs, Targets, and u_out (handled separately)
    exclude_cols = [
        Config.ID_COL,
        Config.BREATH_ID_COL,
        Config.TARGET_COL,
        Config.U_OUT_COL,
    ]
    # We only scale continuous features
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Scaling Logic (Manual save/load to avoid pickle)
    center_path, scale_path = get_scaler_paths()
    X_continuous = df[feature_cols].values.astype(np.float32)

    if split == "train":
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler()
        scaler.fit(X_continuous)

        # Save scaler parameters manually
        np.save(center_path, scaler.center_)
        np.save(scale_path, scaler.scale_)

        X_scaled = scaler.transform(X_continuous)
    else:
        print("Loading RobustScaler parameters for transform...")
        if not os.path.exists(center_path) or not os.path.exists(scale_path):
            raise FileNotFoundError(
                f"Scaler parameters not found. Run train split first."
            )

        center = np.load(center_path)
        scale = np.load(scale_path)

        # Manual transform: (X - center) / scale
        X_scaled = (X_continuous - center) / scale

    # Prepare u_out (Binary, unscaled)
    u_out_flat = df[Config.U_OUT_COL].values.astype(np.float32)

    # Concatenate Scaled Features + Unscaled u_out
    # X shape: (Total_Steps, Num_Features + 1)
    X_combined = np.column_stack([X_scaled, u_out_flat])

    # Reshape to (N_breaths, 80, n_features)
    num_breaths = len(df) // STEPS_PER_BREATH
    if len(df) % STEPS_PER_BREATH != 0:
        raise ValueError(
            f"Data length {len(df)} is not divisible by {STEPS_PER_BREATH}"
        )

    X = X_combined.reshape(num_breaths, STEPS_PER_BREATH, -1)
    u_out = u_out_flat.reshape(num_breaths, STEPS_PER_BREATH)
    ids = df[Config.ID_COL].values.reshape(num_breaths, STEPS_PER_BREATH)

    if split != "test":
        y = df[Config.TARGET_COL].values.reshape(num_breaths, STEPS_PER_BREATH)
    else:
        y = None

    # Save to Cache
    print(f"Saving {split} data to cache...")
    np.save(path_X, X)
    np.save(path_uout, u_out)
    np.save(path_ids, ids)
    if y is not None:
        np.save(path_y, y)

    return {"X": X, "y": y, "u_out": u_out, "ids": ids}
