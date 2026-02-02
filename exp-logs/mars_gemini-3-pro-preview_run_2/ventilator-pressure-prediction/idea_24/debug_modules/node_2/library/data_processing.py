import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from library.config import Config


def add_physics_features(df):
    """
    Computes and adds physics-based features to the dataframe.
    Includes time-integration, interaction terms, and finite differences.
    """
    # Ensure data is sorted by breath and time
    df = df.sort_values([Config.BREATH_ID_COL, "time_step"]).copy()

    # Group by breath_id for window operations
    # Using transform to keep the original index alignment

    # 1. Time Delta (dt)
    # The first time step in a breath has no previous step, so dt is 0 (or we can assume small delta)
    # Here we fill with 0 to match standard practice
    df["dt"] = df.groupby(Config.BREATH_ID_COL)["time_step"].diff().fillna(0)

    # 2. Volume Integration (Time-Weighted)
    # Volume = Integral(Flow * dt) ~= Cumsum(u_in * dt)
    # u_in is proportional to flow (0-100)
    df["volume_chunk"] = df["u_in"] * df["dt"]
    df["u_in_cumsum"] = df.groupby(Config.BREATH_ID_COL)["volume_chunk"].cumsum()

    # 3. Physics Interactions
    # Resistive Pressure component ~ R * Flow
    df["R_u_in"] = df["R"] * df["u_in"]

    # Elastic Pressure component ~ Volume / Compliance
    df["u_in_cumsum_div_C"] = df["u_in_cumsum"] / df["C"]

    # 4. Finite Differences (Dynamics)
    # 1st Difference
    df["u_in_diff1"] = df.groupby(Config.BREATH_ID_COL)["u_in"].diff().fillna(0)
    # 2nd Difference
    df["u_in_diff2"] = df.groupby(Config.BREATH_ID_COL)["u_in_diff1"].diff().fillna(0)

    # Drop temporary column
    df = df.drop(columns=["volume_chunk"])

    return df


def save_scaler_params(scaler, path):
    """Saves RobustScaler parameters (center, scale) to npz."""
    np.savez(path, center=scaler.center_, scale=scaler.scale_)


def load_scaler_params(path):
    """Loads RobustScaler parameters from npz."""
    data = np.load(path)
    return data["center"], data["scale"]


def apply_segregated_scaling(df, is_train, scaler_path):
    """
    Applies RobustScaler to continuous features and passes binary features raw.
    Fits on train, transforms on val/test using saved params.
    """
    continuous_cols = Config.CONTINUOUS_FEATURES
    binary_cols = Config.BINARY_FEATURES

    # Extract continuous part
    X_cont = df[continuous_cols].values

    if is_train:
        # Fit scaler
        scaler = RobustScaler()
        X_cont_scaled = scaler.fit_transform(X_cont)
        # Save params
        save_scaler_params(scaler, scaler_path)
    else:
        # Load params and transform manually to avoid pickle/sklearn version issues
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler params not found at {scaler_path}. Process train data first."
            )

        center, scale = load_scaler_params(scaler_path)
        # RobustScaler logic: (X - center) / scale
        # Handle potential division by zero if scale is 0 (though unlikely with RobustScaler on this data)
        # Sklearn handles 0 scale by setting it to 1. We assume valid scale here.
        X_cont_scaled = (X_cont - center) / scale

    # Reassemble dataframe features
    # We return a numpy array of features directly to save memory/overhead
    X_binary = df[binary_cols].values

    # Concatenate: [Continuous_Scaled, Binary_Raw]
    X_combined = np.hstack([X_cont_scaled, X_binary])

    return X_combined.astype(np.float32)


def reshape_to_tensor(X_flat, y_flat, u_out_flat, num_breaths):
    """
    Reshapes flat arrays into (Num_Breaths, 80, Num_Features).
    Assumes fixed sequence length of 80.
    """
    SEQ_LEN = 80

    # Safety check
    if X_flat.shape[0] != num_breaths * SEQ_LEN:
        raise ValueError(
            f"Data length {X_flat.shape[0]} does not match expected {num_breaths} * {SEQ_LEN}"
        )

    num_features = X_flat.shape[1]

    X = X_flat.reshape(num_breaths, SEQ_LEN, num_features)
    u_out = u_out_flat.reshape(num_breaths, SEQ_LEN)

    if y_flat is not None:
        y = y_flat.reshape(num_breaths, SEQ_LEN)
    else:
        y = None

    return X, y, u_out


def load_dataset(split, debug=False, load_cached_data=True):
    """
    Main function to load and process data for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, use a small subset.
        load_cached_data (bool): If True, try to load from disk.

    Returns:
        tuple: (X, y, u_out) tensors. y is None for test.
    """
    # Define paths
    cache_file = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")
    scaler_path = os.path.join(Config.CACHE_DIR, "scaler_params.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {split} data from {cache_file}...")
        try:
            data = np.load(cache_file)
            X = data["X"]
            u_out = data["u_out"]
            if "y" in data:
                y = data["y"]
            else:
                y = None
            return X, y, u_out
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing {split} data from scratch...")

    # 2. Load Metadata
    if split == "train":
        meta_path = Config.TRAIN_METADATA
        raw_csv = Config.TRAIN_CSV
        is_train_mode = True
    elif split == "val":
        meta_path = Config.VAL_METADATA
        raw_csv = Config.TRAIN_CSV  # Val comes from train file
        is_train_mode = False
    elif split == "test":
        meta_path = Config.TEST_METADATA
        raw_csv = Config.TEST_CSV
        is_train_mode = False
    else:
        raise ValueError(f"Unknown split: {split}")

    df_meta = pd.read_csv(meta_path)

    # Debug Subsetting
    if debug:
        unique_breaths = df_meta[Config.BREATH_ID_COL].unique()
        subset_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
        df_meta = df_meta[df_meta[Config.BREATH_ID_COL].isin(subset_breaths)]
        print(f"DEBUG: Subsetting to {len(subset_breaths)} breaths.")

    # 3. Load Raw Data
    # We load the full raw CSV and filter by the breath_ids in the metadata
    # This is memory intensive but safe. For efficiency with huge files,
    # one might read only specific rows, but pandas read_csv is fast enough for 5GB on this hardware.
    # Optimization: Read only needed columns first? No, we need most columns.

    # To save memory, we can filter the raw dataframe immediately
    target_breath_ids = set(df_meta[Config.BREATH_ID_COL].unique())

    # Load raw data
    # Note: We can't easily skip rows without knowing where they are,
    # but we can filter immediately after load.
    df_raw = pd.read_csv(raw_csv)
    df = df_raw[df_raw[Config.BREATH_ID_COL].isin(target_breath_ids)].copy()
    del df_raw  # Free memory

    # Ensure alignment with metadata (though filter should handle it)
    # The metadata defines the exact set of breaths for this split.

    # 4. Feature Engineering
    df = add_physics_features(df)

    # 5. Segregated Scaling
    # Note: If debug=True, the scaler will be fit on the debug subset.
    # This is acceptable for debugging flows.
    X_combined = apply_segregated_scaling(
        df, is_train=is_train_mode, scaler_path=scaler_path
    )

    # 6. Prepare Targets and Auxiliaries
    u_out_flat = df["u_out"].values

    if Config.TARGET_COL in df.columns:
        y_flat = df[Config.TARGET_COL].values
    else:
        y_flat = None

    # 7. Reshape to Tensor
    num_breaths = df[Config.BREATH_ID_COL].nunique()
    X, y, u_out = reshape_to_tensor(X_combined, y_flat, u_out_flat, num_breaths)

    # 8. Save to Cache
    print(f"Saving {split} data to {cache_file}...")
    save_dict = {"X": X, "u_out": u_out}
    if y is not None:
        save_dict["y"] = y
    np.savez(cache_file, **save_dict)

    return X, y, u_out


def load_train_data(debug=Config.DEBUG, load_cached_data=True):
    return load_dataset("train", debug, load_cached_data)


def load_val_data(debug=Config.DEBUG, load_cached_data=True):
    return load_dataset("val", debug, load_cached_data)


def load_test_data(debug=Config.DEBUG, load_cached_data=True):
    return load_dataset("test", debug, load_cached_data)
