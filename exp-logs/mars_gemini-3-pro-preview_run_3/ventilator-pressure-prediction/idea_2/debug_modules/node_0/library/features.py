import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def engineer_features(df):
    """
    Transforms raw dataframe into model-ready tensors using vectorized operations.
    Computes physics-inspired features (Integral, Derivative, Interactions).

    Args:
        df (pd.DataFrame): Raw input data containing all time steps.

    Returns:
        x_tensor (torch.Tensor): Feature tensor of shape (num_breaths, seq_len, num_features).
        y_tensor (torch.Tensor or None): Target tensor of shape (num_breaths, seq_len).
    """
    # Verify data integrity
    if len(df) % Config.SEQ_LEN != 0:
        raise ValueError(
            f"Dataframe length {len(df)} is not divisible by SEQ_LEN {Config.SEQ_LEN}"
        )

    num_breaths = len(df) // Config.SEQ_LEN

    # Reshape raw columns into (N, 80) matrices for vectorized calculation
    # We assume the dataframe is sorted by breath_id and time_step, which is standard for this dataset
    u_in = df["u_in"].values.reshape(num_breaths, Config.SEQ_LEN)
    u_out = df["u_out"].values.reshape(num_breaths, Config.SEQ_LEN)
    R = df["R"].values.reshape(num_breaths, Config.SEQ_LEN)
    C = df["C"].values.reshape(num_breaths, Config.SEQ_LEN)
    time_step = df["time_step"].values.reshape(num_breaths, Config.SEQ_LEN)

    # --- Feature Engineering ---

    # 1. Integral (Volume Proxy): Cumulative sum of u_in
    # axis=1 represents the time dimension
    u_in_cumsum = np.cumsum(u_in, axis=1)

    # 2. Derivatives (Flow Change/Acceleration): Differences of u_in
    # prepend=0 assumes the valve starts from a neutral/closed state relative to the step before
    u_in_diff1 = np.diff(u_in, axis=1, prepend=0)
    u_in_diff2 = np.diff(u_in_diff1, axis=1, prepend=0)

    # 3. Physics Interactions
    # R * Flow (Resistance component of pressure)
    R_flow = R * u_in
    # Volume / C (Elastic component of pressure)
    # Note: u_in_cumsum is a proxy for volume
    C_volume = u_in_cumsum / C

    # --- Tensor Assembly ---

    # Map feature names to their computed arrays
    feature_map = {
        "time_step": time_step,
        "u_in": u_in,
        "u_out": u_out,
        "R": R,
        "C": C,
        "u_in_cumsum": u_in_cumsum,
        "u_in_diff1": u_in_diff1,
        "u_in_diff2": u_in_diff2,
        "R_flow": R_flow,
        "C_volume": C_volume,
    }

    # Stack features in the exact order defined in Config
    try:
        feature_arrays = [feature_map[col] for col in Config.FEATURE_COLS]
    except KeyError as e:
        raise KeyError(f"Feature {e} defined in Config not found in generated map.")

    # Stack along the last dimension -> (N, 80, Num_Features)
    x_array = np.stack(feature_arrays, axis=-1)
    x_tensor = torch.tensor(x_array, dtype=torch.float32)

    # --- Target Extraction ---
    y_tensor = None
    if Config.TARGET_COL in df.columns:
        y_array = df[Config.TARGET_COL].values.reshape(num_breaths, Config.SEQ_LEN)
        y_tensor = torch.tensor(y_array, dtype=torch.float32)

    return x_tensor, y_tensor


def prepare_dataset(split="train", debug=False, load_cached_data=True):
    """
    Loads data, performs feature engineering, and handles caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, processes a small subset of data.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        x (torch.Tensor): Input features.
        y (torch.Tensor or None): Targets.
    """
    # Ensure working directories exist
    Config.initialize()

    # Determine paths based on split
    if split == "train":
        data_path = Config.TRAIN_DATA_PATH
        base_cache_path = Config.TRAIN_CACHE
    elif split == "val":
        data_path = Config.VAL_DATA_PATH
        base_cache_path = Config.VAL_CACHE
    elif split == "test":
        data_path = Config.TEST_DATA_PATH
        base_cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Invalid split: {split}")

    # Define separate cache paths for X and Y to avoid pickle
    cache_path_x = base_cache_path.replace(".npy", "_x.npy")
    cache_path_y = base_cache_path.replace(".npy", "_y.npy")

    # 1. Try Loading Cache
    # We skip cache loading in debug mode to ensure we actually sample the raw data
    if load_cached_data and not debug:
        if os.path.exists(cache_path_x):
            try:
                x_np = np.load(cache_path_x)
                y_np = np.load(cache_path_y) if os.path.exists(cache_path_y) else None

                # Validation: If train/val split, targets are mandatory
                if split != "test" and y_np is None:
                    print(f"[{split}] Cache found but targets missing. Recomputing...")
                else:
                    print(f"[{split}] Loading features from {cache_path_x}...")
                    x_tensor = torch.from_numpy(x_np)
                    y_tensor = torch.from_numpy(y_np) if y_np is not None else None
                    return x_tensor, y_tensor
            except Exception as e:
                print(f"[{split}] Error loading cache ({e}). Recomputing...")

    # 2. Compute from Scratch
    print(f"[{split}] Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)

    # Debug Sampling
    if debug:
        print(f"[{split}] Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} breaths.")
        unique_breaths = df[Config.BREATH_ID_COL].unique()
        if len(unique_breaths) > Config.DEBUG_SAMPLE_SIZE:
            sample_ids = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]
            df = df[df[Config.BREATH_ID_COL].isin(sample_ids)].copy()

    # Process
    print(f"[{split}] Engineering features...")
    x_tensor, y_tensor = engineer_features(df)

    # 3. Save to Cache
    # Only save if not in debug mode to keep cache clean
    if not debug:
        print(f"[{split}] Saving features to cache: {cache_path_x}")
        np.save(cache_path_x, x_tensor.numpy())

        if y_tensor is not None:
            print(f"[{split}] Saving targets to cache: {cache_path_y}")
            np.save(cache_path_y, y_tensor.numpy())

    return x_tensor, y_tensor
