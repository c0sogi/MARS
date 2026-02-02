import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from library.config import Config


def get_all_feature_names():
    """
    Returns the sorted list of all unique features required by the model.
    This ensures a deterministic column order in the processed numpy arrays.
    """
    return Config.MODEL_FEATURES


def add_engineered_features(df):
    """
    Computes and adds the required physical and PID features to the DataFrame.

    Args:
        df (pd.DataFrame): Raw dataframe containing breath data.

    Returns:
        pd.DataFrame: DataFrame with added feature columns.
    """
    # Ensure data is sorted for correct diff/cumsum operations
    df = df.sort_values(["breath_id", "time_step"])

    # 1. PID Features (Derivatives and Integrals)
    # We re-instantiate groupby for each operation to ensure we pick up newly created columns

    # Integral (Volume proxy)
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # First Derivative (Velocity / Flow Acceleration)
    df["u_in_diff1"] = df.groupby("breath_id")["u_in"].diff().fillna(0)

    # Second Derivative (Jerk)
    df["u_in_diff2"] = df.groupby("breath_id")["u_in_diff1"].diff().fillna(0)

    # 2. Physical Interaction Terms (Equation of Motion)
    # R * Flow (Resistive Pressure component)
    df["R_flow"] = df["R"] * df["u_in"]

    # Volume / C (Elastic Pressure component)
    # C is strictly positive (10, 20, 50), so division is safe.
    df["C_volume"] = df["u_in_cumsum"] / df["C"]

    return df


def load_and_preprocess_data(split, load_cached_data=True):
    """
    Main function to load data, generate features, scale, reshape, and cache.

    Args:
        split (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        X (np.ndarray): Shape (num_breaths, 80, num_features)
        y (np.ndarray): Shape (num_breaths, 80) or None for test.
    """
    # Determine paths based on split
    if split == "train":
        csv_path = Config.TRAIN_PATH
        cache_x_path = Config.TRAIN_CACHE_X
        cache_y_path = Config.TRAIN_CACHE_Y
    elif split == "validation":
        csv_path = Config.VAL_PATH
        cache_x_path = Config.VAL_CACHE_X
        cache_y_path = Config.VAL_CACHE_Y
    elif split == "test":
        csv_path = Config.TEST_PATH
        cache_x_path = Config.TEST_CACHE_X
        cache_y_path = None
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Loading from Cache
    if load_cached_data:
        # Check if cache exists
        cache_exists = os.path.exists(cache_x_path) and (
            cache_y_path is None or os.path.exists(cache_y_path)
        )

        # For training data, we also need the scaler stats to be present
        if split == "train" and not os.path.exists(Config.SCALER_PATH):
            cache_exists = False

        if cache_exists:
            print(f"Loading cached {split} data from {Config.WORKING_DIR}...")
            X = np.load(cache_x_path)
            y = np.load(cache_y_path) if cache_y_path else None
            return X, y
        else:
            print(f"Cache incomplete or missing for {split}. Regenerating...")
    else:
        print(f"Forcing regeneration of {split} data...")

    # 2. Generate Data from Source
    print(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Feature Engineering
    print("Adding engineered features...")
    df = add_engineered_features(df)

    # Identify feature columns to extract
    feature_cols = get_all_feature_names()

    # Scaling Logic
    # We scale continuous features. u_out is binary, keep as is.
    scale_cols = [c for c in feature_cols if c != "u_out"]

    if split == "train":
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        # Save scaler stats manually to avoid pickle usage
        # RobustScaler uses center_ (median) and scale_ (IQR)
        np.savez(Config.SCALER_PATH, center=scaler.center_, scale=scaler.scale_)
        print(f"Scaler statistics saved to {Config.SCALER_PATH}")

    else:
        print("Applying scaler transform...")
        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler stats not found at {Config.SCALER_PATH}. You must run the 'train' split first."
            )

        # Load stats and manually apply transform: (X - center) / scale
        stats = np.load(Config.SCALER_PATH)
        center = stats["center"]
        scale = stats["scale"]

        df[scale_cols] = (df[scale_cols] - center) / scale

    # Reshaping to 3D Arrays
    print(f"Reshaping {split} data to (N_breaths, 80, N_features)...")

    # Ensure strict sorting by breath and time
    df.sort_values(["breath_id", "time_step"], inplace=True)

    num_breaths = df["breath_id"].nunique()
    # Verify data integrity (should be exactly 80 steps per breath)
    if len(df) != num_breaths * 80:
        print(
            f"Warning: Data length {len(df)} is not perfectly divisible by 80. Check data integrity."
        )

    # Extract features and reshape
    X = df[feature_cols].values.reshape(num_breaths, 80, len(feature_cols))

    # Extract targets if available
    if split != "test":
        y = df[Config.TARGET_COL].values.reshape(num_breaths, 80)
    else:
        y = None

    # 3. Save to Cache
    print(f"Saving processed {split} data to cache...")
    os.makedirs(os.path.dirname(cache_x_path), exist_ok=True)

    np.save(cache_x_path, X)
    if y is not None:
        np.save(cache_y_path, y)

    return X, y
