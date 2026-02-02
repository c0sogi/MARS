import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.feature_engineering import load_and_process_data


# ==========================================
# Helper Functions for Scaler Persistence
# ==========================================
def save_scaler_params(scaler, path):
    """
    Saves RobustScaler parameters (center_, scale_) to a .npz file.
    Avoids using pickle.
    """
    if not hasattr(scaler, "center_") or not hasattr(scaler, "scale_"):
        raise ValueError("Scaler must be fitted before saving.")

    np.savez(path, center=scaler.center_, scale=scaler.scale_)


def load_scaler_params(path):
    """
    Loads RobustScaler parameters from a .npz file and returns a fitted scaler.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler params not found at {path}")

    data = np.load(path)
    scaler = RobustScaler()
    scaler.center_ = data["center"]
    scaler.scale_ = data["scale"]
    return scaler


# ==========================================
# Dataset Class
# ==========================================
class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.

    Attributes:
        X (torch.Tensor): Input features of shape (N_breaths, 80, N_features).
        y (torch.Tensor): Target pressure of shape (N_breaths, 80).
        u_out (torch.Tensor): Binary mask for expiratory phase of shape (N_breaths, 80).
    """

    def __init__(self, X, y=None, u_out=None):
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

        if u_out is not None:
            self.u_out = torch.tensor(u_out, dtype=torch.float32)
        else:
            self.u_out = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx]}

        if self.y is not None:
            item["y"] = self.y[idx]

        if self.u_out is not None:
            item["u_out"] = self.u_out[idx]

        return item


# ==========================================
# Main Data Preparation Function
# ==========================================
def prepare_data(load_cached_data=True, debug=False):
    """
    Loads, scales, reshapes, and caches data for the Dual-Stream model.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        debug (bool): If True, uses a subset of data.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, scaler)
    """
    # Ensure working directory exists
    Config.setup()

    # Define Cache Paths
    suffix = "_debug" if debug else ""
    cache_dir = Config.WORKING_DIR

    paths = {
        "X_train": os.path.join(cache_dir, f"X_train{suffix}.npy"),
        "y_train": os.path.join(cache_dir, f"y_train{suffix}.npy"),
        "u_out_train": os.path.join(cache_dir, f"u_out_train{suffix}.npy"),
        "X_val": os.path.join(cache_dir, f"X_val{suffix}.npy"),
        "y_val": os.path.join(cache_dir, f"y_val{suffix}.npy"),
        "u_out_val": os.path.join(cache_dir, f"u_out_val{suffix}.npy"),
        "X_test": os.path.join(cache_dir, f"X_test{suffix}.npy"),
        "u_out_test": os.path.join(cache_dir, f"u_out_test{suffix}.npy"),
        "scaler": os.path.join(cache_dir, f"scaler_params{suffix}.npz"),
    }

    # ------------------------------------------
    # 1. Try Loading from Cache
    # ------------------------------------------
    if load_cached_data:
        # Check if all required files exist
        # Note: We always expect train/val/test to be generated together
        all_exist = all(os.path.exists(p) for p in paths.values())

        if all_exist:
            print("Loading pre-processed data from cache...")
            X_train = np.load(paths["X_train"])
            y_train = np.load(paths["y_train"])
            u_out_train = np.load(paths["u_out_train"])

            X_val = np.load(paths["X_val"])
            y_val = np.load(paths["y_val"])
            u_out_val = np.load(paths["u_out_val"])

            X_test = np.load(paths["X_test"])
            u_out_test = np.load(paths["u_out_test"])

            scaler = load_scaler_params(paths["scaler"])

            train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
            val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
            test_dataset = VentilatorDataset(X_test, None, u_out_test)

            return train_dataset, val_dataset, test_dataset, scaler
        else:
            print("Cache incomplete or missing. Reprocessing data...")

    # ------------------------------------------
    # 2. Process from Scratch
    # ------------------------------------------

    # A. Load DataFrames (with Feature Engineering)
    df_train = load_and_process_data(
        "train", load_cached_data=load_cached_data, debug=debug
    )
    df_val = load_and_process_data(
        "val", load_cached_data=load_cached_data, debug=debug
    )
    df_test = load_and_process_data(
        "test", load_cached_data=load_cached_data, debug=debug
    )

    # B. Identify Feature Columns
    # Exclude ID, Breath ID, Target, Source File
    exclude_cols = {
        Config.ID_COL,
        Config.BREATH_ID_COL,
        Config.TARGET_COL,
        "source_file",
    }
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"Selected {len(feature_cols)} features: {feature_cols}")

    # C. Fit Scaler (RobustScaler)
    print("Fitting Scaler on Training Data...")
    scaler = RobustScaler()
    # We fit on the flattened training data
    scaler.fit(df_train[feature_cols].values)

    # D. Transform and Reshape Helper
    def process_split(df, is_test=False):
        # 1. Extract u_out for mask (before scaling/transforming)
        # Assuming u_out is in feature_cols, we extract it separately
        u_out = df["u_out"].values

        # 2. Scale Features
        X_scaled = scaler.transform(df[feature_cols].values)

        # 3. Reshape to (N_breaths, 80, N_features)
        # Verify length
        num_rows = len(df)
        if num_rows % 80 != 0:
            raise ValueError(
                f"Data length {num_rows} is not divisible by 80. Reshaping impossible."
            )

        num_breaths = num_rows // 80
        num_features = len(feature_cols)

        X_reshaped = X_scaled.reshape(num_breaths, 80, num_features)
        u_out_reshaped = u_out.reshape(num_breaths, 80)

        y_reshaped = None
        if not is_test:
            y = df[Config.TARGET_COL].values
            y_reshaped = y.reshape(num_breaths, 80)

        return X_reshaped, y_reshaped, u_out_reshaped

    print("Transforming and Reshaping Train...")
    X_train, y_train, u_out_train = process_split(df_train, is_test=False)

    print("Transforming and Reshaping Val...")
    X_val, y_val, u_out_val = process_split(df_val, is_test=False)

    print("Transforming and Reshaping Test...")
    X_test, _, u_out_test = process_split(df_test, is_test=True)

    # E. Save to Cache
    print("Saving processed arrays to cache...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["u_out_train"], u_out_train)

    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["u_out_val"], u_out_val)

    np.save(paths["X_test"], X_test)
    np.save(paths["u_out_test"], u_out_test)

    save_scaler_params(scaler, paths["scaler"])

    # F. Create Datasets
    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(X_test, None, u_out_test)

    return train_dataset, val_dataset, test_dataset, scaler
