import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config


# ==========================================
# Dataset Class
# ==========================================
class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    """

    def __init__(self, X, u_out, y=None):
        """
        Args:
            X (np.ndarray): Feature array of shape (Num_Breaths, 80, Num_Features).
            u_out (np.ndarray): Expiratory control signal of shape (Num_Breaths, 80).
            y (np.ndarray, optional): Target pressure array of shape (Num_Breaths, 80).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            X (tensor): Input features.
            u_out (tensor): Expiratory valve signal (used for loss weighting).
            y (tensor): Target pressure (or dummy if test).
        """
        if self.y is not None:
            return self.X[idx], self.u_out[idx], self.y[idx]
        else:
            # Return dummy target for test set inference
            return self.X[idx], self.u_out[idx], torch.zeros_like(self.u_out[idx])


# ==========================================
# Feature Engineering
# ==========================================
def compute_physics_features(df):
    """
    Computes physics-based features using vectorized numpy operations.
    Assumes the dataframe is sorted by breath_id and time_step, and
    that every breath has exactly 80 time steps.
    """
    # Extract raw columns
    # Shape: (N_breaths * 80)
    time_step = df["time_step"].values
    u_in = df["u_in"].values
    u_out = df["u_out"].values
    R = df["R"].values
    C = df["C"].values

    # Reshape to (N_breaths, 80) for vectorized ops
    n_breaths = len(df) // 80

    time_step_r = time_step.reshape(n_breaths, 80)
    u_in_r = u_in.reshape(n_breaths, 80)
    u_out_r = u_out.reshape(n_breaths, 80)
    R_r = R.reshape(n_breaths, 80)
    C_r = C.reshape(n_breaths, 80)

    # 1. Calculate dt (Time Delta)
    # dt = time_step[t] - time_step[t-1]
    # For t=0, we assume dt=0 (or small epsilon, but 0 is safe for integration start)
    dt_r = np.zeros_like(time_step_r)
    dt_r[:, 1:] = time_step_r[:, 1:] - time_step_r[:, :-1]

    # 2. Calculate Volume (Time-Weighted Integration)
    # volume = cumsum(u_in * dt)
    volume_r = np.cumsum(u_in_r * dt_r, axis=1)

    # 3. Interaction Terms
    # Resistive Pressure ~ R * u_in
    R_u_in_r = R_r * u_in_r
    # Elastic Pressure ~ Volume / C
    u_in_cumsum_C_r = volume_r / C_r

    # 4. Lags and Diffs (u_in)
    # Helper to shift and fill
    def lag_array(arr, shift):
        res = np.zeros_like(arr)
        if shift > 0:
            res[:, shift:] = arr[:, :-shift]
        return res

    u_in_lag1_r = lag_array(u_in_r, 1)
    u_in_lag2_r = lag_array(u_in_r, 2)
    u_in_lag3_r = lag_array(u_in_r, 3)
    u_in_lag4_r = lag_array(u_in_r, 4)

    # Diffs (Finite Differences)
    # diff1 = u_in[t] - u_in[t-1]
    u_in_diff1_r = u_in_r - u_in_lag1_r
    # diff2 = diff1[t] - diff1[t-1]
    u_in_diff2_r = u_in_diff1_r - lag_array(u_in_diff1_r, 1)

    # 5. Lags and Diffs (u_out)
    u_out_lag1_r = lag_array(u_out_r, 1)
    u_out_lag2_r = lag_array(u_out_r, 2)
    u_out_diff1_r = u_out_r - u_out_lag1_r

    # Flatten back to 1D arrays
    features_dict = {
        "time_step": time_step,
        "u_in": u_in,
        "u_out": u_out,
        "R": R,
        "C": C,
        "dt": dt_r.flatten(),
        "u_in_cumsum": volume_r.flatten(),
        "R_u_in": R_u_in_r.flatten(),
        "u_in_cumsum_C": u_in_cumsum_C_r.flatten(),
        "u_in_lag1": u_in_lag1_r.flatten(),
        "u_in_lag2": u_in_lag2_r.flatten(),
        "u_in_lag3": u_in_lag3_r.flatten(),
        "u_in_lag4": u_in_lag4_r.flatten(),
        "u_in_diff1": u_in_diff1_r.flatten(),
        "u_in_diff2": u_in_diff2_r.flatten(),
        "u_out_lag1": u_out_lag1_r.flatten(),
        "u_out_lag2": u_out_lag2_r.flatten(),
        "u_out_diff1": u_out_diff1_r.flatten(),
    }

    # Ensure column order matches Config INPUT_DIM expectation (18 features)
    feature_order = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "dt",
        "u_in_cumsum",
        "R_u_in",
        "u_in_cumsum_C",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",
        "u_in_diff2",
        "u_out_lag1",
        "u_out_lag2",
        "u_out_diff1",
    ]

    # Stack features: (N_total, 18)
    X_flat = np.stack([features_dict[f] for f in feature_order], axis=1)

    return X_flat, u_out_r


# ==========================================
# Data Processing Pipeline
# ==========================================
def prepare_datasets(debug=False, load_cached_data=True):
    """
    Loads data, performs feature engineering, scaling, and returns PyTorch Datasets.
    Implements caching to avoid re-processing.
    """

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_processed.npz")
    val_cache_path = os.path.join(cache_dir, "val_processed.npz")
    test_cache_path = os.path.join(cache_dir, "test_processed.npz")
    scaler_path = os.path.join(cache_dir, "scaler_params.npz")

    # --- Attempt to Load Cache ---
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print("Loading cached datasets...")
        train_data = np.load(train_cache_path)
        val_data = np.load(val_cache_path)
        test_data = np.load(test_cache_path)

        train_dataset = VentilatorDataset(
            train_data["X"], train_data["u_out"], train_data["y"]
        )
        val_dataset = VentilatorDataset(val_data["X"], val_data["u_out"], val_data["y"])
        test_dataset = VentilatorDataset(
            test_data["X"], test_data["u_out"], None
        )  # Test has no y

        return train_dataset, val_dataset, test_dataset

    print("Cache not found or reload requested. Processing data from scratch...")

    # --- Load Metadata ---
    print("Loading metadata...")
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)

    train_breath_ids = set(train_meta["breath_id"].unique())
    val_breath_ids = set(val_meta["breath_id"].unique())

    # --- Load Raw Data ---
    print(f"Loading raw train data from {Config.TRAIN_CSV}...")
    df_full_train = pd.read_csv(Config.TRAIN_CSV)

    # Split into Train and Val
    print("Splitting train/val...")
    df_train = df_full_train[df_full_train["breath_id"].isin(train_breath_ids)].copy()
    df_val = df_full_train[df_full_train["breath_id"].isin(val_breath_ids)].copy()

    del df_full_train  # Free memory

    # Debug Subsampling
    if debug:
        print(f"Debug mode: Subsampling {Config.DEBUG_SIZE} breaths...")
        debug_ids = df_train["breath_id"].unique()[: Config.DEBUG_SIZE]
        df_train = df_train[df_train["breath_id"].isin(debug_ids)].copy()

        debug_val_ids = df_val["breath_id"].unique()[: Config.DEBUG_SIZE]
        df_val = df_val[df_val["breath_id"].isin(debug_val_ids)].copy()

    # Sort to ensure correct reshaping
    df_train.sort_values(["breath_id", "id"], inplace=True)
    df_val.sort_values(["breath_id", "id"], inplace=True)

    # --- Feature Engineering (Train) ---
    print("Engineering features for Training set...")
    X_train_flat, u_out_train_r = compute_physics_features(df_train)
    y_train_flat = df_train["pressure"].values

    # Fit Scaler on Train
    print("Fitting RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)

    # Save Scaler Params manually to avoid pickle
    np.savez(scaler_path, center=scaler.center_, scale=scaler.scale_)

    # Reshape Train to (N, 80, F)
    n_train = len(df_train) // 80
    X_train = X_train_scaled_flat.reshape(n_train, 80, -1)
    y_train = y_train_flat.reshape(n_train, 80)

    # --- Feature Engineering (Val) ---
    print("Engineering features for Validation set...")
    X_val_flat, u_out_val_r = compute_physics_features(df_val)
    y_val_flat = df_val["pressure"].values

    # Transform Val
    X_val_scaled_flat = scaler.transform(X_val_flat)

    # Reshape Val
    n_val = len(df_val) // 80
    X_val = X_val_scaled_flat.reshape(n_val, 80, -1)
    y_val = y_val_flat.reshape(n_val, 80)

    # --- Feature Engineering (Test) ---
    print(f"Loading raw test data from {Config.TEST_CSV}...")
    df_test = pd.read_csv(Config.TEST_CSV)

    # Sort Test
    df_test.sort_values(["breath_id", "id"], inplace=True)

    print("Engineering features for Test set...")
    X_test_flat, u_out_test_r = compute_physics_features(df_test)

    # Transform Test
    X_test_scaled_flat = scaler.transform(X_test_flat)

    # Reshape Test
    n_test = len(df_test) // 80
    X_test = X_test_scaled_flat.reshape(n_test, 80, -1)

    # --- Save to Cache ---
    print("Saving processed datasets to cache...")
    np.savez(train_cache_path, X=X_train, u_out=u_out_train_r, y=y_train)
    np.savez(val_cache_path, X=X_val, u_out=u_out_val_r, y=y_val)
    np.savez(test_cache_path, X=X_test, u_out=u_out_test_r)

    # --- Create Datasets ---
    train_dataset = VentilatorDataset(X_train, u_out_train_r, y_train)
    val_dataset = VentilatorDataset(X_val, u_out_val_r, y_val)
    test_dataset = VentilatorDataset(X_test, u_out_test_r, None)

    print(
        f"Data processing complete. Train shape: {X_train.shape}, Val shape: {X_val.shape}"
    )
    return train_dataset, val_dataset, test_dataset
