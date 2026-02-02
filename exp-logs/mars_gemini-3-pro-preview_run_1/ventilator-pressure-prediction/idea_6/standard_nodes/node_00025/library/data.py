import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import get_data_hash, save_npy, load_npy, set_seed


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Serves (X, y, u_out) tuples for training/validation, or (X, u_out) for testing.
    """

    def __init__(self, X, u_out, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.u_out[idx]
        else:
            return self.X[idx], self.u_out[idx]


class DataProcessor:
    """
    Handles data loading, feature engineering, scaling, and caching.
    """

    def __init__(self, config=Config):
        self.config = config
        self.features = config.FEATURES
        self.seq_len = config.SEQ_LEN

    def get_cache_paths(self, split):
        """Generates file paths for cached data based on config hash."""
        data_hash = get_data_hash(self.config)
        base_dir = self.config.WORKING_DIR

        paths = {
            "X": os.path.join(base_dir, f"{split}_{data_hash}_X.npy"),
            "ids": os.path.join(base_dir, f"{split}_{data_hash}_ids.npy"),
            "u_out": os.path.join(base_dir, f"{split}_{data_hash}_u_out.npy"),
        }
        if split != "test":
            paths["y"] = os.path.join(base_dir, f"{split}_{data_hash}_y.npy")

        return paths

    def _engineer_features_vectorized(self, df):
        """
        Performs vectorized feature engineering on the dataframe.
        Assumes data is sorted by breath_id and time_step, with fixed sequence length.
        """
        n_breaths = len(df) // self.seq_len

        # Extract raw arrays and reshape to (N_breaths, 80)
        u_in = df["u_in"].values.reshape(n_breaths, self.seq_len)
        u_out = df["u_out"].values.reshape(n_breaths, self.seq_len)
        time_step = df["time_step"].values.reshape(n_breaths, self.seq_len)
        R = df["R"].values.reshape(n_breaths, self.seq_len)
        C = df["C"].values.reshape(n_breaths, self.seq_len)

        # --- Dynamics ---
        # Lags: Pad with 0 at the beginning of each breath
        # np.hstack concatenates along the second dimension (time)
        zeros_1 = np.zeros((n_breaths, 1))
        zeros_2 = np.zeros((n_breaths, 2))
        zeros_3 = np.zeros((n_breaths, 3))
        zeros_4 = np.zeros((n_breaths, 4))

        u_in_lag1 = np.hstack([zeros_1, u_in[:, :-1]])
        u_in_lag2 = np.hstack([zeros_2, u_in[:, :-2]])
        u_in_lag3 = np.hstack([zeros_3, u_in[:, :-3]])
        u_in_lag4 = np.hstack([zeros_4, u_in[:, :-4]])

        # Diffs
        u_in_diff1 = u_in - u_in_lag1
        u_in_diff2 = u_in_diff1 - np.hstack([zeros_1, u_in_diff1[:, :-1]])

        # --- Physics ---
        # dt: Time difference
        dt = time_step - np.hstack([zeros_1, time_step[:, :-1]])
        # Fix potential artifact at t=0 (dt should be small or 0, usually 0 is safe for integration start)

        # Area (Cumulative Volume): Integral of u_in * dt
        area = np.cumsum(u_in * dt, axis=1)

        # Interactions
        u_in_R = u_in * R
        area_div_C = area / C

        # Map feature names to computed arrays
        feature_map = {
            "u_in": u_in,
            "u_out": u_out,
            "time_step": time_step,
            "R": R,
            "C": C,
            "u_in_lag1": u_in_lag1,
            "u_in_lag2": u_in_lag2,
            "u_in_lag3": u_in_lag3,
            "u_in_lag4": u_in_lag4,
            "u_in_diff1": u_in_diff1,
            "u_in_diff2": u_in_diff2,
            "area": area,
            "u_in_R": u_in_R,
            "area_div_C": area_div_C,
        }

        # Stack features: (N_breaths, 80, N_features)
        # Ensure order matches Config.FEATURES
        X_list = [feature_map[f] for f in self.features]
        X = np.stack(X_list, axis=-1)

        return X

    def load_data(self, split, load_cached_data=True):
        """
        Loads data for a specific split. Uses caching to avoid re-processing.
        Scales data using RobustScaler (fit on train, transform on val/test).
        """
        paths = self.get_cache_paths(split)

        # Check if all cache files exist
        cache_exists = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and cache_exists:
            print(f"Loading cached {split} data from {self.config.WORKING_DIR}...")
            X = load_npy(paths["X"])
            ids = load_npy(paths["ids"])
            u_out = load_npy(paths["u_out"])
            y = load_npy(paths["y"]) if "y" in paths else None
            return X, y, u_out, ids

        print(f"Processing {split} data from scratch...")

        # Determine CSV path
        if split == "train":
            csv_path = self.config.TRAIN_PATH
        elif split == "val":
            csv_path = self.config.VAL_PATH
        else:
            csv_path = self.config.TEST_PATH

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Debug Mode: Sample subset
        if self.config.DEBUG:
            print(f"DEBUG MODE: Sampling subset of {split} data.")
            unique_breaths = df["breath_id"].unique()[:200]
            df = df[df["breath_id"].isin(unique_breaths)].copy()

        # Extract Targets and IDs before reshaping
        ids = df["id"].values
        u_out_flat = df["u_out"].values

        y = None
        if "pressure" in df.columns:
            y = df["pressure"].values

        # Engineer Features -> (N_breaths, 80, N_feats)
        X = self._engineer_features_vectorized(df)

        # Scaling
        n_breaths, seq_len, n_feats = X.shape
        X_flat = X.reshape(-1, n_feats)

        if split == "train":
            print("Fitting RobustScaler on training data...")
            scaler = RobustScaler()
            X_flat = scaler.fit_transform(X_flat)

            # Save scaler parameters for reproducibility and application to val/test
            save_npy(scaler.center_, self.config.SCALER_CENTER_PATH)
            save_npy(scaler.scale_, self.config.SCALER_SCALE_PATH)
        else:
            print("Applying scaler to data...")
            if not os.path.exists(self.config.SCALER_CENTER_PATH):
                raise FileNotFoundError(
                    "Scaler artifacts not found. Please run training data processing first."
                )

            center = load_npy(self.config.SCALER_CENTER_PATH)
            scale = load_npy(self.config.SCALER_SCALE_PATH)

            scaler = RobustScaler()
            scaler.center_ = center
            scaler.scale_ = scale
            X_flat = scaler.transform(X_flat)

        # Reshape back to sequence format
        X = X_flat.reshape(n_breaths, seq_len, n_feats)

        # Reshape auxiliaries to match (N_breaths, 80)
        u_out_reshaped = u_out_flat.reshape(n_breaths, seq_len)
        ids_reshaped = ids.reshape(n_breaths, seq_len)

        if y is not None:
            y_reshaped = y.reshape(n_breaths, seq_len)
        else:
            y_reshaped = None

        # Save to cache
        print(f"Caching {split} data...")
        save_npy(X, paths["X"])
        save_npy(ids_reshaped, paths["ids"])
        save_npy(u_out_reshaped, paths["u_out"])
        if y_reshaped is not None:
            save_npy(y_reshaped, paths["y"])

        return X, y_reshaped, u_out_reshaped, ids_reshaped


def get_dataloaders(config=Config, load_cached_data=True):
    """
    Creates DataLoaders for training and validation sets.
    """
    processor = DataProcessor(config)

    # Train Data
    X_train, y_train, u_out_train, _ = processor.load_data("train", load_cached_data)
    train_dataset = VentilatorDataset(X_train, u_out_train, y_train)

    # Validation Data
    X_val, y_val, u_out_val, _ = processor.load_data("val", load_cached_data)
    val_dataset = VentilatorDataset(X_val, u_out_val, y_val)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(config=Config, load_cached_data=True):
    """
    Creates DataLoader for the test set and returns associated IDs for submission.
    """
    processor = DataProcessor(config)

    X_test, _, u_out_test, ids_test = processor.load_data("test", load_cached_data)
    test_dataset = VentilatorDataset(X_test, u_out_test, y=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, ids_test
