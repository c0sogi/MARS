import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.features import engineer_features
from library.utils import seed_everything


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Returns data in the format: (features, target, mask)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, u_out: np.ndarray):
        """
        Args:
            X (np.ndarray): Scaled features of shape (num_breaths, seq_len, num_features)
            y (np.ndarray): Target pressure of shape (num_breaths, seq_len)
            u_out (np.ndarray): Raw u_out mask of shape (num_breaths, seq_len)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.u_out[idx]


def prepare_data(config: Config):
    """
    Orchestrates data loading, feature engineering, scaling, and DataLoader creation.
    Implements the Dual-Stream logic (Scaled Features + Raw Mask) and caching.

    Args:
        config (Config): Configuration object.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(config.seed)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(config.cache_dir, "train_X.npy"),
        "train_y": os.path.join(config.cache_dir, "train_y.npy"),
        "train_u_out": os.path.join(config.cache_dir, "train_u_out.npy"),
        "val_X": os.path.join(config.cache_dir, "val_X.npy"),
        "val_y": os.path.join(config.cache_dir, "val_y.npy"),
        "val_u_out": os.path.join(config.cache_dir, "val_u_out.npy"),
        "test_X": os.path.join(config.cache_dir, "test_X.npy"),
        "test_ids": os.path.join(
            config.cache_dir, "test_ids.npy"
        ),  # Needed for submission
        "test_u_out": os.path.join(config.cache_dir, "test_u_out.npy"),
    }

    # Check if we can load from cache
    all_cached = all(
        os.path.exists(p) for p in cache_files.values()
    ) and os.path.exists(config.scaler_save_path)

    if config.load_cached_data and all_cached:
        print("Loading processed data from cache...")
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        train_u_out = np.load(cache_files["train_u_out"])

        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        val_u_out = np.load(cache_files["val_u_out"])

        test_X = np.load(cache_files["test_X"])
        # test_ids and test_u_out are loaded when needed or implicitly handled if we returned them,
        # but for DataLoaders we just need X. We'll load test_u_out for consistency.
        test_u_out = np.load(cache_files["test_u_out"])

    else:
        print("Processing data from scratch...")

        # 1. Load Data
        train_df = pd.read_csv(config.train_file)
        val_df = pd.read_csv(config.val_file)
        test_df = pd.read_csv(config.test_file)

        if config.debug:
            print("Debug mode: Subsampling data...")
            train_breaths = train_df["breath_id"].unique()[:100]
            val_breaths = val_df["breath_id"].unique()[:50]
            test_breaths = test_df["breath_id"].unique()[:50]

            train_df = train_df[train_df["breath_id"].isin(train_breaths)].copy()
            val_df = val_df[val_df["breath_id"].isin(val_breaths)].copy()
            test_df = test_df[test_df["breath_id"].isin(test_breaths)].copy()

        # 2. Feature Engineering
        train_df = engineer_features(train_df, "train", config)
        val_df = engineer_features(val_df, "val", config)
        test_df = engineer_features(test_df, "test", config)

        # 3. Prepare Targets and Masks
        # Target: Pressure
        train_y_flat = train_df["pressure"].values
        val_y_flat = val_df["pressure"].values
        # Test has no pressure, fill with 0
        test_y_flat = np.zeros(len(test_df))

        # Stream B: Raw Logic Gate (u_out)
        train_u_out_flat = train_df["u_out"].values
        val_u_out_flat = val_df["u_out"].values
        test_u_out_flat = test_df["u_out"].values

        # Save test IDs for submission
        test_ids = test_df["id"].values
        np.save(cache_files["test_ids"], test_ids)

        # 4. Prepare Stream A (Features)
        # Drop non-feature columns
        drop_cols = ["id", "breath_id", "pressure"]
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

        print(f"Features used ({len(feature_cols)}): {feature_cols}")

        train_X_flat = train_df[feature_cols].values
        val_X_flat = val_df[feature_cols].values
        test_X_flat = test_df[feature_cols].values

        # Scaling
        print("Fitting RobustScaler...")
        scaler = RobustScaler()
        train_X_flat = scaler.fit_transform(train_X_flat)
        val_X_flat = scaler.transform(val_X_flat)
        test_X_flat = scaler.transform(test_X_flat)

        # Save Scaler
        joblib.dump(scaler, config.scaler_save_path)

        # 5. Reshape to (Num_Breaths, 80, Num_Features)
        # We assume each breath is exactly 80 steps
        SEQ_LEN = 80

        def reshape_series(x_flat, seq_len):
            num_samples = x_flat.shape[0] // seq_len
            if x_flat.ndim == 1:
                return x_flat.reshape(num_samples, seq_len)
            else:
                return x_flat.reshape(num_samples, seq_len, x_flat.shape[1])

        train_X = reshape_series(train_X_flat, SEQ_LEN)
        train_y = reshape_series(train_y_flat, SEQ_LEN)
        train_u_out = reshape_series(train_u_out_flat, SEQ_LEN)

        val_X = reshape_series(val_X_flat, SEQ_LEN)
        val_y = reshape_series(val_y_flat, SEQ_LEN)
        val_u_out = reshape_series(val_u_out_flat, SEQ_LEN)

        test_X = reshape_series(test_X_flat, SEQ_LEN)
        test_u_out = reshape_series(test_u_out_flat, SEQ_LEN)

        # 6. Cache Data
        print("Caching processed arrays...")
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["train_u_out"], train_u_out)

        np.save(cache_files["val_X"], val_X)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["val_u_out"], val_u_out)

        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["test_u_out"], test_u_out)

    # Create Datasets
    train_dataset = VentilatorDataset(train_X, train_y, train_u_out)
    val_dataset = VentilatorDataset(val_X, val_y, val_u_out)
    # For test, we create a dummy target array
    test_y_dummy = np.zeros((test_X.shape[0], test_X.shape[1]))
    test_dataset = VentilatorDataset(test_X, test_y_dummy, test_u_out)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(
        f"Data prepared. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
