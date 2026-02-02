import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.utils import feature_engineering


class ForestDataset(Dataset):
    """
    Custom Dataset for Forest Cover Type prediction.
    Wraps numpy arrays and converts them to PyTorch tensors.
    """

    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long() if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return (self.X[idx],)


def engineer_features(df):
    """
    Applies Augmented Physics-Informed Engineering.
    Wraps the utility function to ensure consistency.
    """
    return feature_engineering(df)


def get_dataloaders(
    load_cached_data=True,
    batch_size=4096,
    data_dir="./metadata",
    cache_dir="./working/idea_29",
):
    """
    Loads data, performs feature engineering, handles caching, and returns DataLoaders.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    cache_files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        X_test = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
    else:
        print("Processing data from scratch...")
        # Load Parquet
        train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

        # Extract IDs and Targets
        # Targets are 1-7, convert to 0-6 for PyTorch
        y_train = train_df["Cover_Type"].values - 1
        y_val = val_df["Cover_Type"].values - 1
        test_ids = test_df["Id"].values.astype(np.int64)

        # Drop Id and Target from features
        train_df = train_df.drop(columns=["Id", "Cover_Type"], errors="ignore")
        val_df = val_df.drop(columns=["Id", "Cover_Type"], errors="ignore")
        test_df = test_df.drop(columns=["Id"], errors="ignore")

        # Feature Engineering
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Identify Columns
        # Binary features: Soil_Type*, Wilderness_Area*
        bin_cols = [
            c for c in train_df.columns if "Soil_Type" in c or "Wilderness_Area" in c
        ]
        cont_cols = [c for c in train_df.columns if c not in bin_cols]

        # Standardization (Fit on Train, Transform all)
        scaler = StandardScaler()
        train_df[cont_cols] = scaler.fit_transform(
            train_df[cont_cols].astype(np.float32)
        )
        val_df[cont_cols] = scaler.transform(val_df[cont_cols].astype(np.float32))
        test_df[cont_cols] = scaler.transform(test_df[cont_cols].astype(np.float32))

        # Convert to Numpy (Float32)
        # Ensure column order is identical
        cols = cont_cols + bin_cols
        X_train = train_df[cols].values.astype(np.float32)
        X_val = val_df[cols].values.astype(np.float32)
        X_test = test_df[cols].values.astype(np.float32)

        # Save to Cache
        np.save(cache_files["train_X"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["val_X"], X_val)
        np.save(cache_files["val_y"], y_val)
        np.save(cache_files["test_X"], X_test)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = ForestDataset(X_train, y_train)
    val_dataset = ForestDataset(X_val, y_val)
    test_dataset = ForestDataset(X_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids, X_train.shape[1]
