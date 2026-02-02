import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.utils import TARGET_CLASSES


class ForestDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type data.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray or torch.Tensor): Feature matrix.
            y (np.ndarray or torch.Tensor, optional): Target labels.
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_dataloaders(
    batch_size=1024, num_workers=4, load_cached_data=True, debug_limit=None
):
    """
    Loads data, preprocesses it (scaling numericals, encoding targets),
    caches the result, and returns PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug_limit (int, optional): If set, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define paths
    CACHE_DIR = "./working/idea_2/"
    METADATA_DIR = "./metadata"

    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_files = {
        "train_X": os.path.join(CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),  # Save IDs for submission
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading preprocessed data from cache...")
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        test_X = np.load(cache_files["test_X"])
        # We don't necessarily need to load test_ids here for the loader,
        # but the main loop might need them.
        # For the purpose of returning loaders, we just need X and y.

    else:
        print("Processing data from scratch...")

        # Load Parquet files
        train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

        # Define columns
        id_col = "Id"
        target_col = "Cover_Type"

        # Identify feature columns
        # We exclude Id and Cover_Type.
        # Note: test_df does not have Cover_Type.
        feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

        # Identify Numerical vs Binary columns
        # Based on dataset analysis:
        # Numerical: Elevation, Aspect, Slope, Distances, Hillshades (10 columns)
        # Binary: Wilderness_Area*, Soil_Type* (44 columns)

        num_cols = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]

        # Verify these columns exist
        num_cols = [c for c in num_cols if c in feature_cols]
        bin_cols = [c for c in feature_cols if c not in num_cols]

        print(
            f"Detected {len(num_cols)} numerical and {len(bin_cols)} binary features."
        )

        # Extract features and targets
        # Train
        X_train_num = train_df[num_cols].values.astype(np.float32)
        X_train_bin = train_df[bin_cols].values.astype(np.float32)
        y_train_raw = train_df[target_col].values

        # Val
        X_val_num = val_df[num_cols].values.astype(np.float32)
        X_val_bin = val_df[bin_cols].values.astype(np.float32)
        y_val_raw = val_df[target_col].values

        # Test
        X_test_num = test_df[num_cols].values.astype(np.float32)
        X_test_bin = test_df[bin_cols].values.astype(np.float32)
        test_ids = test_df[id_col].values

        # Scaling Numerical Features
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_val_num = scaler.transform(X_val_num)
        X_test_num = scaler.transform(X_test_num)

        # Concatenate Features
        train_X = np.hstack([X_train_num, X_train_bin])
        val_X = np.hstack([X_val_num, X_val_bin])
        test_X = np.hstack([X_test_num, X_test_bin])

        # Target Mapping
        # Map classes in TARGET_CLASSES to 0..N-1
        class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}

        # Vectorized mapping
        train_y = np.vectorize(class_to_idx.get)(y_train_raw)
        val_y = np.vectorize(class_to_idx.get)(y_val_raw)

        # Save to cache
        print(f"Saving processed data to {CACHE_DIR}...")
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_X"], val_X)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["test_ids"], test_ids)

    # Apply debug limit if requested
    if debug_limit is not None:
        print(f"Subsampling datasets to {debug_limit} samples for debugging.")
        train_X = train_X[:debug_limit]
        train_y = train_y[:debug_limit]
        val_X = val_X[:debug_limit]
        val_y = val_y[:debug_limit]
        test_X = test_X[:debug_limit]

    # Create Datasets
    train_dataset = ForestDataset(train_X, train_y)
    val_dataset = ForestDataset(val_X, val_y)
    test_dataset = ForestDataset(test_X, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
