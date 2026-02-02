import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    """

    def __init__(self, X_cat, X_cont, y=None):
        """
        Args:
            X_cat (np.ndarray): Decomposed categorical features (N, 10).
            X_cont (np.ndarray): Normalized continuous features (N, 30).
            y (np.ndarray, optional): Target labels (N,).
        """
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)

        if y is not None:
            # Reshape to (N, 1) for compatibility with BCEWithLogitsLoss
            self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        item = {"cat": self.X_cat[idx], "cont": self.X_cont[idx]}
        if self.y is not None:
            item["target"] = self.y[idx]
        return item


def decompose_f_27(series):
    """
    Decomposes the string feature 'f_27' into 10 integer-encoded columns.
    Assumes characters are A-Z (mapped to 0-25).
    """
    # Vectorized list comprehension is efficient for string processing
    # ord('A') is 65
    chars = [[ord(c) - 65 for c in s] for s in series.values]
    return np.array(chars, dtype=np.int64)


def normalize_continuous(X_train, X_val, X_test):
    """
    Applies Z-score normalization based on training statistics.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Prevent division by zero
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches the dataset.

    Returns:
        Tuple of numpy arrays: (X_cat_train, X_cont_train, y_train,
                                X_cat_val, X_cont_val, y_val,
                                X_cat_test, X_cont_test)
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached processed data from {cache_path}")
        try:
            data = np.load(cache_path)
            return (
                data["X_cat_train"],
                data["X_cont_train"],
                data["y_train"],
                data["X_cat_val"],
                data["X_cont_val"],
                data["y_val"],
                data["X_cat_test"],
                data["X_cont_test"],
            )
        except Exception as e:
            print(f"Cache load failed: {e}. Reprocessing from scratch.")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # Load Raw Data
    print("Loading raw CSV files...")
    df_train_raw = pd.read_csv(Config.TRAIN_CSV)
    df_test_raw = pd.read_csv(Config.TEST_CSV)

    # Merge to create splits based on metadata IDs
    # We use left join on metadata to preserve the specific stratified split
    print("Merging data with metadata...")
    df_train = train_meta.merge(df_train_raw, on="id", how="left")
    df_val = val_meta.merge(df_train_raw, on="id", how="left")
    df_test = test_meta.merge(df_test_raw, on="id", how="left")

    # Extract Targets (from metadata to be safe, though raw has it too)
    # 'target_x' comes from metadata, 'target_y' from raw. They should be identical.
    y_train = df_train["target_x"].values.astype(np.float32)
    y_val = df_val["target_x"].values.astype(np.float32)

    # Extract and Normalize Continuous Features
    cont_cols = Config.CONT_FEATURES
    print("Processing continuous features...")
    X_cont_train = df_train[cont_cols].values.astype(np.float32)
    X_cont_val = df_val[cont_cols].values.astype(np.float32)
    X_cont_test = df_test[cont_cols].values.astype(np.float32)

    X_cont_train, X_cont_val, X_cont_test = normalize_continuous(
        X_cont_train, X_cont_val, X_cont_test
    )

    # Extract and Decompose Categorical Features
    cat_col = Config.CAT_FEATURE
    print("Processing categorical features...")
    X_cat_train = decompose_f_27(df_train[cat_col])
    X_cat_val = decompose_f_27(df_val[cat_col])
    X_cat_test = decompose_f_27(df_test[cat_col])

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        X_cat_train=X_cat_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_cat_val=X_cat_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_cat_test=X_cat_test,
        X_cont_test=X_cont_test,
    )

    return (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
    )


def get_dataloaders(batch_size=None, load_cached_data=True, debug_samples=None):
    """
    Creates and returns PyTorch DataLoaders for the experiment.

    Args:
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug_samples (int, optional): Limit number of samples for debugging.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Load Data
    data = process_data(load_cached_data=load_cached_data)
    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
    ) = data

    # Debug Subsampling
    if debug_samples is not None:
        print(f"DEBUG MODE: Limiting dataset to {debug_samples} samples.")
        X_cat_train = X_cat_train[:debug_samples]
        X_cont_train = X_cont_train[:debug_samples]
        y_train = y_train[:debug_samples]

        X_cat_val = X_cat_val[:debug_samples]
        X_cont_val = X_cont_val[:debug_samples]
        y_val = y_val[:debug_samples]

        X_cat_test = X_cat_test[:debug_samples]
        X_cont_test = X_cont_test[:debug_samples]

    # Instantiate Datasets
    train_ds = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_ds = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    test_ds = ManufacturingDataset(X_cat_test, X_cont_test, None)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm stats
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
