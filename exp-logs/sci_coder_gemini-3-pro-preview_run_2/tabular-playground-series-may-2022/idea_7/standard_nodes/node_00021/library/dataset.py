import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import ModelConfig


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    """

    def __init__(self, cont_features, cat_features, targets=None):
        """
        Args:
            cont_features (np.ndarray): Normalized continuous features (N, 30).
            cat_features (np.ndarray): Integer-encoded categorical features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.cont_features = torch.FloatTensor(cont_features)
        self.cat_features = torch.LongTensor(cat_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {"cont": self.cont_features[idx], "cat": self.cat_features[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _decompose_f27(series):
    """
    Decomposes the 10-character string 'f_27' into a (N, 10) integer array.
    Maps 'A'->1, 'B'->2, ..., 'Z'->26.
    """
    # Convert series to list of strings, then to list of lists of ordinals
    # ord('A') is 65. We want 'A' -> 1. So ord(c) - 64.
    # We assume the string is always length 10 and contains uppercase A-Z.

    # Efficient vectorized approach using pandas split and apply is okay,
    # but list comprehension is often faster for string ops in python
    chars = [list(s) for s in series]
    # Map to integers
    # We use a fixed offset.
    # ord('A') = 65. 65 - 64 = 1.
    int_mapped = [[ord(c) - 64 for c in row] for row in chars]
    return np.array(int_mapped, dtype=np.int32)


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: Dictionary containing numpy arrays for X_train_cont, X_train_cat, y_train, etc.
    """
    cache_path = ModelConfig.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            data_dict = dict(data)
            required_keys = [
                "X_train_cont",
                "X_train_cat",
                "y_train",
                "X_val_cont",
                "X_val_cat",
                "y_val",
                "X_test_cont",
                "X_test_cat",
                "test_ids",
            ]
            if all(key in data_dict for key in required_keys):
                return data_dict
            print("Cache missing required keys. Re-processing data.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data.")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(ModelConfig.TRAIN_META)
    val_meta = pd.read_csv(ModelConfig.VAL_META)
    test_meta = pd.read_csv(ModelConfig.TEST_META)

    train_ids = set(train_meta["id"])
    val_ids = set(val_meta["id"])
    # Test IDs are in test.csv

    # 3. Load Raw Data
    # train.csv contains both train and val samples
    df_train_full = pd.read_csv(ModelConfig.TRAIN_CSV)
    df_test = pd.read_csv(ModelConfig.TEST_CSV)

    # 4. Split Train/Val based on ID
    # We map IDs to boolean masks for speed
    # Note: df_train_full has 'id' column
    is_train = df_train_full["id"].isin(train_ids)
    is_val = df_train_full["id"].isin(val_ids)

    df_train = df_train_full[is_train].copy()
    df_val = df_train_full[is_val].copy()

    # Ensure sorting aligns with metadata if necessary, but usually ID match is enough.
    # To be safe and deterministic, we sort by ID.
    df_train = df_train.sort_values("id").reset_index(drop=True)
    df_val = df_val.sort_values("id").reset_index(drop=True)
    df_test = df_test.sort_values("id").reset_index(drop=True)

    # 5. Feature Engineering

    # Identify Continuous Columns: f_00 to f_30, excluding f_27
    # All columns starting with f_, excluding f_27
    all_cols = df_train.columns.tolist()
    cont_cols = [c for c in all_cols if c.startswith("f_") and c != "f_27"]

    # Extract Continuous Features
    X_train_cont = df_train[cont_cols].values.astype(np.float32)
    X_val_cont = df_val[cont_cols].values.astype(np.float32)
    X_test_cont = df_test[cont_cols].values.astype(np.float32)

    # Normalize Continuous Features
    # Fit ONLY on training set
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_cont)
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # Extract Categorical Feature (f_27)
    X_train_cat = _decompose_f27(df_train["f_27"])
    X_val_cat = _decompose_f27(df_val["f_27"])
    X_test_cat = _decompose_f27(df_test["f_27"])

    # Extract Targets
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)

    # Extract Test IDs for submission
    test_ids = df_test["id"].values

    # 6. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    arrays = {
        "X_train_cont": X_train_cont,
        "X_train_cat": X_train_cat,
        "y_train": y_train,
        "X_val_cont": X_val_cont,
        "X_val_cat": X_val_cat,
        "y_val": y_val,
        "X_test_cont": X_test_cont,
        "X_test_cat": X_test_cat,
        "test_ids": test_ids,
    }

    np.savez_compressed(cache_path, **arrays)
    print(f"Data processed and saved to {cache_path}")

    return arrays


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to load from cache.
        debug (bool): If True, subsets data for rapid prototyping.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    data = prepare_data(load_cached_data=load_cached_data)

    X_train_cont = data["X_train_cont"]
    X_train_cat = data["X_train_cat"]
    y_train = data["y_train"]

    X_val_cont = data["X_val_cont"]
    X_val_cat = data["X_val_cat"]
    y_val = data["y_val"]

    X_test_cont = data["X_test_cont"]
    X_test_cat = data["X_test_cat"]
    test_ids = data["test_ids"]

    # Debugging: Subset data
    if debug:
        subset_size = ModelConfig.DEBUG_SAMPLE_SIZE
        print(f"Debug mode: Subsetting data to {subset_size} samples.")
        X_train_cont = X_train_cont[:subset_size]
        X_train_cat = X_train_cat[:subset_size]
        y_train = y_train[:subset_size]

        X_val_cont = X_val_cont[:subset_size]
        X_val_cat = X_val_cat[:subset_size]
        y_val = y_val[:subset_size]

        # Keep test set intact or subset? Usually subset for debug speed
        X_test_cont = X_test_cont[:subset_size]
        X_test_cat = X_test_cat[:subset_size]
        test_ids = test_ids[:subset_size]

    # Create Datasets
    train_dataset = ManufacturingDataset(X_train_cont, X_train_cat, y_train)
    val_dataset = ManufacturingDataset(X_val_cont, X_val_cat, y_val)
    test_dataset = ManufacturingDataset(X_test_cont, X_test_cat, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=ModelConfig.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=ModelConfig.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=ModelConfig.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
