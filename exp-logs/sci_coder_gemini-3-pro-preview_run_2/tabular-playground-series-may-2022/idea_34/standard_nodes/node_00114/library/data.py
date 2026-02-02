import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Serves dual-view data: categorical sequence (f_27) and continuous features.
    """

    def __init__(self, cat_data, cont_data, targets=None):
        """
        Args:
            cat_data (np.ndarray): Categorical features, shape (N, 10).
            cont_data (np.ndarray): Continuous features, shape (N, 30).
            targets (np.ndarray, optional): Target labels, shape (N,).
        """
        self.cat_data = torch.from_numpy(cat_data).long()
        self.cont_data = torch.from_numpy(cont_data).float()

        if targets is not None:
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        cat = self.cat_data[idx]
        cont = self.cont_data[idx]

        if self.targets is not None:
            target = self.targets[idx]
            return cat, cont, target
        else:
            return cat, cont


def process_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (normalization, encoding),
    and caches the result to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached processed data from {cache_path}")
        # Load and return as a dictionary to ensure data is in memory
        with np.load(cache_path) as data:
            return {key: data[key] for key in data.files}

    print("Cache not found or reload requested. Processing data from scratch...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 2. Load Metadata
    # Metadata defines the exact split and ordering
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Load Raw Data
    # Reading full CSVs
    df_raw_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_raw_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 4. Merge to align with Metadata
    # We use left merge on metadata to select and order the rows correctly
    # Suffixes handle potential column name collisions (e.g. target in both)
    df_train = train_meta.merge(
        df_raw_train, on="id", how="left", suffixes=("", "_raw")
    )
    df_val = val_meta.merge(df_raw_train, on="id", how="left", suffixes=("", "_raw"))
    df_test = test_meta.merge(df_raw_test, on="id", how="left", suffixes=("", "_raw"))

    # 5. Preprocessing

    # A. Continuous Features (f_00 ... f_30, excluding f_27)
    # Identify continuous columns
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Initialize and fit scaler on Training data ONLY
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    # Transform all splits
    X_cont_train = scaler.transform(df_train[cont_cols]).astype(np.float32)
    X_cont_val = scaler.transform(df_val[cont_cols]).astype(np.float32)
    X_cont_test = scaler.transform(df_test[cont_cols]).astype(np.float32)

    # B. Categorical Feature (f_27)
    # Map characters A-Z to integers 0-25
    def encode_f27(series):
        # Vectorized list comprehension for speed
        # ord('A') is 65
        return np.array([[ord(c) - 65 for c in s] for s in series], dtype=np.int64)

    X_cat_train = encode_f27(df_train["f_27"])
    X_cat_val = encode_f27(df_val["f_27"])
    X_cat_test = encode_f27(df_test["f_27"])

    # C. Targets
    # Metadata guarantees 'target' column exists for train/val
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)

    # 6. Save to Cache
    data_dict = {
        "X_cat_train": X_cat_train,
        "X_cont_train": X_cont_train,
        "y_train": y_train,
        "X_cat_val": X_cat_val,
        "X_cont_val": X_cont_val,
        "y_val": y_val,
        "X_cat_test": X_cat_test,
        "X_cont_test": X_cont_test,
    }

    np.savez(cache_path, **data_dict)
    print(f"Data processed and saved to {cache_path}")

    return data_dict


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets the data for faster iteration.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load data
    data = process_data(load_cached_data=True)

    X_cat_train = data["X_cat_train"]
    X_cont_train = data["X_cont_train"]
    y_train = data["y_train"]

    X_cat_val = data["X_cat_val"]
    X_cont_val = data["X_cont_val"]
    y_val = data["y_val"]

    X_cat_test = data["X_cat_test"]
    X_cont_test = data["X_cont_test"]

    # Handle Debug Mode
    if debug:
        limit = Config.DEBUG_SAMPLE_SIZE
        print(f"DEBUG MODE: Truncating datasets to {limit} samples.")
        X_cat_train = X_cat_train[:limit]
        X_cont_train = X_cont_train[:limit]
        y_train = y_train[:limit]

        X_cat_val = X_cat_val[:limit]
        X_cont_val = X_cont_val[:limit]
        y_val = y_val[:limit]

        X_cat_test = X_cat_test[:limit]
        X_cont_test = X_cont_test[:limit]

    # Instantiate Datasets
    train_dataset = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = ManufacturingDataset(X_cat_test, X_cont_test, targets=None)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
