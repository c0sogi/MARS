import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles continuous features, sequence features (f_27), and targets.
    """

    def __init__(self, continuous_data, sequence_data, targets=None, is_test=False):
        """
        Args:
            continuous_data (np.ndarray): Normalized continuous features (N, 30).
            sequence_data (np.ndarray): Tokenized f_27 sequence (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
            is_test (bool): Flag indicating if this is a test set (no targets).
        """
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.sequence_data = torch.LongTensor(sequence_data)
        self.is_test = is_test

        if not self.is_test:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "sequence": self.sequence_data[idx],
        }

        if not self.is_test:
            item["target"] = self.targets[idx]

        return item


def _tokenize_f27(series):
    """
    Converts a pandas Series of strings (length 10) into a numpy array of shape (N, 10).
    Maps 'A'->1, 'B'->2, ..., 'Z'->26.
    """
    # Convert series to list of strings, then to list of list of chars
    # This is generally faster than apply for large datasets
    # We assume all strings are length 10 and uppercase A-Z based on EDA

    # Create a mapping array for fast lookup
    # ord('A') is 65. We want 'A' -> 1. So x - 64.
    # We can vectorize this using numpy view of the string buffer

    # Convert to numpy array of strings
    arr = series.values.astype(str)

    # View as uint8 (bytes)
    # This creates a 2D array where columns are characters
    # Note: This works for fixed-width ASCII strings.
    # 'S10' means string of length 10.
    arr_view = arr.astype("S10").view("S1").reshape(len(arr), 10)

    # Convert bytes to integers and shift to 1-based index (A=65 -> 1)
    # We cast to uint8 first to get ASCII codes
    tokenized = arr_view.view(np.uint8).astype(np.int64) - 64

    return tokenized


def process_data(load_cached_data=True):
    """
    Loads raw data, performs splitting based on metadata, normalizes continuous features,
    tokenizes sequence features, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from .npz cache first.

    Returns:
        dict: Dictionary containing processed numpy arrays for train, val, and test.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return {
                "train_cont": data["train_cont"],
                "train_seq": data["train_seq"],
                "train_target": data["train_target"],
                "val_cont": data["val_cont"],
                "val_seq": data["val_seq"],
                "val_target": data["val_target"],
                "test_cont": data["test_cont"],
                "test_seq": data["test_seq"],
                "test_ids": data["test_ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw Data
    # We load everything into memory. Given 220GB RAM, this is safe.
    df_train_full = pd.read_csv(Config.TRAIN_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Index raw data by ID for fast lookup
    df_train_full.set_index("id", inplace=True)
    df_test.set_index("id", inplace=True)

    # Extract Subsets based on Metadata IDs
    # Using loc to preserve order and ensure alignment
    train_ids = train_meta["id"].values
    val_ids = val_meta["id"].values
    test_ids = test_meta["id"].values

    df_train = df_train_full.loc[train_ids]
    df_val = df_train_full.loc[val_ids]
    # df_test is already correct, but let's ensure alignment with metadata
    df_test = df_test.loc[test_ids]

    # Define Feature Columns
    # Continuous: f_00 to f_30, excluding f_27
    all_cols = df_train.columns.tolist()
    cont_cols = [
        c for c in all_cols if c != "target" and c != "f_27" and c.startswith("f_")
    ]
    seq_col = "f_27"

    # 3. Normalization (Continuous)
    # Fit scaler ONLY on training data
    print("Normalizing continuous features...")
    train_cont = df_train[cont_cols].values.astype(np.float32)
    val_cont = df_val[cont_cols].values.astype(np.float32)
    test_cont = df_test[cont_cols].values.astype(np.float32)

    mean = np.mean(train_cont, axis=0)
    std = np.std(train_cont, axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    train_cont = (train_cont - mean) / std
    val_cont = (val_cont - mean) / std
    test_cont = (test_cont - mean) / std

    # 4. Tokenization (Sequence f_27)
    print("Tokenizing f_27 sequence...")
    train_seq = _tokenize_f27(df_train[seq_col])
    val_seq = _tokenize_f27(df_val[seq_col])
    test_seq = _tokenize_f27(df_test[seq_col])

    # 5. Targets
    train_target = df_train["target"].values.astype(np.float32)
    val_target = df_val["target"].values.astype(np.float32)

    # 6. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        train_cont=train_cont,
        train_seq=train_seq,
        train_target=train_target,
        val_cont=val_cont,
        val_seq=val_seq,
        val_target=val_target,
        test_cont=test_cont,
        test_seq=test_seq,
        test_ids=test_ids,
    )

    return {
        "train_cont": train_cont,
        "train_seq": train_seq,
        "train_target": train_target,
        "val_cont": val_cont,
        "val_seq": val_seq,
        "val_target": val_target,
        "test_cont": test_cont,
        "test_seq": test_seq,
        "test_ids": test_ids,
    }


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    data = process_data(load_cached_data=load_cached_data)

    train_cont = data["train_cont"]
    train_seq = data["train_seq"]
    train_target = data["train_target"]

    val_cont = data["val_cont"]
    val_seq = data["val_seq"]
    val_target = data["val_target"]

    test_cont = data["test_cont"]
    test_seq = data["test_seq"]
    test_ids = data["test_ids"]

    # Debugging Subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating datasets to {Config.DEBUG_SAMPLES} samples.")
        limit = Config.DEBUG_SAMPLES
        train_cont = train_cont[:limit]
        train_seq = train_seq[:limit]
        train_target = train_target[:limit]

        val_cont = val_cont[:limit]
        val_seq = val_seq[:limit]
        val_target = val_target[:limit]

        test_cont = test_cont[:limit]
        test_seq = test_seq[:limit]
        test_ids = test_ids[:limit]

    # Create Datasets
    train_dataset = ManufacturingDataset(
        train_cont, train_seq, train_target, is_test=False
    )
    val_dataset = ManufacturingDataset(val_cont, val_seq, val_target, is_test=False)
    test_dataset = ManufacturingDataset(test_cont, test_seq, targets=None, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, test_ids
