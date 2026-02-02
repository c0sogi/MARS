import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    def __init__(self, continuous_data, sequence_data, targets=None, ids=None):
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.sequence_data = torch.LongTensor(sequence_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.ids = ids

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "sequence": self.sequence_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.ids is not None:
            item["id"] = self.ids[idx]
        return item


def _decompose_f27(series):
    """
    Decomposes the string feature f_27 into a (N, 10) integer array.
    Maps 'A' -> 0, ..., 'Z' -> 25.
    """
    # Convert series to list of strings, then to numpy array of ASCII values
    # We assume fixed length of 10.
    # Faster approach: View as int8 (ASCII)
    # Each char is 1 byte. 'A' is 65.

    # Ensure series is string type
    s_values = series.astype(str).values

    # Create a buffer.
    # Note: This assumes pure ASCII and fixed length.
    # A safe vectorized way using list comprehension if length varies,
    # but here length is fixed 10.
    # We will use a robust list comprehension method to ensure correctness.

    # Map A-Z to 0-25
    base = ord("A")

    # Pre-allocate
    n = len(s_values)
    seq_len = Config.SEQ_LEN
    out = np.zeros((n, seq_len), dtype=np.int32)

    for i, s in enumerate(s_values):
        # Slicing to 10 just in case, though data is clean
        chars = [ord(c) - base for c in s[:seq_len]]
        out[i] = chars

    return out


def process_data(load_cached_data=True):
    """
    Loads raw data, processes features, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Dictionary containing processed numpy arrays for train, val, test.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading processed data from {Config.PROCESSED_DATA_PATH}...")
        try:
            return np.load(Config.PROCESSED_DATA_PATH)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # 2. Load Raw Data
    # We read the full files.
    # Note: train.csv contains both train and val samples.
    # test.csv contains test samples.
    raw_train = pd.read_csv(Config.TRAIN_CSV)
    raw_test = pd.read_csv(Config.TEST_CSV)

    # 3. Merge/Filter based on Metadata
    # We use 'id' to map.
    # Create indexed dataframes for fast lookup
    raw_train.set_index("id", inplace=True)
    raw_test.set_index("id", inplace=True)

    # Extract subsets
    # reindex ensures we get the rows in the exact order of metadata
    X_train_raw = raw_train.reindex(train_meta["id"])
    X_val_raw = raw_train.reindex(val_meta["id"])
    X_test_raw = raw_test.reindex(test_meta["id"])

    # 4. Feature Definition
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    seq_col = "f_27"
    target_col = "target"

    # 5. Process Continuous Features
    print("Normalizing continuous features...")
    scaler = StandardScaler()

    # Fit only on training set
    X_cont_train = scaler.fit_transform(
        X_train_raw[cont_cols].values.astype(np.float32)
    )

    # Transform others
    X_cont_val = scaler.transform(X_val_raw[cont_cols].values.astype(np.float32))
    X_cont_test = scaler.transform(X_test_raw[cont_cols].values.astype(np.float32))

    # 6. Process Sequence Features
    print("Decomposing sequence features...")
    X_seq_train = _decompose_f27(X_train_raw[seq_col])
    X_seq_val = _decompose_f27(X_val_raw[seq_col])
    X_seq_test = _decompose_f27(X_test_raw[seq_col])

    # 7. Extract Targets and IDs
    y_train = X_train_raw[target_col].values.astype(np.float32)
    y_val = X_val_raw[target_col].values.astype(np.float32)
    # Test set has no target

    ids_train = train_meta["id"].values
    ids_val = val_meta["id"].values
    ids_test = test_meta["id"].values

    # 8. Save to Cache
    data_dict = {
        "X_cont_train": X_cont_train,
        "X_seq_train": X_seq_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_cont_val": X_cont_val,
        "X_seq_val": X_seq_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_cont_test": X_cont_test,
        "X_seq_test": X_seq_test,
        "ids_test": ids_test,
    }

    print(f"Saving processed data to {Config.PROCESSED_DATA_PATH}...")
    np.savez_compressed(Config.PROCESSED_DATA_PATH, **data_dict)

    return data_dict


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    data = process_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ManufacturingDataset(
        continuous_data=data["X_cont_train"],
        sequence_data=data["X_seq_train"],
        targets=data["y_train"],
        ids=data["ids_train"],
    )

    val_dataset = ManufacturingDataset(
        continuous_data=data["X_cont_val"],
        sequence_data=data["X_seq_val"],
        targets=data["y_val"],
        ids=data["ids_val"],
    )

    test_dataset = ManufacturingDataset(
        continuous_data=data["X_cont_test"],
        sequence_data=data["X_seq_test"],
        targets=None,
        ids=data["ids_test"],
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
