import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library import config, utils


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Serves continuous features, sequence features, and targets.
    """

    def __init__(self, cont_features, seq_features, targets=None):
        self.cont_features = torch.FloatTensor(cont_features)
        self.seq_features = torch.LongTensor(seq_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {
            "continuous": self.cont_features[idx],
            "sequence": self.seq_features[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _encode_sequence(series):
    """
    Encodes a pandas Series of strings (length 10) into a numpy array of integers.
    Mapping: 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.
    """
    # Convert series to list of strings
    strings = series.values
    # Vectorized conversion: create a 2D array of characters, then map to int
    # Since all strings are length 10, we can just iterate efficiently
    # Using list comprehension is fast enough for ~1M rows
    encoded = np.array([[ord(c) - 64 for c in s] for s in strings], dtype=np.int64)
    return encoded


def process_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (scaling, encoding), and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        dict: A dictionary containing processed numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(config.PROCESSED_DATA_PATH):
        print(f"Loading cached data from {config.PROCESSED_DATA_PATH}...")
        try:
            data = np.load(config.PROCESSED_DATA_PATH)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    print("Processing data from scratch...")

    # 2. Load Raw Data
    print(f"Reading {config.TRAIN_PATH}...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    print(f"Reading {config.TEST_PATH}...")
    test_df = pd.read_csv(config.TEST_PATH)

    # Load Metadata to identify the strict training set for scaler fitting
    print(f"Reading {config.TRAIN_META_PATH}...")
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    train_ids_set = set(train_meta["id"].values)

    # 3. Define Columns
    # Continuous features: f_00 to f_30 (excluding f_27 which is categorical)
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    # 4. Preprocess Continuous Features
    # Fit Scaler ONLY on the training subset (defined by metadata)
    print("Fitting StandardScaler on training subset...")
    scaler = StandardScaler()

    # Filter train_df to get only rows belonging to the training split
    train_subset_mask = train_df["id"].isin(train_ids_set)
    scaler.fit(train_df.loc[train_subset_mask, cont_cols])

    print("Transforming continuous features...")
    train_cont = scaler.transform(train_df[cont_cols]).astype(np.float32)
    test_cont = scaler.transform(test_df[cont_cols]).astype(np.float32)

    # 5. Preprocess Sequence Feature (f_27)
    print("Encoding sequence feature f_27...")
    train_seq = _encode_sequence(train_df["f_27"])
    test_seq = _encode_sequence(test_df["f_27"])

    # 6. Extract Targets and IDs
    train_target = train_df["target"].values.astype(np.float32).reshape(-1, 1)
    train_ids = train_df["id"].values.astype(np.int64)
    test_ids = test_df["id"].values.astype(np.int64)

    # 7. Save to Cache
    processed_data = {
        "train_cont": train_cont,
        "train_seq": train_seq,
        "train_target": train_target,
        "train_ids": train_ids,
        "test_cont": test_cont,
        "test_seq": test_seq,
        "test_ids": test_ids,
    }

    print(f"Saving processed data to {config.PROCESSED_DATA_PATH}...")
    np.savez(config.PROCESSED_DATA_PATH, **processed_data)

    return processed_data


def get_dataloaders(batch_size=config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets based on metadata splits.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Get Processed Data (Full Arrays)
    data = process_data(load_cached_data=load_cached_data)

    # 2. Load Metadata for Splitting
    train_meta = pd.read_csv(config.TRAIN_META_PATH)
    val_meta = pd.read_csv(config.VAL_META_PATH)
    test_meta = pd.read_csv(config.TEST_META_PATH)

    # 3. Create ID Lookups
    # The processed data contains all training data (train + val) in 'train_*' arrays.
    # We need to map the IDs in metadata to indices in these arrays.

    # Create a map from ID -> Index for the full training array
    # Assuming unique IDs
    print("Mapping IDs to array indices...")
    train_id_to_idx = {uid: i for i, uid in enumerate(data["train_ids"])}
    test_id_to_idx = {uid: i for i, uid in enumerate(data["test_ids"])}

    # Helper to retrieve indices for a metadata dataframe
    def get_indices(meta_df, id_map, name):
        indices = []
        missing = 0
        for uid in meta_df["id"].values:
            if uid in id_map:
                indices.append(id_map[uid])
            else:
                missing += 1
        if missing > 0:
            print(
                f"Warning: {missing} IDs from {name} metadata not found in processed data."
            )
        return np.array(indices)

    train_indices = get_indices(train_meta, train_id_to_idx, "Train")
    val_indices = get_indices(val_meta, train_id_to_idx, "Validation")
    test_indices = get_indices(test_meta, test_id_to_idx, "Test")

    # 4. Construct Datasets
    print("Constructing Datasets...")

    train_dataset = ManufacturingDataset(
        cont_features=data["train_cont"][train_indices],
        seq_features=data["train_seq"][train_indices],
        targets=data["train_target"][train_indices],
    )

    val_dataset = ManufacturingDataset(
        cont_features=data["train_cont"][val_indices],
        seq_features=data["train_seq"][val_indices],
        targets=data["train_target"][val_indices],
    )

    test_dataset = ManufacturingDataset(
        cont_features=data["test_cont"][test_indices],
        seq_features=data["test_seq"][test_indices],
        targets=None,
    )

    # 5. Create DataLoaders
    print(f"Creating DataLoaders (Batch Size: {batch_size})...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader
