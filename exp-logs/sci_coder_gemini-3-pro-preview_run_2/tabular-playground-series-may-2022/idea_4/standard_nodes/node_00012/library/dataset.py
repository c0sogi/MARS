import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, continuous, tokens, targets=None):
        """
        PyTorch Dataset for the Manufacturing task.

        Args:
            continuous (np.ndarray): Normalized continuous features (N, 30).
            tokens (np.ndarray): Tokenized categorical features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.continuous = torch.FloatTensor(continuous)
        self.tokens = torch.LongTensor(tokens)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        item = {"continuous": self.continuous[idx], "tokens": self.tokens[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def process_data(load_cached_data=True):
    """
    Reads raw data, performs feature engineering (tokenization, normalization),
    and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Dictionary containing processed numpy arrays and vocab size.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading cached processed data from {Config.PROCESSED_DATA_PATH}...")
        try:
            with np.load(Config.PROCESSED_DATA_PATH) as data:
                return {key: data[key] for key in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Processing data from scratch...")

    # 2. Load Raw Data
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # 3. Process Categorical Feature (f_27)
    # Combine to build full vocabulary
    all_cat = pd.concat([train_df[Config.CAT_COL], test_df[Config.CAT_COL]], axis=0)

    # Extract unique characters and sort them for determinism
    unique_chars = sorted(list(set("".join(all_cat.unique()))))

    # Create mapping: Char -> Int (1-based index, 0 reserved for padding)
    char_to_idx = {c: i + 1 for i, c in enumerate(unique_chars)}
    vocab_size = len(unique_chars) + 1

    def tokenize_series(series):
        # Convert each string to a list of integers
        return np.array([[char_to_idx[c] for c in s] for s in series], dtype=np.int32)

    train_tokens = tokenize_series(train_df[Config.CAT_COL])
    test_tokens = tokenize_series(test_df[Config.CAT_COL])

    # 4. Process Continuous Features
    # Fit StandardScaler on TRAIN data only
    train_cont = train_df[Config.NUM_COLS].values.astype(np.float32)
    test_cont = test_df[Config.NUM_COLS].values.astype(np.float32)

    mean = np.mean(train_cont, axis=0)
    std = np.std(train_cont, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0

    train_cont = (train_cont - mean) / std
    test_cont = (test_cont - mean) / std

    # 5. Extract Targets and IDs
    train_targets = train_df[Config.TARGET_COL].values.astype(np.float32)
    train_ids = train_df[Config.ID_COL].values
    test_ids = test_df[Config.ID_COL].values

    # 6. Save to Cache
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
    np.savez(
        Config.PROCESSED_DATA_PATH,
        train_tokens=train_tokens,
        train_continuous=train_cont,
        train_targets=train_targets,
        train_ids=train_ids,
        test_tokens=test_tokens,
        test_continuous=test_cont,
        test_ids=test_ids,
        vocab_size=np.array(vocab_size),  # Save as scalar
    )

    print("Data processing complete and cached.")

    return {
        "train_tokens": train_tokens,
        "train_continuous": train_cont,
        "train_targets": train_targets,
        "train_ids": train_ids,
        "test_tokens": test_tokens,
        "test_continuous": test_cont,
        "test_ids": test_ids,
        "vocab_size": np.array(vocab_size),
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for Train, Validation, and Test sets using metadata splits.

    Args:
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab_size)
    """
    # 1. Get Processed Data
    data = process_data(load_cached_data=load_cached_data)
    vocab_size = int(data["vocab_size"])

    # 2. Load Metadata for Splits
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Map Metadata IDs to Processed Data Indices
    # The processed train arrays contain ALL training data (Train + Val).
    # We need to pick specific rows based on the metadata IDs.

    # Create a hash map for O(1) lookup: ID -> Array Index
    train_id_map = {uid: idx for idx, uid in enumerate(data["train_ids"])}
    test_id_map = {uid: idx for idx, uid in enumerate(data["test_ids"])}

    # Get indices
    train_indices = train_meta["id"].map(train_id_map).values
    val_indices = val_meta["id"].map(train_id_map).values
    test_indices = test_meta["id"].map(test_id_map).values

    # 4. Instantiate Datasets
    train_ds = ManufacturingDataset(
        continuous=data["train_continuous"][train_indices],
        tokens=data["train_tokens"][train_indices],
        targets=data["train_targets"][train_indices],
    )

    val_ds = ManufacturingDataset(
        continuous=data["train_continuous"][val_indices],
        tokens=data["train_tokens"][val_indices],
        targets=data["train_targets"][val_indices],
    )

    test_ds = ManufacturingDataset(
        continuous=data["test_continuous"][test_indices],
        tokens=data["test_tokens"][test_indices],
        targets=None,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab_size
