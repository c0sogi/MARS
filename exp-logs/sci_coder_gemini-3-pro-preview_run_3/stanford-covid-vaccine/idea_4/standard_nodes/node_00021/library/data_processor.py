import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Constants and Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Holds pre-loaded tensors in memory for efficiency.
    """

    def __init__(self, features, targets, ids):
        """
        Args:
            features (np.ndarray): Input features of shape (N, Seq_Len, Channels).
            targets (np.ndarray): Target values of shape (N, Seq_Len, 5).
            ids (np.ndarray): Array of sample IDs.
        """
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "inputs": self.features[idx],
            "targets": self.targets[idx],
            "ids": self.ids[idx],
        }


def preprocess_data(parquet_path, is_test=False):
    """
    Loads raw data from Parquet, performs one-hot encoding, and formats targets.

    Args:
        parquet_path (str): Path to the parquet file.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        tuple: (features, targets, ids) as numpy arrays.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH
    n_channels = Config.INPUT_CHANNELS  # 14

    # Initialize arrays
    features = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)

    # Process features
    # Vectorization is tricky with strings, using optimized iteration
    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values
    ids = df["id"].values.astype(str)

    for i in range(n_samples):
        # 1. Sequence (Channels 0-3)
        seq = sequences[i]
        for j, char in enumerate(seq):
            if char in SEQ_MAP:
                features[i, j, SEQ_MAP[char]] = 1.0

        # 2. Structure (Channels 4-6)
        struc = structures[i]
        for j, char in enumerate(struc):
            if char in STRUCT_MAP:
                features[i, j, 4 + STRUCT_MAP[char]] = 1.0

        # 3. Loop Type (Channels 7-13)
        loop = loop_types[i]
        for j, char in enumerate(loop):
            if char in LOOP_MAP:
                features[i, j, 7 + LOOP_MAP[char]] = 1.0

    # Process targets
    # Targets are shape (N, 107, 5).
    # Ground truth is provided for first 68 bases. We pad the rest with 0.
    targets = np.zeros((n_samples, seq_len, Config.OUTPUT_CHANNELS), dtype=np.float32)

    if not is_test:
        for k, col in enumerate(TARGET_COLS):
            # Each row in these columns is a list/array of length 68
            col_data = df[col].values
            for i in range(n_samples):
                val = col_data[i]
                length = len(val)
                targets[i, :length, k] = val

    return features, targets, ids


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    Handles caching of preprocessed data to .npz files.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Helper to handle cache logic
    def load_or_process(cache_path, metadata_path, is_test=False):
        # np.savez appends .npz, so we check for that
        real_cache_path = (
            cache_path if cache_path.endswith(".npz") else cache_path + ".npz"
        )

        if load_cached_data and os.path.exists(real_cache_path):
            try:
                data = np.load(real_cache_path)
                return data["features"], data["targets"], data["ids"]
            except Exception as e:
                print(f"Failed to load cache {real_cache_path}: {e}. Reprocessing.")

        # Process
        print(f"Processing {metadata_path}...")
        features, targets, ids = preprocess_data(metadata_path, is_test=is_test)

        # Save
        print(f"Saving cache to {real_cache_path}...")
        np.savez(real_cache_path, features=features, targets=targets, ids=ids)

        return features, targets, ids

    # 1. Train Data
    train_feats, train_targets, train_ids = load_or_process(
        Config.TRAIN_CACHE, Config.TRAIN_METADATA, is_test=False
    )
    train_dataset = RNADataset(train_feats, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Validation Data
    val_feats, val_targets, val_ids = load_or_process(
        Config.VAL_CACHE, Config.VAL_METADATA, is_test=False
    )
    val_dataset = RNADataset(val_feats, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Test Data
    test_feats, test_targets, test_ids = load_or_process(
        Config.TEST_CACHE, Config.TEST_METADATA, is_test=True
    )
    test_dataset = RNADataset(test_feats, test_targets, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
