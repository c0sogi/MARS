import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_all


class RNADataset(Dataset):
    def __init__(self, inputs, adjacency, targets, ids):
        """
        Args:
            inputs (np.ndarray): Shape (N, seq_len, 14) - One-hot encoded features.
            adjacency (np.ndarray): Shape (N, seq_len) - Pairing indices.
            targets (np.ndarray): Shape (N, seq_len, 5) - Regression targets.
            ids (list): List of sample IDs.
        """
        self.inputs = inputs
        self.adjacency = adjacency
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs: Float32
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        # Adjacency: Long (Int64)
        adj = torch.tensor(self.adjacency[idx], dtype=torch.long)
        # Targets: Float32
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return x, adj, y


def parse_structure_adjacency(structure_str, seq_len):
    """
    Parses dot-bracket structure string into an adjacency array.

    Args:
        structure_str (str): Dot-bracket string e.g. "((..))"
        seq_len (int): Length of sequence.

    Returns:
        np.ndarray: Array of length seq_len.
                    adj[i] = j if i is paired with j.
                    adj[i] = -1 if i is unpaired.
    """
    adj = np.full(seq_len, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i
    return adj


def one_hot_encode(seq, mapping, num_channels):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    # Map characters to indices, default to 0 if not found (should not happen with clean data)
    indices = np.array([mapping.get(c, 0) for c in seq])
    # Create one-hot matrix
    return np.eye(num_channels)[indices]


def process_data(df):
    """
    Converts a pandas DataFrame into numpy arrays for the dataset.

    Returns:
        inputs: (N, 107, 14)
        adjacency: (N, 107)
        targets: (N, 107, 5)
        ids: List of IDs
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_DIM), dtype=np.float32)
    adjacency = np.zeros((num_samples, seq_len), dtype=np.int64)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    ids = df["id"].tolist()

    # Check if target columns exist (Training/Validation data)
    has_targets = all(col in df.columns for col in Config.TARGET_COLS)

    for idx, row in df.iterrows():
        # 1. Input Features
        # Sequence (4 channels)
        seq_feat = one_hot_encode(row["sequence"], Config.TOKEN2INT, 4)
        # Structure (3 channels)
        struct_feat = one_hot_encode(row["structure"], Config.STRUCT2INT, 3)
        # Loop Type (7 channels)
        loop_feat = one_hot_encode(row["predicted_loop_type"], Config.LOOP2INT, 7)

        # Concatenate features: (107, 4+3+7=14)
        inputs[idx] = np.concatenate([seq_feat, struct_feat, loop_feat], axis=1)

        # 2. Adjacency
        adjacency[idx] = parse_structure_adjacency(row["structure"], seq_len)

        # 3. Targets
        if has_targets:
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Targets are provided for the first SEQ_SCORED positions (68)
                # We pad the rest with 0s (already initialized to 0)
                length = len(val_list)
                targets[idx, :length, t_i] = val_list
        # For test set, targets remain 0

    return inputs, adjacency, targets, ids


def load_or_process_data(parquet_path, cache_path, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from Parquet and caches it.
    Uses .npz format to avoid pickling objects.
    """
    # Adjust cache path to ensure .npz extension
    if not cache_path.endswith(".npz"):
        real_cache_path = cache_path + ".npz"
    else:
        real_cache_path = cache_path

    # Try loading from cache
    if load_cached_data and os.path.exists(real_cache_path):
        try:
            data = np.load(real_cache_path)
            inputs = data["inputs"]
            adjacency = data["adjacency"]
            targets = data["targets"]
            ids = data["ids"].tolist()
            return inputs, adjacency, targets, ids
        except Exception as e:
            print(f"Failed to load cache from {real_cache_path}: {e}. Reprocessing...")

    # Process from scratch
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    inputs, adjacency, targets, ids = process_data(df)

    # Save to cache
    os.makedirs(os.path.dirname(real_cache_path), exist_ok=True)
    np.savez(
        real_cache_path, inputs=inputs, adjacency=adjacency, targets=targets, ids=ids
    )

    return inputs, adjacency, targets, ids


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_all(Config.SEED)

    # --- Train Data ---
    train_inputs, train_adj, train_targets, train_ids = load_or_process_data(
        Config.TRAIN_DATA_PATH, Config.TRAIN_CACHE, load_cached_data
    )

    # --- Validation Data ---
    val_inputs, val_adj, val_targets, val_ids = load_or_process_data(
        Config.VAL_DATA_PATH, Config.VAL_CACHE, load_cached_data
    )

    # --- Test Data ---
    test_inputs, test_adj, test_targets, test_ids = load_or_process_data(
        Config.TEST_DATA_PATH, Config.TEST_CACHE, load_cached_data
    )

    # --- Debug Mode ---
    if debug:
        subset = Config.DEBUG_SUBSET_SIZE
        train_inputs = train_inputs[:subset]
        train_adj = train_adj[:subset]
        train_targets = train_targets[:subset]
        train_ids = train_ids[:subset]

        val_inputs = val_inputs[:subset]
        val_adj = val_adj[:subset]
        val_targets = val_targets[:subset]
        val_ids = val_ids[:subset]

        test_inputs = test_inputs[:subset]
        test_adj = test_adj[:subset]
        test_targets = test_targets[:subset]
        test_ids = test_ids[:subset]

    # --- Create Datasets ---
    train_dataset = RNADataset(train_inputs, train_adj, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_adj, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_adj, test_targets, test_ids)

    # --- Create DataLoaders ---
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
