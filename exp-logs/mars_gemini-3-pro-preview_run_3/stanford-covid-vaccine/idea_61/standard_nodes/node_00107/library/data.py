import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants and Mappings
# ==========================================
NUC_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# ==========================================
# Helper Functions
# ==========================================


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate pair indices.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (L,) where arr[i] is the index of the base paired with i.
                    Unpaired bases are assigned -1.
    """
    length = len(structure)
    pairs = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i

    return pairs


def one_hot(seq, map_dict, num_classes):
    """
    Converts a sequence string into a one-hot encoded matrix.

    Args:
        seq (str): Input sequence.
        map_dict (dict): Mapping from character to index.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (L, num_classes).
    """
    res = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in map_dict:
            res[i, map_dict[char]] = 1.0
    return res


# ==========================================
# Dataset Class
# ==========================================


class RNADataset(Dataset):
    def __init__(self, inputs, targets=None, pair_indices=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, 14).
            targets (np.ndarray, optional): Shape (N, 107, 5).
            pair_indices (np.ndarray, optional): Shape (N, 107).
            ids (np.ndarray, optional): Array of sample IDs.
        """
        self.inputs = inputs
        self.targets = targets
        self.pair_indices = pair_indices
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 1. Inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        item = {"inputs": x}

        # 2. Structural Interaction Data
        if self.pair_indices is not None:
            raw_pair_idx = self.pair_indices[idx]

            # Mask: 1 if paired, 0 if unpaired
            # This allows the model to zero out h_j for unpaired bases
            pair_mask = (raw_pair_idx != -1).astype(np.float32)

            # Safe Indices: Replace -1 with 0 to prevent gather errors.
            # The mask will ensure the value gathered at index 0 is ignored if the base is actually unpaired.
            safe_pair_idx = raw_pair_idx.copy()
            safe_pair_idx[safe_pair_idx == -1] = 0

            item["pair_index"] = torch.tensor(safe_pair_idx, dtype=torch.long)
            item["pair_mask"] = torch.tensor(pair_mask, dtype=torch.float32)

        # 3. Targets: (107, 5)
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = y

        # 4. Metadata
        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


# ==========================================
# Data Processing & Caching
# ==========================================


def process_data(parquet_path, split_name, load_cached_data=True):
    """
    Loads raw parquet data, processes it into tensors, and handles caching.

    Args:
        parquet_path (str): Path to the parquet file.
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (inputs, targets, pair_indices, ids)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            pair_indices = data["pair_indices"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None
            return inputs, targets, pair_indices, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    n_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM

    # Initialize arrays
    inputs = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((n_samples, seq_len), dtype=np.int32)
    ids = df["id"].values

    # Check for targets
    has_targets = "reactivity" in df.columns
    targets = None
    if has_targets:
        targets = np.zeros((n_samples, seq_len, Config.OUTPUT_DIM), dtype=np.float32)

    # Iterate and process
    for i, row in df.iterrows():
        # A. Features
        # 4 Channels: Sequence
        seq_oh = one_hot(row["sequence"], NUC_MAP, 4)
        # 3 Channels: Structure
        struct_oh = one_hot(row["structure"], STRUCT_MAP, 3)
        # 7 Channels: Loop Type
        loop_oh = one_hot(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate: (107, 14)
        inputs[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # B. Structure Adjacency
        pair_indices[i] = get_structure_adj(row["structure"])

        # C. Targets
        if has_targets:
            # Targets are provided for the first `seq_scored` positions (usually 68).
            # We pad the rest with zeros to match (107, 5).
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val = row[col]
                # val is expected to be a list or array
                if isinstance(val, (list, np.ndarray)):
                    length = len(val)
                    targets[i, :length, t_idx] = val
                else:
                    # Fallback for unexpected format, though parquet should preserve lists
                    pass

    # 3. Save Cache
    save_dict = {"inputs": inputs, "pair_indices": pair_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez(cache_path, **save_dict)
    print(f"Saved {split_name} data to {cache_path}")

    return inputs, targets, pair_indices, ids


# ==========================================
# Main Interface
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Train
    train_inputs, train_targets, train_pairs, train_ids = process_data(
        Config.TRAIN_DATA_PATH, "train", load_cached_data
    )
    train_ds = RNADataset(train_inputs, train_targets, train_pairs, train_ids)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Validation
    val_inputs, val_targets, val_pairs, val_ids = process_data(
        Config.VAL_DATA_PATH, "val", load_cached_data
    )
    val_ds = RNADataset(val_inputs, val_targets, val_pairs, val_ids)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test
    test_inputs, _, test_pairs, test_ids = process_data(
        Config.TEST_DATA_PATH, "test", load_cached_data
    )
    test_ds = RNADataset(test_inputs, None, test_pairs, test_ids)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
