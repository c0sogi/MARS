import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =============================================================================
# Constants & Mappings
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# =============================================================================
# Dataset Class
# =============================================================================
class RNADataset(Dataset):
    def __init__(self, inputs, bpp_indices, bpp_mask, targets=None, ids=None):
        """
        PyTorch Dataset for RNA data.

        Args:
            inputs (np.ndarray): Input features of shape (N, seq_len, 14).
            bpp_indices (np.ndarray): Adjacency indices for base pairs (N, seq_len).
            bpp_mask (np.ndarray): Mask for base pairs (N, seq_len).
            targets (np.ndarray, optional): Target values of shape (N, seq_len, 5).
            ids (list, optional): List of sequence IDs.
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.bpp_indices = torch.tensor(bpp_indices, dtype=torch.long)
        self.bpp_mask = torch.tensor(bpp_mask, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "bpp_indices": self.bpp_indices[idx],
            "bpp_mask": self.bpp_mask[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.ids is not None:
            sample["ids"] = self.ids[idx]

        return sample


# =============================================================================
# Preprocessing Functions
# =============================================================================
def one_hot_encode(seq, mapping, length):
    """One-hot encodes a sequence string based on a mapping dictionary."""
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def get_structure_adj(structure, seq_len):
    """
    Parses dot-bracket structure to generate adjacency indices and mask.

    Returns:
        indices: Array where indices[i] = j if i is paired with j.
                 If unpaired, indices[i] = i (self-loop placeholder).
        mask: Array where mask[i] = 1.0 if paired, 0.0 if unpaired.
    """
    indices = np.arange(seq_len)
    mask = np.zeros(seq_len)

    stack = []
    for i, char in enumerate(structure):
        if i >= seq_len:
            break
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
    return indices, mask


def preprocess_data(df, config):
    """
    Converts DataFrame columns into numpy arrays for the model.
    """
    n_samples = len(df)
    seq_len = config.seq_length

    # Initialize arrays
    # Input Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    inputs = np.zeros((n_samples, seq_len, 14), dtype=np.float32)
    bpp_indices = np.zeros((n_samples, seq_len), dtype=np.int64)
    bpp_mask = np.zeros((n_samples, seq_len), dtype=np.float32)

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    has_targets = all(col in df.columns for col in target_cols)

    targets = None
    if has_targets:
        targets = np.zeros((n_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].tolist()

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Feature Encoding
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Structure Adjacency
        inds, msk = get_structure_adj(row["structure"], seq_len)
        bpp_indices[idx] = inds
        bpp_mask[idx] = msk

        # 3. Targets
        if has_targets:
            for t_i, col in enumerate(target_cols):
                val = row[col]
                # Parquet preserves lists/arrays.
                # We pad the provided ground truth (length 68) to seq_len (107) with 0s.
                if isinstance(val, (list, np.ndarray)):
                    length = min(len(val), seq_len)
                    targets[idx, :length, t_i] = val

    return inputs, bpp_indices, bpp_mask, targets, ids


# =============================================================================
# Data Loading & Caching
# =============================================================================
def get_or_create_data(config, mode, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from metadata parquet files.

    Args:
        config: Configuration object.
        mode: 'train', 'val', or 'test'.
        load_cached_data: Boolean to enable/disable cache loading.

    Returns:
        Tuple of (inputs, bpp_indices, bpp_mask, targets, ids)
    """
    # Resolve paths
    if mode == "train":
        parquet_path = config.train_file
        cache_path = config.train_cache
    elif mode == "val":
        parquet_path = config.val_file
        cache_path = config.val_cache
    elif mode == "test":
        parquet_path = config.test_file
        cache_path = config.test_cache
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure cache path ends with .npz for np.savez
    if cache_path.endswith(".npy"):
        cache_file = cache_path[:-4] + ".npz"
    else:
        cache_file = cache_path + ".npz"

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            # Load tensors
            data = np.load(cache_file)
            inputs = data["inputs"]
            bpp_indices = data["bpp_indices"]
            bpp_mask = data["bpp_mask"]

            if "targets" in data:
                targets = data["targets"]
            else:
                targets = None

            # Load IDs from parquet to avoid pickling strings in numpy
            # Reading just the ID column is fast
            df_ids = pd.read_parquet(parquet_path, columns=["id"])
            if config.debug and config.debug_subset_size:
                df_ids = df_ids.iloc[: config.debug_subset_size]
            ids = df_ids["id"].tolist()

            return inputs, bpp_indices, bpp_mask, targets, ids

        except Exception as e:
            print(f"Cache load failed for {mode}: {e}. Recomputing...")

    # Process from scratch
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Handle Debug Mode
    if config.debug and config.debug_subset_size:
        df = df.iloc[: config.debug_subset_size].reset_index(drop=True)

    inputs, bpp_indices, bpp_mask, targets, ids = preprocess_data(df, config)

    # Save to cache (excluding IDs to avoid pickle)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    save_dict = {"inputs": inputs, "bpp_indices": bpp_indices, "bpp_mask": bpp_mask}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez(cache_file, **save_dict)

    return inputs, bpp_indices, bpp_mask, targets, ids


def get_dataloaders(config, load_cached_data=True):
    """
    Constructs DataLoaders for train, validation, and test sets.
    """
    # Train Loader
    train_data = get_or_create_data(config, "train", load_cached_data)
    train_dataset = RNADataset(*train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Loader
    val_data = get_or_create_data(config, "val", load_cached_data)
    val_dataset = RNADataset(*val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Test Loader
    test_data = get_or_create_data(config, "test", load_cached_data)
    test_dataset = RNADataset(*test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
