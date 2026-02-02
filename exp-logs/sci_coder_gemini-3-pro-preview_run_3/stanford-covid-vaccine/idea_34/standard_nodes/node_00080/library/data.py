import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import config
from library.utils import load_or_process_data


def get_one_hot(seq, token_map):
    """
    Converts a sequence string into a one-hot numpy array.
    Args:
        seq (str): The sequence string (e.g., "AGUC...").
        token_map (dict): Mapping from character to index.
    Returns:
        np.ndarray: One-hot encoded array of shape (L, Vocab).
    """
    vocab_size = len(token_map)
    seq_len = len(seq)
    one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in token_map:
            idx = token_map[char]
            one_hot[i, idx] = 1.0

    return one_hot


def get_structure_indices(structure):
    """
    Parses dot-bracket structure to find pairing indices for structural interaction.

    Args:
        structure (str): Dot-bracket notation string.

    Returns:
        pair_index (np.ndarray): (L,) array where pair_index[i] = j if i is paired with j.
                                 If unpaired, defaults to 0 (to be masked).
        pair_mask (np.ndarray): (L,) array where 1.0 indicates a paired base, 0.0 unpaired.
    """
    L = len(structure)
    pair_index = np.zeros(L, dtype=np.int64)
    pair_mask = np.zeros(L, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_index[i] = j
                pair_index[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0
            else:
                # Handle unbalanced brackets gracefully if present
                pass

    return pair_index, pair_mask


def process_data(split_name):
    """
    Loads raw parquet data and processes it into numpy arrays.
    Designed to be used with the caching utility.

    Args:
        split_name (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    if split_name == "train":
        path = config.TRAIN_PARQUET
    elif split_name == "val":
        path = config.VAL_PARQUET
    elif split_name == "test":
        path = config.TEST_PARQUET
    else:
        raise ValueError(f"Unknown split: {split_name}")

    print(f"Loading {split_name} data from {path}...")
    df = pd.read_parquet(path)

    # Initialize containers
    all_inputs = []
    all_pair_indices = []
    all_pair_masks = []
    all_targets = []
    all_ids = []

    # Pre-fetch maps from config
    map_seq = config.TOKEN_MAP_SEQ
    map_struct = config.TOKEN_MAP_STRUCT
    map_loop = config.TOKEN_MAP_LOOP
    target_cols = config.TARGET_COLS

    for idx, row in df.iterrows():
        # 1. Feature Engineering (One-Hot Encoding)
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Validation check for sequence length
        if len(seq) != config.SEQ_LEN:
            continue

        oh_seq = get_one_hot(seq, map_seq)  # (107, 4)
        oh_struct = get_one_hot(struct, map_struct)  # (107, 3)
        oh_loop = get_one_hot(loop, map_loop)  # (107, 7)

        # Concatenate channels: (107, 14)
        feature_tensor = np.concatenate([oh_seq, oh_struct, oh_loop], axis=1)
        all_inputs.append(feature_tensor)

        # 2. Structural Interaction Indices
        p_idx, p_mask = get_structure_indices(struct)
        all_pair_indices.append(p_idx)
        all_pair_masks.append(p_mask)

        # 3. Target Processing
        if split_name == "test":
            # Test set has no targets, use zeros
            tgt = np.zeros((config.SEQ_LEN, config.NUM_TARGETS), dtype=np.float32)
        else:
            # Train/Val have targets for first 68 positions
            # We extract them and pad to 107
            sample_targets = []
            for col in target_cols:
                val = row[col]
                # Ensure we have a valid list/array
                if hasattr(val, "__len__"):
                    sample_targets.append(np.array(val, dtype=np.float32))
                else:
                    # Fallback for missing data
                    sample_targets.append(np.zeros(config.SEQ_SCORED, dtype=np.float32))

            # Stack to shape (68, 5)
            t_matrix = np.stack(sample_targets, axis=1)

            # Pad to (107, 5)
            padded_targets = np.zeros(
                (config.SEQ_LEN, config.NUM_TARGETS), dtype=np.float32
            )
            padded_targets[: config.SEQ_SCORED, :] = t_matrix
            tgt = padded_targets

        all_targets.append(tgt)
        all_ids.append(row["id"])

    # Convert lists to numpy arrays
    data_dict = {
        "inputs": np.array(all_inputs, dtype=np.float32),  # (N, 107, 14)
        "pair_indices": np.array(all_pair_indices, dtype=np.int64),  # (N, 107)
        "pair_masks": np.array(all_pair_masks, dtype=np.float32),  # (N, 107)
        "targets": np.array(all_targets, dtype=np.float32),  # (N, 107, 5)
        "ids": np.array(all_ids),  # (N,)
    }

    return data_dict


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, data_dict):
        self.inputs = torch.from_numpy(data_dict["inputs"])
        self.pair_indices = torch.from_numpy(data_dict["pair_indices"])
        self.pair_masks = torch.from_numpy(data_dict["pair_masks"])
        self.targets = torch.from_numpy(data_dict["targets"])
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_masks": self.pair_masks[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


def get_dataloaders(debug=False):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    Handles caching via load_or_process_data.

    Args:
        debug (bool): If True, subsets data to 100 samples for quick testing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define cache paths from config
    train_cache = config.TRAIN_CACHE
    val_cache = config.VAL_CACHE
    test_cache = config.TEST_CACHE

    # Load or process data
    # We pass the specific split_name to the process_data function via kwargs
    train_data = load_or_process_data(
        train_cache, process_data, load_cached_data=True, split_name="train"
    )
    val_data = load_or_process_data(
        val_cache, process_data, load_cached_data=True, split_name="val"
    )
    test_data = load_or_process_data(
        test_cache, process_data, load_cached_data=True, split_name="test"
    )

    # Handle Debug Mode
    if debug:
        print("Debug mode active: Subsampling datasets to 100 samples.")
        for d in [train_data, val_data, test_data]:
            for k in ["inputs", "pair_indices", "pair_masks", "targets", "ids"]:
                d[k] = d[k][:100]

    # Initialize Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=True,  # Stabilizes training (Batch Normalization / Size consistency)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
