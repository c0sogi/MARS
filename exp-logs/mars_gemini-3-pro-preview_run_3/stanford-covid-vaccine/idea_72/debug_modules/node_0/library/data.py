import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# ==========================================
# Token Dictionaries
# ==========================================
TOKEN_DICT_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_DICT_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN_DICT_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate adjacency indices and a mask.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        adj (np.ndarray): Array of length L. adj[i] = j if i is paired with j.
                          If unpaired, adj[i] = i (self-loop) to ensure valid indexing.
        mask (np.ndarray): Array of length L. 1.0 if paired, 0.0 if unpaired.
    """
    length = len(structure)
    adj = np.arange(length)  # Default to self-loop for safe indexing
    mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return adj, mask


def preprocess_inputs(df):
    """
    Converts dataframe columns into tensor-ready numpy arrays.

    Args:
        df (pd.DataFrame): Input dataframe containing sequence, structure, etc.

    Returns:
        inputs (np.ndarray): Shape (N, 107, 14)
        bpp_indices (np.ndarray): Shape (N, 107)
        bpp_masks (np.ndarray): Shape (N, 107)
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN
    n_channels = Config.INPUT_CHANNELS  # 14

    # Initialize arrays
    inputs = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    bpp_indices = np.zeros((n_samples, seq_len), dtype=np.int64)
    bpp_masks = np.zeros((n_samples, seq_len), dtype=np.float32)

    # Iterate and fill
    for idx, row in df.iterrows():
        # 1. Sequence (Channels 0-3)
        seq = row["sequence"]
        for i, char in enumerate(seq):
            if char in TOKEN_DICT_SEQ:
                inputs[idx, i, TOKEN_DICT_SEQ[char]] = 1.0

        # 2. Structure (Channels 4-6)
        struct = row["structure"]
        for i, char in enumerate(struct):
            if char in TOKEN_DICT_STRUCT:
                inputs[idx, i, 4 + TOKEN_DICT_STRUCT[char]] = 1.0

        # 3. Loop Type (Channels 7-13)
        loop = row["predicted_loop_type"]
        for i, char in enumerate(loop):
            if char in TOKEN_DICT_LOOP:
                inputs[idx, i, 7 + TOKEN_DICT_LOOP[char]] = 1.0

        # 4. Adjacency Map & Mask
        adj, mask = get_structure_adj(struct)
        bpp_indices[idx] = adj
        bpp_masks[idx] = mask

    return inputs, bpp_indices, bpp_masks


def preprocess_targets(df):
    """
    Extracts targets from dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        targets (np.ndarray): Shape (N, 68, 5)
    """
    n_samples = len(df)
    pred_len = Config.PRED_LEN  # 68
    num_targets = Config.NUM_TARGETS  # 5

    targets = np.zeros((n_samples, pred_len, num_targets), dtype=np.float32)
    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        for t_i, col in enumerate(target_cols):
            # The parquet file stores these as lists/arrays
            val_list = row[col]
            # Ensure we only take the first 68 positions
            length = min(len(val_list), pred_len)
            targets[idx, :length, t_i] = val_list[:length]

    return targets


class RNADataset(Dataset):
    def __init__(self, inputs, bpp_indices, bpp_masks, targets=None, ids=None):
        self.inputs = inputs
        self.bpp_indices = bpp_indices
        self.bpp_masks = bpp_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # BPP Indices: (107,)
        bpp_idx = torch.tensor(self.bpp_indices[idx], dtype=torch.long)

        # BPP Mask: (107,) - Used to zero out unpaired interactions
        bpp_mask = torch.tensor(self.bpp_masks[idx], dtype=torch.float32)

        item = {"inputs": x, "bpp_indices": bpp_idx, "bpp_masks": bpp_mask}

        if self.targets is not None:
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = y

        if self.ids is not None:
            item["ids"] = self.ids[idx]

        return item


def get_data(mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes it into tensors, and manages caching.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (inputs, bpp_indices, bpp_masks, targets, ids)
    """
    # Determine source file
    if mode == "train":
        source_path = Config.TRAIN_PARQUET
    elif mode == "val":
        source_path = Config.VAL_PARQUET
    elif mode == "test":
        source_path = Config.TEST_PARQUET
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file path
    cache_path = os.path.join(cache_dir, f"{mode}_data_cache.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Verify integrity of keys
            if "inputs" in data and "bpp_indices" in data and "bpp_masks" in data:
                inputs = data["inputs"]
                bpp_indices = data["bpp_indices"]
                bpp_masks = data["bpp_masks"]
                ids = data["ids"]
                targets = data["targets"] if "targets" in data else None
                # print(f"Loaded {mode} data from cache: {cache_path}")
                return inputs, bpp_indices, bpp_masks, targets, ids
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

    # 2. Compute from scratch
    # print(f"Processing {mode} data from {source_path}...")
    df = pd.read_parquet(source_path)

    # Reset index to ensure alignment
    df = df.reset_index(drop=True)

    inputs, bpp_indices, bpp_masks = preprocess_inputs(df)
    ids = df["id"].values

    targets = None
    if mode in ["train", "val"]:
        targets = preprocess_targets(df)

    # 3. Save to cache
    save_dict = {
        "inputs": inputs,
        "bpp_indices": bpp_indices,
        "bpp_masks": bpp_masks,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)
    # print(f"Saved processed data to {cache_path}")

    return inputs, bpp_indices, bpp_masks, targets, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        load_cached_data (bool): Whether to use cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Train Loader
    train_inputs, train_bpp, train_mask, train_targets, train_ids = get_data(
        "train", load_cached_data
    )
    train_dataset = RNADataset(
        train_inputs, train_bpp, train_mask, train_targets, train_ids
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    # Val Loader
    val_inputs, val_bpp, val_mask, val_targets, val_ids = get_data(
        "val", load_cached_data
    )
    val_dataset = RNADataset(val_inputs, val_bpp, val_mask, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Test Loader
    test_inputs, test_bpp, test_mask, test_targets, test_ids = get_data(
        "test", load_cached_data
    )
    test_dataset = RNADataset(
        test_inputs, test_bpp, test_mask, targets=None, ids=test_ids
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
