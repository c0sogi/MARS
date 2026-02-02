import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# ==========================================
# Helper Functions
# ==========================================


def get_pair_indices(structure):
    """
    Parses a dot-bracket structure string and returns an array of paired indices.
    If a base is unpaired, it maps to itself (self-loop) to ensure valid gathering
    in the interaction module.

    Args:
        structure (str): Dot-bracket string (e.g., ".(..).").

    Returns:
        np.ndarray: Array of shape (len(structure),) where arr[i] is the index of the pair.
    """
    n = len(structure)
    # Default to self-loop (index points to itself)
    pair_indices = np.arange(n, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Set bidirectional link
                pair_indices[i] = j
                pair_indices[j] = i

    return pair_indices


def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping.

    Args:
        seq (str): Input string.
        mapping (dict): Character to index mapping.
        length (int): Expected number of channels.

    Returns:
        np.ndarray: One-hot encoded array of shape (len(seq), length).
    """
    arr = np.zeros((len(seq), length), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_data(df, has_targets=True):
    """
    Preprocesses the dataframe into numpy arrays for inputs, pair indices, and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        has_targets (bool): Whether to extract targets (True for train/val, False for test).

    Returns:
        dict: Dictionary containing 'inputs', 'pair_indices', 'ids', and optionally 'targets'.
    """
    # 1. Prepare Input Features
    sequences = df["sequence"].tolist()
    structures = df["structure"].tolist()
    loop_types = df["predicted_loop_type"].tolist()
    ids = df["id"].values

    inputs_list = []
    pairs_list = []

    for seq, struc, loop in zip(sequences, structures, loop_types):
        # Sequence encoding (4 channels)
        enc_seq = one_hot_encode(seq, SEQ_MAP, 4)
        # Structure encoding (3 channels)
        enc_struc = one_hot_encode(struc, STRUCT_MAP, 3)
        # Loop encoding (7 channels)
        enc_loop = one_hot_encode(loop, LOOP_MAP, 7)

        # Concatenate channels: (L, 14)
        combined = np.concatenate([enc_seq, enc_struc, enc_loop], axis=1)
        inputs_list.append(combined)

        # Generate pair indices for interaction module
        pairs_list.append(get_pair_indices(struc))

    inputs = np.array(inputs_list, dtype=np.float32)  # Shape: (N, 107, 14)
    pair_indices = np.array(pairs_list, dtype=np.int64)  # Shape: (N, 107)

    result = {"inputs": inputs, "pair_indices": pair_indices, "ids": ids}

    # 2. Prepare Targets (if applicable)
    if has_targets:
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        t_list = []

        for col in target_cols:
            # df[col] contains lists of floats. Convert to 2D array (N, 68)
            # We assume all lists in a column have the same length (68)
            col_data = np.array(df[col].tolist(), dtype=np.float32)
            t_list.append(col_data)

        # Stack to create (N, 68, 5)
        # axis=2 puts the 5 targets in the last dimension
        targets = np.stack(t_list, axis=2)
        result["targets"] = targets

    return result


def load_or_process_data(
    split_name, parquet_path, cache_path, load_cached_data=True, debug=False
):
    """
    Loads data from cache or processes from Parquet file.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        parquet_path (str): Path to input parquet file.
        cache_path (str): Path to save/load .npy cache.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes a subset.

    Returns:
        dict: Data dictionary.
    """
    # Modify cache path for debug mode to avoid corrupting full cache
    if debug:
        cache_path = cache_path.replace(".npy", "_debug.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split_name} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    if debug:
        df = df.iloc[:100].reset_index(drop=True)
        print(f"Debug mode: reduced {split_name} size to {len(df)}")

    has_targets = split_name != "test"
    data = preprocess_data(df, has_targets=has_targets)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data)
    print(f"Saved {split_name} data to cache: {cache_path}")

    return data


# ==========================================
# Dataset Class
# ==========================================


class RNADataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Dictionary containing 'inputs', 'pair_indices', 'ids', and optional 'targets'.
        """
        self.inputs = data["inputs"]  # (N, 107, 14)
        self.pair_indices = data["pair_indices"]  # (N, 107)
        self.ids = data["ids"]
        self.targets = data.get("targets")  # (N, 68, 5) or None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        x = torch.from_numpy(self.inputs[idx])
        pairs = torch.from_numpy(self.pair_indices[idx])

        item = {"inputs": x, "pair_indices": pairs, "id": self.ids[idx]}

        if self.targets is not None:
            y = torch.from_numpy(self.targets[idx])
            item["targets"] = y

        return item


# ==========================================
# DataLoader Factory
# ==========================================


def get_dataloaders(
    debug=False, load_cached_data=True, batch_size=None, num_workers=None
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        debug (bool): Enable debug mode.
        load_cached_data (bool): Use cached .npy files.
        batch_size (int, optional): Override config batch size.
        num_workers (int, optional): Override config num workers.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use Config defaults if not provided
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Load Data Dictionaries
    train_data = load_or_process_data(
        "train", Config.TRAIN_PATH, Config.TRAIN_CACHE, load_cached_data, debug
    )
    val_data = load_or_process_data(
        "val", Config.VAL_PATH, Config.VAL_CACHE, load_cached_data, debug
    )
    test_data = load_or_process_data(
        "test", Config.TEST_PATH, Config.TEST_CACHE, load_cached_data, debug
    )

    # Create Dataset Objects
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,  # Important for batch norm / stability
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    return train_loader, val_loader, test_loader
