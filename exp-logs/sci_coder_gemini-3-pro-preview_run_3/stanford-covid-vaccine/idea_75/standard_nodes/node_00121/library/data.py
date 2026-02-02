import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mappings for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, features, pair_indices, targets=None):
        """
        Args:
            features (np.ndarray): Input features of shape (N, 107, 14).
            pair_indices (np.ndarray): Adjacency indices of shape (N, 107).
            targets (np.ndarray, optional): Targets of shape (N, 107, 5).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            # Create dummy targets for test set if not provided
            self.targets = torch.zeros((len(features), 107, 5), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.pair_indices[idx], self.targets[idx]


def get_structure_adj(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to generate a pair index map.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").
        seq_len (int): Length of sequence.

    Returns:
        np.ndarray: Array of shape (seq_len,) where arr[i] is the index of the pair
                    for base i. If unpaired, arr[i] = i (self-loop).
    """
    stack = []
    indices = np.arange(seq_len)  # Default to self-loop for unpaired

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                indices[start] = i
                indices[i] = start

    return indices


def process_data(df, config, is_test=False):
    """
    Process raw pandas DataFrame into numpy arrays for features, adjacency, and targets.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize arrays
    # Features: 4 (seq) + 3 (struct) + 7 (loop) = 14 channels
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    # Target columns in the dataframe
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, row in df.iterrows():
        # 1. Sequence One-Hot (Channels 0-3)
        seq = row["sequence"]
        for j, char in enumerate(seq):
            if char in SEQ_MAP:
                features[i, j, SEQ_MAP[char]] = 1.0

        # 2. Structure One-Hot (Channels 4-6)
        struct = row["structure"]
        for j, char in enumerate(struct):
            if char in STRUCT_MAP:
                features[i, j, 4 + STRUCT_MAP[char]] = 1.0

        # 3. Loop Type One-Hot (Channels 7-13)
        loop = row["predicted_loop_type"]
        for j, char in enumerate(loop):
            if char in LOOP_MAP:
                features[i, j, 7 + LOOP_MAP[char]] = 1.0

        # 4. Adjacency Map
        pair_indices[i] = get_structure_adj(struct, seq_len)

        # 5. Targets (if not test)
        if not is_test:
            # Targets are lists of length `seq_scored` (68).
            # We pad them to `seq_len` (107) with zeros.
            for k, col in enumerate(target_cols):
                val_list = row[col]
                # Check if val_list is a valid list/array
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    targets[i, :length, k] = val_list

    return features, pair_indices, targets


def load_or_process_data(mode, config, load_cached_data=True):
    """
    Loads data from cache or processes it from Parquet files.

    Args:
        mode (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (features, pair_indices, targets)
    """
    # Determine file paths
    if mode == "train":
        parquet_path = config.train_file
        cache_path = config.train_cache_path
    elif mode == "val":
        parquet_path = config.val_file
        cache_path = config.val_cache_path
    elif mode == "test":
        parquet_path = config.test_file
        cache_path = config.test_cache_path
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            return data["features"], data["pair_indices"], data["targets"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    is_test = mode == "test"
    features, pair_indices, targets = process_data(df, config, is_test=is_test)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_path}")
    np.savez(cache_path, features=features, pair_indices=pair_indices, targets=targets)

    return features, pair_indices, targets


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.
        debug (bool): If True, uses a smaller subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    config = Config(debug=debug)

    # Load Data
    train_feats, train_pairs, train_targets = load_or_process_data(
        "train", config, load_cached_data
    )
    val_feats, val_pairs, val_targets = load_or_process_data(
        "val", config, load_cached_data
    )
    test_feats, test_pairs, test_targets = load_or_process_data(
        "test", config, load_cached_data
    )

    # Debug Mode: Slice data
    if debug:
        subset_size = 32
        train_feats = train_feats[:subset_size]
        train_pairs = train_pairs[:subset_size]
        train_targets = train_targets[:subset_size]

        val_feats = val_feats[:subset_size]
        val_pairs = val_pairs[:subset_size]
        val_targets = val_targets[:subset_size]

        test_feats = test_feats[:subset_size]
        test_pairs = test_pairs[:subset_size]
        test_targets = test_targets[:subset_size]

    # Create Datasets
    train_dataset = RNADataset(train_feats, train_pairs, train_targets)
    val_dataset = RNADataset(val_feats, val_pairs, val_targets)
    test_dataset = RNADataset(test_feats, test_pairs, test_targets)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
