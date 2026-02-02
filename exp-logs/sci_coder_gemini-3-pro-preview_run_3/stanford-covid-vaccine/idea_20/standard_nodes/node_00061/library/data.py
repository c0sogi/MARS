import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    def __init__(
        self, features, adjacency, targets=None, masks=None, ids=None, is_test=False
    ):
        """
        Args:
            features: (N, Seq_Len, Channels)
            adjacency: (N, Seq_Len) - Indices of paired bases, -1 if unpaired
            targets: (N, Seq_Len, Num_Targets) - Padded targets
            masks: (N, Seq_Len) - 1 for scored positions, 0 otherwise
            ids: List of IDs (for test set)
            is_test: Boolean flag
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.adjacency = torch.tensor(adjacency, dtype=torch.long)
        self.is_test = is_test

        if not is_test:
            self.targets = torch.tensor(targets, dtype=torch.float32)
            self.masks = torch.tensor(masks, dtype=torch.float32)
        else:
            self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_test:
            return self.features[idx], self.adjacency[idx], self.ids[idx]
        else:
            return (
                self.features[idx],
                self.adjacency[idx],
                self.targets[idx],
                self.masks[idx],
            )


def parse_structure_pairs(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array of indices where arr[i] = j if i is paired with j, else -1.
    """
    pairs = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def one_hot_encode(seq, token2int, length, num_channels):
    """
    One-hot encodes a sequence string.
    """
    arr = np.zeros((length, num_channels), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in token2int:
            arr[i, token2int[char]] = 1.0
    return arr


def preprocess_data(input_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from parquet, processes features/targets, and handles caching.
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            features = data["features"]
            adjacency = data["adjacency"]

            if is_test:
                ids = data["ids"]
                return RNADataset(features, adjacency, ids=ids, is_test=True)
            else:
                targets = data["targets"]
                masks = data["masks"]
                return RNADataset(
                    features, adjacency, targets=targets, masks=masks, is_test=False
                )
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Debug mode: reduce size
    if Config.debug:
        df = df.head(100)

    num_samples = len(df)
    seq_len = Config.seq_len

    # Initialize arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    features = np.zeros((num_samples, seq_len, Config.input_channels), dtype=np.float32)
    adjacency = np.zeros((num_samples, seq_len), dtype=np.int32)

    if not is_test:
        targets = np.zeros((num_samples, seq_len, Config.num_targets), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    else:
        ids = df["id"].values

    # Process loop
    for idx, row in df.iterrows():
        # --- Features ---
        # 1. Sequence (4 channels)
        seq_oh = one_hot_encode(row["sequence"], Config.token2int_seq, seq_len, 4)

        # 2. Structure (3 channels)
        struct_oh = one_hot_encode(
            row["structure"], Config.token2int_struct, seq_len, 3
        )

        # 3. Loop Type (7 channels)
        loop_oh = one_hot_encode(
            row["predicted_loop_type"], Config.token2int_loop, seq_len, 7
        )

        # Concatenate
        features[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # --- Adjacency ---
        adjacency[idx] = parse_structure_pairs(row["structure"], seq_len)

        # --- Targets (Train/Val only) ---
        if not is_test:
            # Targets are provided as lists of length seq_scored (68)
            # We pad them to seq_len (107) with 0
            # We create a mask of 1s for the first seq_scored positions

            scored_len = row["seq_scored"]

            for t_i, col in enumerate(Config.target_cols):
                val_list = row[col]
                # Ensure val_list is array-like
                if isinstance(val_list, (list, np.ndarray)):
                    # Copy available data
                    length = min(len(val_list), seq_len)
                    targets[idx, :length, t_i] = val_list[:length]

            # Create mask
            masks[idx, :scored_len] = 1.0

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if is_test:
        np.savez(cache_path, features=features, adjacency=adjacency, ids=ids)
        return RNADataset(features, adjacency, ids=ids, is_test=True)
    else:
        np.savez(
            cache_path,
            features=features,
            adjacency=adjacency,
            targets=targets,
            masks=masks,
        )
        return RNADataset(
            features, adjacency, targets=targets, masks=masks, is_test=False
        )


def get_loaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.
    """
    # Train
    train_dataset = preprocess_data(
        Config.train_path,
        Config.train_cache_path,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Validation
    val_dataset = preprocess_data(
        Config.val_path,
        Config.val_cache_path,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Test
    test_dataset = preprocess_data(
        Config.test_path,
        Config.test_cache_path,
        load_cached_data=load_cached_data,
        is_test=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
