import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Constants & Mappings
# =============================================================================
TOKEN_DICT = {
    "sequence": {"A": 0, "G": 1, "C": 2, "U": 3},
    "structure": {".": 0, "(": 1, ")": 2},
    "predicted_loop_type": {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6},
}


# =============================================================================
# Dataset Class
# =============================================================================
class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Returns:
        inputs: (Seq_Len, 14) - One-hot encoded features
        adjacency: (Seq_Len, Window_Size) - Indices for structural interaction
        targets/ids: (Pred_Len, 5) or String ID
    """

    def __init__(self, inputs, adjacency, targets=None, ids=None):
        self.inputs = inputs
        self.adjacency = adjacency
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (Seq_Len, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Adjacency: (Seq_Len, Window_Size)
        # We use LongTensor for indices
        adj = torch.tensor(self.adjacency[idx], dtype=torch.long)

        if self.targets is not None:
            # Targets: (Pred_Len, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, adj, y
        else:
            # Inference mode: Return ID to construct submission
            sample_id = self.ids[idx]
            return x, adj, sample_id


# =============================================================================
# Helper Functions
# =============================================================================
def get_pair_map(structure_str):
    """
    Parses a dot-bracket structure string to create a mapping of paired indices.
    Returns an array where arr[i] = j if i is paired with j, else -1.
    """
    pair_map = np.full(len(structure_str), -1, dtype=int)
    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i
    return pair_map


def process_data(df, config, mode="train"):
    """
    Core processing logic:
    1. One-hot encodes sequence, structure, and loop type.
    2. Generates windowed adjacency maps.
    3. Extracts targets (if available).
    """
    # Extract raw data
    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values

    N = len(df)
    L = config.seq_len

    # -------------------------------------------------------------------------
    # 1. Feature Encoding
    # -------------------------------------------------------------------------
    # Sequence (4), Structure (3), Loop (7) -> Total 14 channels
    feat_seq = np.zeros((N, L, 4), dtype=np.float32)
    feat_struc = np.zeros((N, L, 3), dtype=np.float32)
    feat_loop = np.zeros((N, L, 7), dtype=np.float32)

    for i in range(N):
        # Sequence
        seq = sequences[i]
        for j, char in enumerate(seq):
            if j < L:
                feat_seq[i, j, TOKEN_DICT["sequence"].get(char, 0)] = 1.0

        # Structure
        struc = structures[i]
        for j, char in enumerate(struc):
            if j < L:
                feat_struc[i, j, TOKEN_DICT["structure"].get(char, 0)] = 1.0

        # Loop Type
        loop = loop_types[i]
        for j, char in enumerate(loop):
            if j < L:
                feat_loop[i, j, TOKEN_DICT["predicted_loop_type"].get(char, 0)] = 1.0

    # Concatenate all features: (N, L, 14)
    inputs = np.concatenate([feat_seq, feat_struc, feat_loop], axis=2)

    # -------------------------------------------------------------------------
    # 2. Windowed Adjacency Map Generation
    # -------------------------------------------------------------------------
    window_size = config.window_size
    half_win = window_size // 2

    # Initialize with -1 (padding/unpaired)
    adjacency = np.full((N, L, window_size), -1, dtype=np.int32)

    for i in range(N):
        struc = structures[i]
        pm = get_pair_map(struc)

        for pos in range(L):
            # If current position 'pos' is paired with 'pair_idx'
            pair_idx = pm[pos] if pos < len(pm) else -1

            if pair_idx != -1:
                # Gather indices around the partner: [pair_idx-1, pair_idx, pair_idx+1]
                # for window_size=3
                for w_idx, offset in enumerate(range(-half_win, half_win + 1)):
                    neighbor = pair_idx + offset
                    # Boundary check
                    if 0 <= neighbor < L:
                        adjacency[i, pos, w_idx] = neighbor
                    else:
                        adjacency[i, pos, w_idx] = -1
            # If unpaired, it remains -1

    # -------------------------------------------------------------------------
    # 3. Target Extraction
    # -------------------------------------------------------------------------
    targets = None
    if mode in ["train", "val"]:
        T = config.pred_len  # 68
        num_targets = len(config.target_cols)
        targets = np.zeros((N, T, num_targets), dtype=np.float32)

        for t_idx, col in enumerate(config.target_cols):
            # Each row in df[col] is a list/array. Stack them.
            # We assume data integrity from metadata step.
            col_data = np.vstack(df[col].values)
            # Ensure we take exactly the scored length
            targets[:, :, t_idx] = col_data[:, :T]

    # IDs for submission
    ids = df["id"].values if "id" in df.columns else None

    return inputs, adjacency, targets, ids


def get_data_hash(df_len, split, config):
    """
    Generates a unique hash for caching based on data properties and config.
    """
    hasher = hashlib.md5()
    hasher.update(f"{split}_{df_len}".encode("utf-8"))
    hasher.update(f"win{config.window_size}".encode("utf-8"))
    hasher.update(f"seq{config.seq_len}".encode("utf-8"))
    return hasher.hexdigest()


def get_dataset(config, split="train", load_cached_data=True):
    """
    Loads data from Parquet, processes it (or loads from cache), and returns a Dataset.
    """
    # Determine path
    if split == "train":
        path = config.train_metadata_path
    elif split == "val":
        path = config.val_metadata_path
    elif split == "test":
        path = config.test_metadata_path
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load DataFrame
    df = pd.read_parquet(path)

    # Determine Cache Path
    data_hash = get_data_hash(len(df), split, config)
    cache_path = os.path.join(config.cache_dir, f"{split}_data_{data_hash}.npz")

    # Try Loading Cache
    loaded = False
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            inputs = cached["inputs"]
            adjacency = cached["adjacency"]

            if split in ["train", "val"]:
                targets = cached["targets"]
                ids = None
            else:
                targets = None
                ids = cached["ids"]

            print(f"Loaded {split} data from cache: {cache_path}")
            loaded = True
        except Exception as e:
            print(f"Failed to load cache ({e}). Reprocessing.")

    # Process if not loaded
    if not loaded:
        print(f"Processing {split} data from scratch...")
        inputs, adjacency, targets, ids = process_data(df, config, mode=split)

        # Save to cache
        save_dict = {"inputs": inputs, "adjacency": adjacency}
        if targets is not None:
            save_dict["targets"] = targets
        if ids is not None:
            save_dict["ids"] = ids

        np.savez_compressed(cache_path, **save_dict)
        print(f"Saved {split} data to cache: {cache_path}")

    return RNADataset(inputs, adjacency, targets, ids)


# =============================================================================
# Public API
# =============================================================================
def get_dataloaders(config, load_cached_data=True):
    """
    Returns train and validation dataloaders.
    """
    train_ds = get_dataset(config, "train", load_cached_data)
    val_ds = get_dataset(config, "val", load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(config, load_cached_data=True):
    """
    Returns test dataloader.
    """
    test_ds = get_dataset(config, "test", load_cached_data)

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
