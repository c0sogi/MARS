import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    def __init__(self, sequences, loop_types, distances, targets=None, masks=None):
        """
        PyTorch Dataset for RNA data.

        Args:
            sequences (np.ndarray): (N, seq_len) int array of nucleotide indices.
            loop_types (np.ndarray): (N, seq_len) int array of loop type indices.
            distances (np.ndarray): (N, seq_len) float array of signed pairing distances.
            targets (np.ndarray, optional): (N, seq_len, n_targets) float array of ground truth.
            masks (np.ndarray, optional): (N, seq_len) bool array indicating valid target positions.
        """
        self.sequences = sequences
        self.loop_types = loop_types
        self.distances = distances
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert numpy rows to tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_types[idx], dtype=torch.long)
        dist = torch.tensor(self.distances[idx], dtype=torch.float)

        item = {"sequence": seq, "loop_type": loop, "distance": dist}

        if self.targets is not None:
            # Targets are float32
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            # Mask indicates which positions have valid ground truth (first 68)
            mask = (
                torch.tensor(self.masks[idx], dtype=torch.bool)
                if self.masks is not None
                else torch.ones_like(target[:, 0], dtype=torch.bool)
            )
            item["target"] = target
            item["mask"] = mask

        return item


def parse_structure(structure_str, seq_len):
    """
    Parses dot-bracket structure string into a signed distance array.

    Logic:
    - If base i and j are paired (i < j):
      - position i gets value (j - i)  [Positive]
      - position j gets value (i - j)  [Negative]
    - Unpaired bases get 0.
    """
    dists = np.zeros(seq_len, dtype=np.float32)
    stack = []
    for i, char in enumerate(structure_str):
        if i >= seq_len:
            break

        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i is current (closing), j is previous (opening)
                # distance for opening (j) -> points downstream -> positive
                dists[j] = float(i - j)
                # distance for closing (i) -> points upstream -> negative
                dists[i] = float(j - i)
    return dists


def tokenize_sequence(seq_str, vocab_map, seq_len):
    """
    Maps characters to integer IDs based on the provided vocab map.
    """
    token_ids = np.zeros(seq_len, dtype=np.int32)
    for i, char in enumerate(seq_str):
        if i >= seq_len:
            break
        token_ids[i] = vocab_map.get(char, 0)
    return token_ids


def prepare_arrays(df, mode="train"):
    """
    Converts dataframe columns to numpy arrays suitable for the dataset.
    Handles tokenization, structure parsing, and target padding.
    """
    n_samples = len(df)
    seq_len = Config.seq_len

    # Pre-allocate input arrays
    sequences = np.zeros((n_samples, seq_len), dtype=np.int32)
    loop_types = np.zeros((n_samples, seq_len), dtype=np.int32)
    distances = np.zeros((n_samples, seq_len), dtype=np.float32)

    # Pre-allocate target arrays (only for train/val)
    targets = None
    masks = None

    has_targets = mode in ["train", "val"]

    if has_targets:
        # 3 scored targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Shape: (N, 107, 3) - we pad the unscored positions
        targets = np.zeros((n_samples, seq_len, Config.n_targets), dtype=np.float32)
        masks = np.zeros((n_samples, seq_len), dtype=np.bool_)

    # Extract raw values from DataFrame
    seq_list = df["sequence"].values
    struct_list = df["structure"].values
    loop_list = df["predicted_loop_type"].values

    if has_targets:
        # In parquet, these are stored as arrays/lists
        reactivity = df["reactivity"].values
        deg_Mg_pH10 = df["deg_Mg_pH10"].values
        deg_Mg_50C = df["deg_Mg_50C"].values

    for i in range(n_samples):
        # 1. Sequence Tokenization
        sequences[i] = tokenize_sequence(seq_list[i], Config.vocab_map, seq_len)

        # 2. Loop Type Tokenization
        loop_types[i] = tokenize_sequence(loop_list[i], Config.loop_type_map, seq_len)

        # 3. Structure Distance Parsing
        distances[i] = parse_structure(struct_list[i], seq_len)

        # 4. Targets Processing
        if has_targets:
            # Get raw lists (expected length 68)
            r = reactivity[i]
            d1 = deg_Mg_pH10[i]
            d2 = deg_Mg_50C[i]

            # Use actual length of the target vector (usually 68)
            current_scored_len = len(r)

            # Fill targets (columns: reactivity, deg_Mg_pH10, deg_Mg_50C)
            targets[i, :current_scored_len, 0] = r
            targets[i, :current_scored_len, 1] = d1
            targets[i, :current_scored_len, 2] = d2

            # Set mask to True for scored positions
            masks[i, :current_scored_len] = True

    return sequences, loop_types, distances, targets, masks


def process_data(mode, load_cached_data=True):
    """
    Orchestrates data loading. Checks cache first, otherwise computes and saves.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        Tuple of numpy arrays.
    """
    cache_file = os.path.join(Config.working_dir, f"{mode}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached {mode} data from {cache_file}...")
            data = np.load(cache_file)
            sequences = data["sequences"]
            loop_types = data["loop_types"]
            distances = data["distances"]

            if mode in ["train", "val"]:
                targets = data["targets"]
                masks = data["masks"]
                return sequences, loop_types, distances, targets, masks
            else:
                return sequences, loop_types, distances, None, None
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {mode} data from source...")
    if mode == "train":
        df = pd.read_parquet(Config.train_file)
    elif mode == "val":
        df = pd.read_parquet(Config.val_file)
    elif mode == "test":
        df = pd.read_parquet(Config.test_file)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Debugging: Use a small subset if configured
    if Config.debug:
        print(
            f"DEBUG MODE: Reducing {mode} dataset to {Config.debug_subset_size} samples."
        )
        df = df.head(Config.debug_subset_size)

    sequences, loop_types, distances, targets, masks = prepare_arrays(df, mode)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if mode in ["train", "val"]:
        np.savez_compressed(
            cache_file,
            sequences=sequences,
            loop_types=loop_types,
            distances=distances,
            targets=targets,
            masks=masks,
        )
    else:
        np.savez_compressed(
            cache_file, sequences=sequences, loop_types=loop_types, distances=distances
        )

    return sequences, loop_types, distances, targets, masks


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Train Loader
    train_seq, train_loop, train_dist, train_tgt, train_mask = process_data(
        "train", load_cached_data
    )
    train_ds = RNADataset(train_seq, train_loop, train_dist, train_tgt, train_mask)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Val Loader
    val_seq, val_loop, val_dist, val_tgt, val_mask = process_data(
        "val", load_cached_data
    )
    val_ds = RNADataset(val_seq, val_loop, val_dist, val_tgt, val_mask)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Test Loader
    test_seq, test_loop, test_dist, _, _ = process_data("test", load_cached_data)
    test_ds = RNADataset(test_seq, test_loop, test_dist)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
