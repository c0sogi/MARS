import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IDEA_DIR,
    SEQ_LEN,
    PRED_LEN,
    BATCH_SIZE,
    DEBUG,
    DEBUG_SUBSET_SIZE,
    SEED,
)

# Dictionaries for One-Hot Encoding
NUC_MAP = {c: i for i, c in enumerate("AGCU")}
STRUCT_MAP = {c: i for i, c in enumerate("().")}
LOOP_MAP = {c: i for i, c in enumerate("SMIBHEX")}


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string to generate pair indices and a mask.

    Args:
        structure (str): Dot-bracket string (e.g., ".(..).").

    Returns:
        indices (np.ndarray): Array of length L. indices[i] = j if paired with j, else i.
        mask (np.ndarray): Array of length L. 1.0 if paired, 0.0 if unpaired.
    """
    length = len(structure)
    indices = np.arange(length)  # Default to self-loop (will be masked)
    mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
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


def one_hot_encode(seq, mapping, num_classes):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_dataframe(df, has_targets=True):
    """
    Processes a pandas DataFrame into numpy arrays for features and targets.
    """
    num_samples = len(df)

    # Initialize arrays
    # Features: (N, 107, 14)
    # 4 (Nuc) + 3 (Struct) + 7 (Loop) = 14
    features = np.zeros((num_samples, SEQ_LEN, 14), dtype=np.float32)

    # Adjacency: (N, 107)
    pair_indices = np.zeros((num_samples, SEQ_LEN), dtype=np.int64)
    pair_masks = np.zeros((num_samples, SEQ_LEN, 1), dtype=np.float32)

    # Targets: (N, 107, 5)
    # We pad the 68 targets to 107 with zeros.
    targets = np.zeros((num_samples, SEQ_LEN, 5), dtype=np.float32)

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # 1. Features
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Ensure lengths match SEQ_LEN (107)
        if len(seq) != SEQ_LEN:
            continue

        # One-Hot Encoding
        oh_seq = one_hot_encode(seq, NUC_MAP, 4)
        oh_struct = one_hot_encode(struct, STRUCT_MAP, 3)
        oh_loop = one_hot_encode(loop, LOOP_MAP, 7)

        features[idx] = np.concatenate([oh_seq, oh_struct, oh_loop], axis=1)

        # 2. Structure Adjacency
        p_idx, p_mask = get_structure_indices(struct)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask.reshape(-1, 1)

        # 3. Targets
        if has_targets:
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                # val_list is a list of floats of length PRED_LEN (68)
                # We copy it into the first 68 positions of the target array
                length_to_copy = min(len(val_list), SEQ_LEN)
                targets[idx, :length_to_copy, t_i] = val_list[:length_to_copy]

    return features, pair_indices, pair_masks, targets


def get_cache_path(df_ids, prefix):
    """
    Generates a unique cache filename based on the hash of sample IDs.
    """
    # Create a hash of the IDs to ensure data consistency
    ids_str = "".join(sorted(df_ids.astype(str).tolist()))
    ids_hash = hashlib.md5(ids_str.encode("utf-8")).hexdigest()
    filename = f"{prefix}_data_{ids_hash}.npz"
    return os.path.join(IDEA_DIR, filename)


def load_or_process_data(metadata_path, prefix, load_cached_data=True):
    """
    Loads data from parquet, checks cache, processes if needed, and returns arrays.
    """
    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)

    if DEBUG:
        df = df.head(DEBUG_SUBSET_SIZE)

    # Determine if targets exist (Test set usually doesn't have them)
    has_targets = "reactivity" in df.columns

    # Generate Cache Path
    cache_path = get_cache_path(df["id"], prefix)

    # Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return (
                data["features"],
                data["pair_indices"],
                data["pair_masks"],
                data["targets"],
                df["id"].values,
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process Data
    print(f"Processing data for {prefix}...")
    features, pair_indices, pair_masks, targets = process_dataframe(
        df, has_targets=has_targets
    )

    # Save Cache
    print(f"Saving data to cache {cache_path}...")
    np.savez_compressed(
        cache_path,
        features=features,
        pair_indices=pair_indices,
        pair_masks=pair_masks,
        targets=targets,
    )

    return features, pair_indices, pair_masks, targets, df["id"].values


class RNADataset(Dataset):
    def __init__(self, features, pair_indices, pair_masks, targets, ids):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.pair_masks = torch.tensor(pair_masks, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.pair_indices[idx],
            self.pair_masks[idx],
            self.targets[idx],
        )


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    # Process/Load Data
    train_data = load_or_process_data(TRAIN_METADATA_PATH, "train", load_cached_data)
    val_data = load_or_process_data(VAL_METADATA_PATH, "val", load_cached_data)
    test_data = load_or_process_data(TEST_METADATA_PATH, "test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(*train_data)
    val_dataset = RNADataset(*val_data)
    test_dataset = RNADataset(*test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
