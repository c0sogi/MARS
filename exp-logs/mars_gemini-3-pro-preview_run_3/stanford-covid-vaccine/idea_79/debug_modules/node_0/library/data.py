import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Stores features, structural indices, and targets in memory.
    """

    def __init__(self, features, pair_indices, targets=None, ids=None):
        """
        Args:
            features (np.ndarray): Input features of shape (N, Seq_Len, 14).
            pair_indices (np.ndarray): Structural pair indices (N, Seq_Len).
                                       -1 indicates unpaired.
            targets (np.ndarray, optional): Target values (N, Seq_Scored, 5).
            ids (list/np.ndarray, optional): Sample IDs.
        """
        self.features = features
        self.pair_indices = pair_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        pairs = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        item = {"inputs": feat, "pair_indices": pairs}

        if self.ids is not None:
            item["id"] = self.ids[idx]

        if self.targets is not None:
            # Targets are (68, 5)
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = target

        return item


def parse_structure_pairs(structure_str):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] = j if i is paired with j, else -1.
    """
    n = len(structure_str)
    pairs = np.full(n, -1, dtype=np.int32)
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


def process_dataframe(df, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays for the model.

    Args:
        df (pd.DataFrame): Dataframe containing sequence, structure, etc.
        mode (str): 'train' (includes targets) or 'test' (no targets).

    Returns:
        dict: Dictionary containing 'features', 'pair_indices', 'targets', 'ids'.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # 14 channels: 4 seq + 3 struct + 7 loop
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Process sequences
    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values
    ids = df["id"].values

    for i in range(num_samples):
        seq = sequences[i]
        struc = structures[i]
        loop = loop_types[i]

        # 1. One-Hot Encoding
        # Sequence (0-3)
        for j, char in enumerate(seq):
            if char in seq_map:
                features[i, j, seq_map[char]] = 1.0

        # Structure (4-6)
        for j, char in enumerate(struc):
            if char in struct_map:
                features[i, j, 4 + struct_map[char]] = 1.0

        # Loop Type (7-13)
        for j, char in enumerate(loop):
            if char in loop_map:
                features[i, j, 7 + loop_map[char]] = 1.0

        # 2. Pair Indices
        pair_indices[i] = parse_structure_pairs(struc)

    # Process Targets if available
    targets = None
    if mode == "train":
        # Targets are lists in the dataframe columns
        # We need to stack them: (N, 68, 5)
        target_cols = Config.TARGET_COLS
        target_arrays = []

        for col in target_cols:
            # Convert column of lists to numpy array (N, 68)
            # Use np.vstack to stack the lists vertically
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along the last dimension -> (N, 68, 5)
        targets = np.stack(target_arrays, axis=2).astype(np.float32)

    return {
        "features": features,
        "pair_indices": pair_indices,
        "targets": targets,
        "ids": ids,
    }


def load_or_process_data(metadata_path, cache_path, mode="train", load_cache=True):
    """
    Loads data from cache if available, otherwise processes from metadata parquet.
    """
    # 1. Try loading from cache
    if load_cache and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Reconstruct dict
            result = {
                "features": data["features"],
                "pair_indices": data["pair_indices"],
                "ids": data["ids"],
            }
            if "targets" in data and mode == "train":
                result["targets"] = data["targets"]
            elif mode == "train":
                # If we expect targets but cache doesn't have them, force reprocess
                raise ValueError("Cache missing targets for training data")
            else:
                result["targets"] = None

            print(f"Loaded {mode} data from cache: {cache_path}")
            return result
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)
    processed = process_dataframe(df, mode=mode)

    # 3. Save to cache
    save_dict = {
        "features": processed["features"],
        "pair_indices": processed["pair_indices"],
        "ids": processed["ids"],
    }
    if processed["targets"] is not None:
        save_dict["targets"] = processed["targets"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {mode} data to cache: {cache_path}")

    return processed


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): Whether to attempt loading from .npz cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Load Data ---
    train_data = load_or_process_data(
        Config.TRAIN_METADATA,
        Config.TRAIN_CACHE,
        mode="train",
        load_cache=load_cached_data,
    )

    val_data = load_or_process_data(
        Config.VAL_METADATA,
        Config.VAL_CACHE,
        mode="train",  # Val has targets, so treat as train mode
        load_cache=load_cached_data,
    )

    test_data = load_or_process_data(
        Config.TEST_METADATA,
        Config.TEST_CACHE,
        mode="test",
        load_cache=load_cached_data,
    )

    # --- Debugging Subset ---
    if Config.DEBUG_SUBSET_SIZE is not None:
        limit = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG: Slicing datasets to {limit} samples.")

        def slice_dict(d, n):
            d["features"] = d["features"][:n]
            d["pair_indices"] = d["pair_indices"][:n]
            d["ids"] = d["ids"][:n]
            if d["targets"] is not None:
                d["targets"] = d["targets"][:n]
            return d

        train_data = slice_dict(train_data, limit)
        val_data = slice_dict(val_data, limit)
        # Usually we don't slice test for submission, but for full debug pipeline we might
        # Keep test full unless explicitly desired, but here we slice for consistency if debugging
        # test_data = slice_dict(test_data, limit)

    # --- Create Datasets ---
    train_dataset = RNADataset(
        train_data["features"],
        train_data["pair_indices"],
        train_data["targets"],
        train_data["ids"],
    )

    val_dataset = RNADataset(
        val_data["features"],
        val_data["pair_indices"],
        val_data["targets"],
        val_data["ids"],
    )

    test_dataset = RNADataset(
        test_data["features"],
        test_data["pair_indices"],
        targets=None,
        ids=test_data["ids"],
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
