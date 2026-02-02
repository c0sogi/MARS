import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =============================================================================
# Helper Functions for Data Processing
# =============================================================================


def get_structure_indices(structure_str):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns:
        pair_index: numpy array of shape (L,), where pair_index[i] = j if i and j are paired.
                    If i is unpaired, pair_index[i] = -1.
    """
    L = len(structure_str)
    pair_index = np.full(L, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_index[i] = j
                pair_index[j] = i

    return pair_index


def encode_sequence_features(df):
    """
    Encodes sequence, structure, and loop type into a one-hot tensor.
    Shape: (N, L, 14)
    Channels:
      0-3: A, G, C, U
      4-6: ., (, )
      7-13: S, M, I, B, H, E, X
    """
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Maps
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Initialize output array
    # 4 + 3 + 7 = 14 channels
    features = np.zeros((n_samples, seq_len, 14), dtype=np.float32)

    # Initialize pair indices and masks
    pair_indices = np.zeros((n_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((n_samples, seq_len), dtype=np.float32)

    for i in range(n_samples):
        seq = sequences[i]
        struct = structures[i]
        loop = loops[i]

        # Parse structure for pairs
        p_idx = get_structure_indices(struct)
        pair_indices[i] = p_idx
        # Mask: 1 if paired (idx != -1), 0 otherwise
        pair_masks[i] = (p_idx != -1).astype(np.float32)
        # Fix -1 indices to 0 for gather safety (masked out anyway)
        pair_indices[i][p_idx == -1] = 0

        for j in range(seq_len):
            # Sequence one-hot
            if j < len(seq) and seq[j] in seq_map:
                features[i, j, seq_map[seq[j]]] = 1.0

            # Structure one-hot
            if j < len(struct) and struct[j] in struct_map:
                features[i, j, 4 + struct_map[struct[j]]] = 1.0

            # Loop one-hot
            if j < len(loop) and loop[j] in loop_map:
                features[i, j, 7 + loop_map[loop[j]]] = 1.0

    return features, pair_indices, pair_masks


def encode_targets(df):
    """
    Extracts and pads targets.
    Returns: (N, 107, 5)
    """
    target_cols = Config.TARGET_COLS
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    targets = np.zeros((n_samples, seq_len, 5), dtype=np.float32)

    # If targets don't exist (test set), return zeros
    if target_cols[0] not in df.columns:
        return targets

    for i, col in enumerate(target_cols):
        # Each row in the dataframe column is a list/array of floats
        # We stack them.
        # Note: The Parquet load preserves lists.
        values = df[col].values

        for idx, val_list in enumerate(values):
            # val_list is length 68
            length = len(val_list)
            targets[idx, :length, i] = val_list

    return targets


def process_dataframe(df, is_test=False):
    """
    Process a dataframe into numpy arrays.
    """
    features, pair_indices, pair_masks = encode_sequence_features(df)

    if not is_test:
        targets = encode_targets(df)
    else:
        # Dummy targets for test
        targets = np.zeros((len(df), Config.SEQ_LENGTH, 5), dtype=np.float32)

    # Also extract IDs for submission
    ids = df["id"].values

    return {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
            "pair_masks": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }


# =============================================================================
# Data Loading and Caching
# =============================================================================


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Implements caching logic using .npz files.
    """
    set_seed()

    # Define cache paths
    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # Helper to load or process
    def get_data(cache_path, metadata_path, is_test=False):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading cached data from {cache_path}...")
                loaded = np.load(cache_path, allow_pickle=True)
                return {
                    "features": loaded["features"],
                    "pair_indices": loaded["pair_indices"],
                    "pair_masks": loaded["pair_masks"],
                    "targets": loaded["targets"],
                    "ids": loaded["ids"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Processing data from {metadata_path}...")
        df = pd.read_parquet(metadata_path)
        data_dict = process_dataframe(df, is_test=is_test)

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=data_dict["features"],
            pair_indices=data_dict["pair_indices"],
            pair_masks=data_dict["pair_masks"],
            targets=data_dict["targets"],
            ids=data_dict["ids"],
        )
        print(f"Saved processed data to {cache_path}")

        return data_dict

    # Load Datasets
    train_data = get_data(train_cache, Config.TRAIN_METADATA_PATH, is_test=False)
    val_data = get_data(val_cache, Config.VAL_METADATA_PATH, is_test=False)
    test_data = get_data(test_cache, Config.TEST_METADATA_PATH, is_test=True)

    # Create Dataset Objects
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
