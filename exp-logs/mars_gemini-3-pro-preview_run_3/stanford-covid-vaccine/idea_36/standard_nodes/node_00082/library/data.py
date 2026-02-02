import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Dictionaries for One-Hot Encoding
TOKEN_DICT = {
    "sequence": {"A": 0, "G": 1, "C": 2, "U": 3},
    "structure": {"(": 0, ")": 1, ".": 2},
    "loop_type": {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6},
}


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns:
        pair_index (np.ndarray): Array of shape (L,) where pair_index[i] is the index
                                 of the base paired with i. If unpaired, defaults to 0 (safe index).
        pair_mask (np.ndarray): Array of shape (L,) where 1 indicates paired, 0 unpaired.
    """
    seq_len = len(structure)
    pair_index = np.zeros(seq_len, dtype=np.int64)  # Default to 0 for safety in gather
    pair_mask = np.zeros(seq_len, dtype=np.float32)

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

    return pair_index, pair_mask


def one_hot_encode(seq, token_map, length):
    """
    One-hot encodes a sequence string based on the provided map.
    """
    vocab_size = len(token_map)
    encoding = np.zeros((length, vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in token_map:
            encoding[i, token_map[char]] = 1.0
    return encoding


def process_dataframe(df, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays for inputs, targets, and auxiliary data.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize containers
    # Features: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    X = np.zeros((num_samples, seq_len, Config.NUM_NODE_FEATURES), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: 5 columns
    # We pad targets to seq_len.
    # For train/val, we have data for first 68 positions.
    # For test, we have no targets.
    y = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # Mask for scoring (1 for valid target positions, 0 otherwise)
    target_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # IDs for submission
    ids = []

    for idx, row in df.iterrows():
        # 1. Inputs
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Enforce length check
        current_len = len(seq)
        if current_len != seq_len:
            # Handle potential edge cases if any, though metadata guarantees 107
            pass

        # One-hot encoding
        oh_seq = one_hot_encode(seq, TOKEN_DICT["sequence"], seq_len)
        oh_struct = one_hot_encode(struct, TOKEN_DICT["structure"], seq_len)
        oh_loop = one_hot_encode(loop, TOKEN_DICT["loop_type"], seq_len)

        # Concatenate features
        X[idx] = np.concatenate([oh_seq, oh_struct, oh_loop], axis=1)

        # Structure pairing
        p_idx, p_mask = get_couples(struct)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 2. Targets (if available)
        if mode in ["train", "val"]:
            # Targets are lists of floats
            for t_i, col in enumerate(Config.TARGET_COLS):
                if col in row:
                    val_list = row[col]
                    # The provided lists are of length seq_scored (68)
                    length_scored = len(val_list)
                    y[idx, :length_scored, t_i] = np.array(val_list, dtype=np.float32)

            # Create mask
            # Config.SEQ_SCORED is 68
            target_masks[idx, : Config.SEQ_SCORED] = 1.0

        ids.append(row["id"])

    return {
        "X": X,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "y": y,
        "target_masks": target_masks,
        "ids": np.array(ids),
    }


def get_data(mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches it as .npy files.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    # Determine paths
    if mode == "train":
        input_path = Config.TRAIN_DATA_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif mode == "val":
        input_path = Config.VAL_DATA_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif mode == "test":
        input_path = Config.TEST_DATA_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

    # Process from scratch
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    data = process_dataframe(df, mode=mode)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data)

    return data


class RNADataset(Dataset):
    def __init__(self, data):
        self.X = torch.from_numpy(data["X"]).float()
        self.pair_indices = torch.from_numpy(data["pair_indices"]).long()
        self.pair_masks = torch.from_numpy(data["pair_masks"]).float()
        self.y = torch.from_numpy(data["y"]).float()
        self.target_masks = torch.from_numpy(data["target_masks"]).float()
        self.ids = data["ids"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_masks": self.pair_masks[idx],
            "y": self.y[idx],
            "target_masks": self.target_masks[idx],
            "id": self.ids[idx],
        }


def get_loaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load data
    train_data = get_data("train", load_cached_data)
    val_data = get_data("val", load_cached_data)
    test_data = get_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
