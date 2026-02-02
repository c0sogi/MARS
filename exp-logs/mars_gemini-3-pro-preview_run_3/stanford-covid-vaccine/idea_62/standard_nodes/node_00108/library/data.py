import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mappings for One-Hot Encoding
TOKEN_TO_INDEX_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_TO_INDEX_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN_TO_INDEX_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure_str):
    """
    Parses a dot-bracket structure string to generate pair indices and a mask.

    Args:
        structure_str (str): Dot-bracket notation string (e.g., "..((..))..").

    Returns:
        indices (np.ndarray): Array of shape (L,) where indices[i] is the index of the base
                              paired with i. If unpaired, defaults to 0 (safe index).
        mask (np.ndarray): Array of shape (L,) where mask[i] is 1 if paired, 0 otherwise.
    """
    length = len(structure_str)
    indices = np.zeros(length, dtype=np.int64)  # Default to 0 (safe index for gather)
    mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Register pair
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return indices, mask


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    arr = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_dataframe(df, is_test=False):
    """
    Converts a dataframe into numpy arrays for inputs, masks, and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        dict: Dictionary containing numpy arrays.
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize containers
    # Inputs: (N, L, 14) -> 4 seq + 3 struct + 7 loop
    inputs = np.zeros((n_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)

    # Adjacency info: (N, L)
    bpp_indices = np.zeros((n_samples, seq_len), dtype=np.int64)
    bpp_masks = np.zeros((n_samples, seq_len), dtype=np.float32)

    # Targets: (N, 68, 5)
    # Only initialized if not test
    targets = None
    if not is_test:
        targets = np.zeros(
            (n_samples, Config.SEQ_SCORED, Config.NUM_TARGETS), dtype=np.float32
        )

    ids = []

    for idx, row in df.iterrows():
        # 1. Features
        # Sequence (4)
        seq_ohe = one_hot_encode(row["sequence"], TOKEN_TO_INDEX_SEQ, 4)
        # Structure (3)
        struct_ohe = one_hot_encode(row["structure"], TOKEN_TO_INDEX_STRUCT, 3)
        # Loop Type (7)
        loop_ohe = one_hot_encode(row["predicted_loop_type"], TOKEN_TO_INDEX_LOOP, 7)

        # Concatenate channels
        inputs[idx] = np.concatenate([seq_ohe, struct_ohe, loop_ohe], axis=1)

        # 2. Adjacency
        p_idx, p_mask = get_structure_adj(row["structure"])
        bpp_indices[idx] = p_idx
        bpp_masks[idx] = p_mask

        # 3. Targets (if train/val)
        if not is_test:
            # Stack the 5 target lists
            # Each column in df is a list of floats
            t_matrix = np.stack(
                [row[col] for col in Config.TARGET_COLS], axis=1
            )  # Shape (68, 5)
            targets[idx] = t_matrix

        ids.append(row["id"])

    data_dict = {
        "inputs": inputs,
        "bpp_indices": bpp_indices,
        "bpp_masks": bpp_masks,
        "ids": np.array(ids),
    }

    if not is_test:
        data_dict["targets"] = targets

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.inputs = torch.from_numpy(data_dict["inputs"])
        self.bpp_indices = torch.from_numpy(data_dict["bpp_indices"])
        self.bpp_masks = torch.from_numpy(data_dict["bpp_masks"])
        self.ids = data_dict["ids"]
        self.is_test = is_test

        if not is_test:
            self.targets = torch.from_numpy(data_dict["targets"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (107, 14)
        # BPP Indices: (107,)
        # BPP Masks: (107,)
        sample = {
            "inputs": self.inputs[idx],
            "bpp_indices": self.bpp_indices[idx],
            "bpp_masks": self.bpp_masks[idx],
            "id": self.ids[idx],
        }

        if not self.is_test:
            # Targets: (68, 5)
            sample["targets"] = self.targets[idx]

        return sample


def get_data(load_cached_data=True):
    """
    Main function to load and process data.
    Handles caching mechanism.

    Args:
        load_cached_data (bool): If True, attempts to load from .npz cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_cache.npz")
    val_cache_path = os.path.join(cache_dir, "val_cache.npz")
    test_cache_path = os.path.join(cache_dir, "test_cache.npz")

    # Helper to save/load
    def save_cache(path, data_dict):
        np.savez(path, **data_dict)

    def load_cache(path):
        loaded = np.load(path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}

    # Check if cache exists and is requested
    cache_exists = (
        os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
        and os.path.exists(test_cache_path)
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        try:
            train_data = load_cache(train_cache_path)
            val_data = load_cache(val_cache_path)
            test_data = load_cache(test_cache_path)

            # Reconstruct datasets
            train_dataset = RNADataset(train_data, is_test=False)
            val_dataset = RNADataset(val_data, is_test=False)
            test_dataset = RNADataset(test_data, is_test=True)

            return train_dataset, val_dataset, test_dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print("Processing data from Parquet files...")

    # Load Parquet
    df_train = pd.read_parquet(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_parquet(Config.VAL_METADATA_PATH)
    df_test = pd.read_parquet(Config.TEST_METADATA_PATH)

    # Debugging Subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Using subset of {Config.DEBUG_SUBSET_SIZE}")
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # Preprocess
    train_data = preprocess_dataframe(df_train, is_test=False)
    val_data = preprocess_dataframe(df_val, is_test=False)
    test_data = preprocess_dataframe(df_test, is_test=True)

    # Save Cache
    print("Saving data to cache...")
    save_cache(train_cache_path, train_data)
    save_cache(val_cache_path, val_data)
    save_cache(test_cache_path, test_data)

    # Create Datasets
    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(load_cached_data=True):
    """
    Wrapper to get DataLoaders directly.
    """
    train_ds, val_ds, test_ds = get_data(load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
