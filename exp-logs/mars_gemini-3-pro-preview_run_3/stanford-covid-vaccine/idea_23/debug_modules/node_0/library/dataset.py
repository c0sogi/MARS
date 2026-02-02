import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ------------------------------------------------------------------------------
# Mappings
# ------------------------------------------------------------------------------
# Maps characters to integer indices for one-hot encoding
TOKEN_TO_INDEX_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_TO_INDEX_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN_TO_INDEX_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def get_pair_map(structure):
    """
    Parses a dot-bracket structure string to generate a pair index map.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len(structure),) where arr[i] is the index
                    of the base paired with i, or -1 if unpaired.
    """
    length = len(structure)
    pair_map = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i

    return pair_map


def one_hot_encode(indices, num_classes):
    """
    Manual one-hot encoding for numpy arrays.

    Args:
        indices (list): List of integer indices.
        num_classes (int): Number of classes (dimension of one-hot vector).

    Returns:
        np.ndarray: One-hot encoded array of shape (len(indices), num_classes).
    """
    return np.eye(num_classes)[indices]


def process_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into numpy arrays for inputs, pair maps, and targets.

    Args:
        df (pd.DataFrame): The dataframe containing sequence and structure data.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        tuple: (inputs, pair_indices, targets, masks, ids)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Inputs: (N, 107, 14) -> 4 seq + 3 struct + 7 loop
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_DIM), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = []

    for idx, row in df.iterrows():
        # 1. Parse Sequences
        # Nucleotide Sequence (4 channels)
        seq_ints = [TOKEN_TO_INDEX_SEQ.get(c, 0) for c in row["sequence"]]
        inputs[idx, :, 0:4] = one_hot_encode(seq_ints, 4)

        # Structure Sequence (3 channels)
        struct_ints = [TOKEN_TO_INDEX_STRUCT.get(c, 2) for c in row["structure"]]
        inputs[idx, :, 4:7] = one_hot_encode(struct_ints, 3)

        # Loop Type Sequence (7 channels)
        loop_ints = [TOKEN_TO_INDEX_LOOP.get(c, 5) for c in row["predicted_loop_type"]]
        inputs[idx, :, 7:14] = one_hot_encode(loop_ints, 7)

        # 2. Pair Map
        pair_indices[idx, :] = get_pair_map(row["structure"])

        # 3. Targets & Mask
        # seq_scored determines how many positions have ground truth
        scored_len = int(row["seq_scored"])

        if not is_test:
            # Extract targets from lists
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_list = []
            for col in Config.TARGET_COLS:
                val = row[col]
                # Ensure it's a list/array
                if isinstance(val, (list, np.ndarray)):
                    t_list.append(val)
                else:
                    # Fallback for errors or missing data
                    t_list.append(np.zeros(scored_len))

            # Stack targets (5, scored_len) -> Transpose to (scored_len, 5)
            # Note: The raw data lists are length `scored_len` (68)
            sample_targets = np.array(t_list, dtype=np.float32).T

            # Fill the target array up to scored_len
            # We clip scored_len to seq_len just in case, though it should be <= 107
            valid_len = min(scored_len, seq_len)
            targets[idx, :valid_len, :] = sample_targets[:valid_len, :]

            # Set mask (1 for valid positions, 0 for padded/unscored positions)
            masks[idx, :valid_len] = 1.0
        else:
            # For test set, targets remain 0. Mask remains 0 (or can be used to indicate sequence length)
            # We generally don't calculate loss on test set, so mask=0 is fine.
            pass

        ids.append(row["id"])

    return inputs, pair_indices, targets, masks, np.array(ids)


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, targets, masks, ids):
        self.inputs = inputs
        self.pair_indices = pair_indices
        self.targets = targets
        self.masks = masks
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
            "mask": torch.tensor(self.masks[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }


# ------------------------------------------------------------------------------
# Data Loading & Caching
# ------------------------------------------------------------------------------


def load_data(split, load_cached_data=True):
    """
    Loads data for a specific split, utilizing caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (inputs, pair_indices, targets, masks, ids)
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA
        cache_path = Config.CACHE_TRAIN
        is_test = False
    elif split == "val":
        meta_path = Config.VAL_METADATA
        cache_path = Config.CACHE_VAL
        is_test = False
    elif split == "test":
        meta_path = Config.TEST_METADATA
        cache_path = Config.CACHE_TEST
        is_test = True
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} data from cache: {cache_path}")
            # allow_pickle=True is required for string arrays (ids)
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["inputs"],
                data["pair_indices"],
                data["targets"],
                data["masks"],
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from source...")

    # 2. Process from scratch
    print(f"Processing {split} data from metadata: {meta_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)

    # Process dataframe into tensors
    inputs, pair_indices, targets, masks, ids = process_dataframe(df, is_test=is_test)

    # 3. Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        pair_indices=pair_indices,
        targets=targets,
        masks=masks,
        ids=ids,
    )

    return inputs, pair_indices, targets, masks, ids


def get_dataloader(
    split, batch_size=None, shuffle=None, num_workers=None, load_cached_data=True
):
    """
    Factory function to create a DataLoader for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        shuffle (bool, optional): Whether to shuffle. Defaults to True for train.
        num_workers (int, optional): Number of workers. Defaults to Config.NUM_WORKERS.
        load_cached_data (bool, optional): Whether to use cached data. Defaults to True.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    # Set defaults if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if shuffle is None:
        shuffle = split == "train"
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Load data (cached or fresh)
    inputs, pair_indices, targets, masks, ids = load_data(split, load_cached_data)

    # Handle Debug Mode: Slice dataset to a small subset
    if Config.DEBUG:
        debug_size = min(len(inputs), Config.DEBUG_SIZE)
        inputs = inputs[:debug_size]
        pair_indices = pair_indices[:debug_size]
        targets = targets[:debug_size]
        masks = masks[:debug_size]
        ids = ids[:debug_size]
        print(f"DEBUG MODE: Sliced {split} dataset to {len(inputs)} samples.")

    # Instantiate Dataset
    dataset = RNADataset(inputs, pair_indices, targets, masks, ids)

    # Instantiate DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
