import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Constants & Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string and returns an array of indices.
    If index i is paired with j, arr[i] = j.
    If index i is unpaired, arr[i] = -1.
    """
    n = len(structure)
    indices = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i

    return indices


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns shape (Length, Vocab_Size)
    """
    seq_len = len(seq)
    arr = np.zeros((seq_len, vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_dataframe(df, mode="train"):
    """
    Process a dataframe into numpy arrays.
    mode: 'train' (includes targets) or 'test' (no targets)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Channels: 4 (Nucleotides) + 3 (Structure) + 7 (Loop Type) = 14
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: 5 columns
    targets = None
    if mode == "train":
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = []

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Features
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, 4)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, 3)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate features along the channel dimension
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Structure Indices (Adjacency for Graph Module)
        bpp_indices[idx] = get_structure_indices(row["structure"])

        # 3. Targets (if train)
        if mode == "train":
            # Targets are lists of length 68 (Config.PRED_LEN). We pad to 107.
            # The remaining positions are zeros (masked by loss function slicing).
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    targets[idx, :length, t_i] = val_list

        # 4. IDs
        if "id" in row:
            ids.append(row["id"])

    return inputs, bpp_indices, targets, ids


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, bpp_indices, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, 14)
            bpp_indices (np.ndarray): Shape (N, 107)
            targets (np.ndarray, optional): Shape (N, 107, 5)
            ids (list, optional): List of sample IDs
        """
        self.inputs = inputs
        self.bpp_indices = bpp_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "bpp_indices": torch.tensor(self.bpp_indices[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


# =========================================================================
# Data Loading & Caching Logic
# =========================================================================


def get_data(load_cached_data=True, debug=False):
    """
    Loads data from Parquet files, processes features, and caches them as .npz files.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.
        debug (bool): If True, loads a small subset of data for debugging.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filenames (differentiate debug vs full)
    suffix = "_debug" if debug else ""
    train_cache = os.path.join(Config.CACHE_DIR, f"train_data{suffix}.npz")
    val_cache = os.path.join(Config.CACHE_DIR, f"val_data{suffix}.npz")
    test_cache = os.path.join(Config.CACHE_DIR, f"test_data{suffix}.npz")

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading data from cache (debug={debug})...")
        try:
            train_data = np.load(train_cache, allow_pickle=True)
            val_data = np.load(val_cache, allow_pickle=True)
            test_data = np.load(test_cache, allow_pickle=True)

            train_dataset = RNADataset(
                train_data["inputs"], train_data["bpp_indices"], train_data["targets"]
            )
            val_dataset = RNADataset(
                val_data["inputs"], val_data["bpp_indices"], val_data["targets"]
            )
            test_dataset = RNADataset(
                test_data["inputs"], test_data["bpp_indices"], ids=test_data["ids"]
            )

            return train_dataset, val_dataset, test_dataset
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from scratch (debug={debug})...")

    # Load Metadata
    df_train = pd.read_parquet(Config.TRAIN_METADATA)
    df_val = pd.read_parquet(Config.VAL_METADATA)
    df_test = pd.read_parquet(Config.TEST_METADATA)

    # Apply Debug Subset
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        df_train = df_train.iloc[:subset_size].reset_index(drop=True)
        df_val = df_val.iloc[:subset_size].reset_index(drop=True)
        df_test = df_test.iloc[:subset_size].reset_index(drop=True)

    # Process Dataframes
    train_inputs, train_bpp, train_targets, _ = process_dataframe(
        df_train, mode="train"
    )
    val_inputs, val_bpp, val_targets, _ = process_dataframe(df_val, mode="train")
    test_inputs, test_bpp, _, test_ids = process_dataframe(df_test, mode="test")

    # Save to Cache
    np.savez(
        train_cache, inputs=train_inputs, bpp_indices=train_bpp, targets=train_targets
    )
    np.savez(val_cache, inputs=val_inputs, bpp_indices=val_bpp, targets=val_targets)
    np.savez(test_cache, inputs=test_inputs, bpp_indices=test_bpp, ids=test_ids)

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_bpp, train_targets)
    val_dataset = RNADataset(val_inputs, val_bpp, val_targets)
    test_dataset = RNADataset(test_inputs, test_bpp, ids=test_ids)

    return train_dataset, val_dataset, test_dataset


def get_loaders(debug=False):
    """
    Wrapper to get DataLoaders directly.
    """
    train_ds, val_ds, test_ds = get_data(load_cached_data=True, debug=debug)

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
