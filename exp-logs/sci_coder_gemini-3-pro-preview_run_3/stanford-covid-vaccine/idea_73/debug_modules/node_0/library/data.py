import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Hyperparameters

# --------------------------------------------------------------------------
# Constants & Mappings
# --------------------------------------------------------------------------
TOKEN_TO_INDEX_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_TO_INDEX_STRUCT = {".": 0, "(": 1, ")": 2}
TOKEN_TO_INDEX_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------


def one_hot_encode(sequence, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns shape (length, num_categories).
    """
    num_categories = len(mapping)
    encoding = np.zeros((length, num_categories), dtype=np.float32)

    # Fill encoding
    # If a character is not in mapping (should not happen in clean data), it remains 0
    for i, char in enumerate(sequence):
        if i >= length:
            break
        if char in mapping:
            encoding[i, mapping[char]] = 1.0

    return encoding


def get_adjacency_map(structure, length):
    """
    Parses dot-bracket structure to create an adjacency map.
    Returns shape (length,).
    Values: index of paired base, or -1 if unpaired.
    """
    adj_map = np.full(length, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure):
        if i >= length:
            break

        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                adj_map[start_idx] = i
                adj_map[i] = start_idx

    return adj_map


def process_dataframe(df, is_test=False):
    """
    Converts dataframe columns into numpy arrays for inputs, adjacency, and targets.
    """
    num_samples = len(df)
    seq_len = Hyperparameters.SEQ_LENGTH
    input_dim = Hyperparameters.INPUT_DIM  # 14

    # Pre-allocate arrays
    # Inputs: (N, 107, 14)
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    # Adjacency: (N, 107)
    adjacency = np.zeros((num_samples, seq_len), dtype=np.int64)
    # IDs
    ids = df["id"].values

    # Targets: (N, 68, 5) - only if not test
    targets = None
    if not is_test:
        seq_scored = Hyperparameters.SEQ_SCORED
        num_targets = Hyperparameters.NUM_TARGETS
        targets = np.zeros((num_samples, seq_scored, num_targets), dtype=np.float32)

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Sequence Features (4 channels)
        seq_enc = one_hot_encode(row["sequence"], TOKEN_TO_INDEX_SEQ, seq_len)

        # 2. Structure Features (3 channels)
        struct_enc = one_hot_encode(row["structure"], TOKEN_TO_INDEX_STRUCT, seq_len)

        # 3. Loop Type Features (7 channels)
        loop_enc = one_hot_encode(
            row["predicted_loop_type"], TOKEN_TO_INDEX_LOOP, seq_len
        )

        # Concatenate features
        inputs[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 4. Adjacency Map
        adjacency[idx] = get_adjacency_map(row["structure"], seq_len)

        # 5. Targets (if training/val)
        if not is_test:
            # Extract lists and stack
            # Each target col is a list of floats
            row_targets = []
            for col in TARGET_COLS:
                val = row[col]
                # Ensure it's a list or array
                if not isinstance(val, (list, np.ndarray)):
                    # Fallback for safety, though parquet handles this
                    val = [0.0] * seq_scored
                row_targets.append(val)

            # Stack to (5, 68) then transpose to (68, 5)
            # Note: row_targets is list of length 5, each element length 68
            t_array = np.array(row_targets, dtype=np.float32).T

            # Handle potential length mismatch if parquet load was weird (clip or pad)
            # Though metadata guarantees correctness.
            current_len = t_array.shape[0]
            target_len = Hyperparameters.SEQ_SCORED

            if current_len >= target_len:
                targets[idx] = t_array[:target_len, :]
            else:
                targets[idx, :current_len, :] = t_array

    return inputs, adjacency, targets, ids


def load_cached_data(mode, path, load_cache=True):
    """
    Loads data from parquet, processes it, and caches it to disk.
    mode: 'train', 'val', or 'test'
    """
    cache_file = os.path.join(Hyperparameters.CACHE_DIR, f"{mode}_data.npz")

    # Try loading from cache
    if load_cache and os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            inputs = data["inputs"]
            adjacency = data["adjacency"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None
            return inputs, adjacency, targets, ids
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from {path}...")
    df = pd.read_parquet(path)

    is_test = mode == "test"
    inputs, adjacency, targets, ids = process_dataframe(df, is_test=is_test)

    # Save to cache
    save_dict = {"inputs": inputs, "adjacency": adjacency, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_file, **save_dict)
    print(f"Cached {mode} data to {cache_file}")

    return inputs, adjacency, targets, ids


# --------------------------------------------------------------------------
# Dataset Class
# --------------------------------------------------------------------------


class RNADataset(Dataset):
    def __init__(self, inputs, adjacency, targets=None, ids=None):
        self.inputs = inputs
        self.adjacency = adjacency
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Adjacency: (107,)
        adj = torch.tensor(self.adjacency[idx], dtype=torch.long)

        sample = {"inputs": x, "adjacency": adj, "id": self.ids[idx]}

        if self.targets is not None:
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        return sample


# --------------------------------------------------------------------------
# Data Loaders
# --------------------------------------------------------------------------


def get_dataloaders(load_cached_data_flag=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    # 1. Load Data
    train_inputs, train_adj, train_targets, train_ids = load_cached_data(
        "train", Hyperparameters.TRAIN_DATA_PATH, load_cached_data_flag
    )

    val_inputs, val_adj, val_targets, val_ids = load_cached_data(
        "val", Hyperparameters.VAL_DATA_PATH, load_cached_data_flag
    )

    test_inputs, test_adj, test_targets, test_ids = load_cached_data(
        "test", Hyperparameters.TEST_DATA_PATH, load_cached_data_flag
    )

    # 2. Apply Subset Fraction (for debugging)
    fraction = Hyperparameters.DATA_SUBSET_FRACTION
    if fraction < 1.0:
        print(f"Subsetting data to {fraction*100}%...")

        def subset_arrays(inp, adj, tgt, ids_arr):
            n = int(len(inp) * fraction)
            # Use fixed indices for reproducibility in subsetting
            indices = np.arange(len(inp))
            np.random.shuffle(indices)  # Seed is set in config/utils
            indices = indices[:n]

            s_inp = inp[indices]
            s_adj = adj[indices]
            s_ids = ids_arr[indices]
            s_tgt = tgt[indices] if tgt is not None else None
            return s_inp, s_adj, s_tgt, s_ids

        train_inputs, train_adj, train_targets, train_ids = subset_arrays(
            train_inputs, train_adj, train_targets, train_ids
        )
        val_inputs, val_adj, val_targets, val_ids = subset_arrays(
            val_inputs, val_adj, val_targets, val_ids
        )
        # Usually we don't subset test for submission, but for consistency in debug mode if desired.
        # However, for submission generation, we usually want full test.
        # Assuming debug mode implies checking pipeline integrity, we subset test too.
        test_inputs, test_adj, test_targets, test_ids = subset_arrays(
            test_inputs, test_adj, test_targets, test_ids
        )

    # 3. Create Datasets
    train_dataset = RNADataset(train_inputs, train_adj, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_adj, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_adj, None, test_ids)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Hyperparameters.BATCH_SIZE,
        shuffle=True,
        num_workers=Hyperparameters.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability in training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Hyperparameters.BATCH_SIZE,
        shuffle=False,
        num_workers=Hyperparameters.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Hyperparameters.BATCH_SIZE,
        shuffle=False,
        num_workers=Hyperparameters.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
