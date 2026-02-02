import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Feature Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_indices(structure_str):
    """
    Parses a dot-bracket structure string to identify base pairs.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (seq_len,) where arr[i] = j if base i is paired with base j.
                    If base i is unpaired, arr[i] = -1.
    """
    n = len(structure_str)
    pair_indices = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i

    return pair_indices


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a string sequence based on a character mapping.

    Args:
        seq (str): Input string.
        mapping (dict): Dictionary mapping characters to integers.
        vocab_size (int): Total size of the vocabulary (channel count).

    Returns:
        np.ndarray: One-hot encoded array of shape (Length, Vocab_Size).
    """
    # Map characters to indices, defaulting to 0 if unknown (though data is clean)
    indices = [mapping.get(c, 0) for c in seq]

    # Create one-hot array
    one_hot = np.zeros((len(seq), vocab_size), dtype=np.float32)
    one_hot[np.arange(len(seq)), indices] = 1.0
    return one_hot


def preprocess_data(df, is_test=False):
    """
    Converts DataFrame columns into numpy arrays for inputs, pair indices, and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): Whether this is test data (no targets).

    Returns:
        tuple: (inputs, pair_indices, targets)
               inputs: (N, 107, 14)
               pair_indices: (N, 107)
               targets: (N, 68, 5) or None if is_test
    """
    sequences = df["sequence"].tolist()
    structures = df["structure"].tolist()
    loops = df["predicted_loop_type"].tolist()

    input_list = []
    pair_idx_list = []

    # Process features
    for s, st, l in zip(sequences, structures, loops):
        # 1. One-Hot Encoding
        s_oh = one_hot_encode(s, SEQ_MAP, 4)
        st_oh = one_hot_encode(st, STRUCT_MAP, 3)
        l_oh = one_hot_encode(l, LOOP_MAP, 7)

        # Concatenate channels: (L, 4+3+7) = (L, 14)
        combined = np.concatenate([s_oh, st_oh, l_oh], axis=1)
        input_list.append(combined)

        # 2. Structure Indices
        p_idx = get_structure_indices(st)
        pair_idx_list.append(p_idx)

    inputs = np.array(input_list, dtype=np.float32)
    pair_indices = np.array(pair_idx_list, dtype=np.int32)

    targets = None
    if not is_test:
        # Process targets
        # Targets in parquet are stored as lists/arrays in columns.
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        t_list = []

        for _, row in df.iterrows():
            row_targets = []
            for col in target_cols:
                val = row[col]
                # Ensure it's a list or array
                if isinstance(val, (list, np.ndarray)):
                    row_targets.append(val)
                else:
                    # Fallback (should not happen with valid parquet)
                    row_targets.append(np.zeros(Config.SEQ_SCORED))

            # Stack channels: (5, 68) -> Transpose to (68, 5)
            t_stack = np.stack(row_targets, axis=1)
            t_list.append(t_stack)

        targets = np.array(t_list, dtype=np.float32)

    return inputs, pair_indices, targets


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Serves (Input, Pair_Index, Target) tuples.
    """

    def __init__(self, inputs, pair_indices, targets=None, ids=None):
        self.inputs = inputs
        self.pair_indices = pair_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        # Pair Indices: (107,) - Used for Structural Attention
        p = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        if self.targets is not None:
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p, y
        else:
            # Return ID for submission generation
            id_val = self.ids[idx] if self.ids is not None else str(idx)
            return x, p, id_val


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Loads data, preprocesses (with caching), and returns PyTorch DataLoaders.

    Args:
        debug (bool): If True, loads only a small subset of data.
        load_cached_data (bool): If True, attempts to load preprocessed .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define Cache Paths
    train_cache_path = Config.TRAIN_CACHE
    val_cache_path = Config.VAL_CACHE
    test_cache_path = Config.TEST_CACHE

    def load_or_process(parquet_path, cache_path, is_test=False):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path) and not debug:
            try:
                data_dict = np.load(cache_path, allow_pickle=True).item()
                # Simple validation check
                if "inputs" in data_dict and "pair_indices" in data_dict:
                    return data_dict
            except Exception as e:
                print(f"Cache load failed for {cache_path}: {e}")

        # 2. Process from scratch
        print(f"Processing data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        if debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        inputs, pair_indices, targets = preprocess_data(df, is_test=is_test)
        ids = df["id"].tolist()

        data_dict = {"inputs": inputs, "pair_indices": pair_indices, "ids": ids}
        if targets is not None:
            data_dict["targets"] = targets

        # Save to cache (only if not debugging)
        if not debug:
            print(f"Saving cache to {cache_path}...")
            np.save(cache_path, data_dict)

        return data_dict

    # Load Datasets
    train_data = load_or_process(Config.TRAIN_PARQUET, train_cache_path, is_test=False)
    val_data = load_or_process(Config.VAL_PARQUET, val_cache_path, is_test=False)
    test_data = load_or_process(Config.TEST_PARQUET, test_cache_path, is_test=True)

    # Create Dataset Objects
    train_dataset = RNADataset(
        train_data["inputs"], train_data["pair_indices"], train_data["targets"]
    )
    val_dataset = RNADataset(
        val_data["inputs"], val_data["pair_indices"], val_data["targets"]
    )
    test_dataset = RNADataset(
        test_data["inputs"], test_data["pair_indices"], ids=test_data["ids"]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
