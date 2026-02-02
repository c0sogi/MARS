import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, inputs, pair_indices, pair_masks, targets, ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.pair_masks = torch.tensor(pair_masks, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_masks": self.pair_masks[idx],
            "targets": self.targets[idx],
            "ids": self.ids[idx],
        }


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate pair indices and masks.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        pair_index (np.ndarray): Array of shape (L,) where value at i is the index of the pair.
                                 Unpaired positions are set to 0 (safe index).
        pair_mask (np.ndarray): Array of shape (L,) where 1 indicates paired, 0 unpaired.
    """
    length = len(structure)
    pairs = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i

    # Create mask: 1 if paired, 0 if unpaired
    pair_mask = (pairs != -1).astype(np.float32)

    # Create index: If unpaired, point to 0 (arbitrary valid index), mask will zero out the effect
    pair_index = pairs.copy()
    pair_index[pairs == -1] = 0

    return pair_index, pair_mask


def one_hot(seq, map_dict, length):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((length, len(map_dict)), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in map_dict:
            arr[i, map_dict[char]] = 1.0
    return arr


def preprocess_data(df, has_targets=True):
    """
    Converts a pandas DataFrame into numpy arrays required for the model.
    """
    # Ensure index is reset to align with array indexing
    df = df.reset_index(drop=True)

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    # Inputs: (N, L, 14) -> 4 (Seq) + 3 (Struct) + 7 (Loop)
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    ids = df["id"].values

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # 1. Feature Encoding
        s_emb = one_hot(row["sequence"], SEQ_MAP, seq_len)
        st_emb = one_hot(row["structure"], STRUCT_MAP, seq_len)
        l_emb = one_hot(row["predicted_loop_type"], LOOP_MAP, seq_len)

        inputs[idx] = np.concatenate([s_emb, st_emb, l_emb], axis=1)

        # 2. Adjacency Map
        p_idx, p_mask = get_structure_adj(row["structure"])
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets
        if has_targets:
            for t_i, col in enumerate(target_cols):
                val = row[col]
                # Targets are lists of length 68 (Config.PRED_LEN)
                # We pad them to 107 with zeros (already initialized to 0)
                if isinstance(val, (list, np.ndarray)):
                    length = min(len(val), seq_len)
                    targets[idx, :length, t_i] = val

    return inputs, pair_indices, pair_masks, targets, ids


def load_or_process(path, cache_path, load_cached_data, debug=False):
    """
    Loads data from cache if available, otherwise processes raw parquet files.
    """
    # Ensure cache path ends with .npz for np.savez_compressed
    if not cache_path.endswith(".npz"):
        real_cache_path = cache_path.replace(".npy", ".npz")
        if real_cache_path == cache_path:
            real_cache_path = cache_path + ".npz"
    else:
        real_cache_path = cache_path

    # Try loading from cache
    if load_cached_data and os.path.exists(real_cache_path):
        print(f"Loading cached data from {real_cache_path}")
        try:
            data = np.load(real_cache_path, allow_pickle=True)
            inputs = data["inputs"]
            pair_indices = data["pair_indices"]
            pair_masks = data["pair_masks"]
            targets = data["targets"]
            ids = data["ids"]

            if debug:
                print("Debug mode: slicing cached data to 100 samples.")
                inputs = inputs[:100]
                pair_indices = pair_indices[:100]
                pair_masks = pair_masks[:100]
                targets = targets[:100]
                ids = ids[:100]

            return inputs, pair_indices, pair_masks, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing data from {path}")
    df = pd.read_parquet(path)

    if debug:
        print("Debug mode: slicing raw data to 100 samples.")
        df = df.head(100)

    # Check if targets exist (Test set won't have them)
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    has_targets = all(col in df.columns for col in target_cols)

    inputs, pair_indices, pair_masks, targets, ids = preprocess_data(
        df, has_targets=has_targets
    )

    # Save cache (Skip if debugging to avoid overwriting full cache with partial data)
    if not debug:
        print(f"Saving cache to {real_cache_path}")
        os.makedirs(os.path.dirname(real_cache_path), exist_ok=True)
        np.savez_compressed(
            real_cache_path,
            inputs=inputs,
            pair_indices=pair_indices,
            pair_masks=pair_masks,
            targets=targets,
            ids=ids,
        )

    return inputs, pair_indices, pair_masks, targets, ids


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main entry point to get PyTorch DataLoaders for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, uses a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Train
    train_inputs, train_pidx, train_pmask, train_targets, train_ids = load_or_process(
        Config.TRAIN_DATA_PATH, Config.TRAIN_CACHE, load_cached_data, debug
    )
    train_dataset = RNADataset(
        train_inputs, train_pidx, train_pmask, train_targets, train_ids
    )

    # Val
    val_inputs, val_pidx, val_pmask, val_targets, val_ids = load_or_process(
        Config.VAL_DATA_PATH, Config.VAL_CACHE, load_cached_data, debug
    )
    val_dataset = RNADataset(val_inputs, val_pidx, val_pmask, val_targets, val_ids)

    # Test
    test_inputs, test_pidx, test_pmask, test_targets, test_ids = load_or_process(
        Config.TEST_DATA_PATH, Config.TEST_CACHE, load_cached_data, debug
    )
    test_dataset = RNADataset(
        test_inputs, test_pidx, test_pmask, test_targets, test_ids
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
