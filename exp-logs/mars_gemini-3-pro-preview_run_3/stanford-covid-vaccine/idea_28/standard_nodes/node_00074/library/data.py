import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Token Mappings
# =========================================================================
TOKEN_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure_str):
    """
    Parses dot-bracket structure string to find base pairs for the
    Zero-Masked Channel-Gating mechanism.

    Args:
        structure_str (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        pair_index (np.ndarray): Array of shape (L,) where arr[i] is the index of the pair.
                                 If unpaired, defaults to 0 (safe for gather, masked later).
        pair_mask (np.ndarray): Array of shape (L,) where 1.0 means paired, 0.0 unpaired.
    """
    L = len(structure_str)
    pair_index = np.zeros(L, dtype=np.int64)
    pair_mask = np.zeros(L, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Register pair (undirected edge)
                pair_index[i] = j
                pair_index[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0

    return pair_index, pair_mask


def one_hot(seq_indices, vocab_size):
    """
    Performs one-hot encoding for a sequence of indices.

    Args:
        seq_indices (list or np.ndarray): Input indices.
        vocab_size (int): Dimension of the one-hot vector.

    Returns:
        np.ndarray: One-hot encoded array of shape (L, vocab_size).
    """
    L = len(seq_indices)
    res = np.zeros((L, vocab_size), dtype=np.float32)
    res[np.arange(L), seq_indices] = 1.0
    return res


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        PyTorch Dataset for RNA Degradation Prediction.

        Args:
            data_dict (dict): Dictionary containing processed numpy arrays.
            mode (str): 'train', 'val', or 'test'.
        """
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Features: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Structural Interaction Data
        p_idx = torch.tensor(self.pair_indices[idx], dtype=torch.long)
        p_mask = torch.tensor(self.pair_masks[idx], dtype=torch.float32)

        sample = {
            "features": x,
            "pair_indices": p_idx,
            "pair_masks": p_mask,
            "id": self.ids[idx],
        }

        if self.mode != "test":
            # Targets: (107, 5) - Padded with zeros beyond seq_scored
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        return sample


def process_data(parquet_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from Parquet, preprocesses features and structure maps, and caches result.

    Args:
        parquet_path (str): Path to input parquet file.
        cache_path (str): Path to save/load .npy cache.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing test set (no targets).

    Returns:
        dict: Dictionary containing processed arrays.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reprocessing...")

    # 2. Load Raw Data
    df = pd.read_parquet(parquet_path)

    # 3. Initialize Containers
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH  # 107

    # Features: (N, 107, 14) -> 4 Seq + 3 Struct + 7 Loop
    all_inputs = np.zeros((n_samples, seq_len, 14), dtype=np.float32)

    # Structural Interaction Maps
    all_pair_indices = np.zeros((n_samples, seq_len), dtype=np.int64)
    all_pair_masks = np.zeros((n_samples, seq_len), dtype=np.float32)

    # Targets
    if not is_test:
        all_targets = np.zeros((n_samples, seq_len, 5), dtype=np.float32)

    ids = []

    # 4. Iterate and Process
    for idx, row in df.iterrows():
        # --- ID ---
        ids.append(row["id"])

        # --- Features ---
        # Sequence (4 channels)
        seq_ints = [TOKEN_SEQ[c] for c in row["sequence"]]
        seq_oh = one_hot(seq_ints, 4)

        # Structure (3 channels)
        struct_ints = [TOKEN_STRUCT[c] for c in row["structure"]]
        struct_oh = one_hot(struct_ints, 3)

        # Loop Type (7 channels)
        loop_ints = [TOKEN_LOOP[c] for c in row["predicted_loop_type"]]
        loop_oh = one_hot(loop_ints, 7)

        # Concatenate: (107, 14)
        sample_features = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)
        all_inputs[idx] = sample_features

        # --- Structural Interaction ---
        p_idx, p_mask = get_structure_adj(row["structure"])
        all_pair_indices[idx] = p_idx
        all_pair_masks[idx] = p_mask

        # --- Targets ---
        if not is_test:
            # Targets are provided as lists of length 68 (seq_scored)
            # We pad them to 107 with zeros for consistent tensor shapes.
            # The loss function/metric will handle slicing.
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                length = len(val_list)
                all_targets[idx, :length, t_i] = val_list

    # 5. Save Cache
    data_dict = {
        "inputs": all_inputs,
        "pair_indices": all_pair_indices,
        "pair_masks": all_pair_masks,
        "ids": ids,
    }

    if not is_test:
        data_dict["targets"] = all_targets

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data_dict)

    return data_dict


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates data loading and creation of PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Process/Load Data
    train_data = process_data(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_CACHE_PATH,
        load_cached_data,
        is_test=False,
    )
    val_data = process_data(
        Config.VAL_METADATA_PATH, Config.VAL_CACHE_PATH, load_cached_data, is_test=False
    )
    test_data = process_data(
        Config.TEST_METADATA_PATH,
        Config.TEST_CACHE_PATH,
        load_cached_data,
        is_test=True,
    )

    # Debugging Subset Logic
    if Config.DEBUG:
        subset_size = Config.DEBUG_SUBSET_SIZE

        def slice_dict(d, size, is_test_set=False):
            keys_to_slice = ["inputs", "pair_indices", "pair_masks"]
            if not is_test_set:
                keys_to_slice.append("targets")

            for k in keys_to_slice:
                if k in d:
                    d[k] = d[k][:size]

            if "ids" in d:
                d["ids"] = d["ids"][:size]
            return d

        train_data = slice_dict(train_data, subset_size, is_test_set=False)
        val_data = slice_dict(val_data, subset_size, is_test_set=False)
        test_data = slice_dict(test_data, subset_size, is_test_set=True)

    # Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # Create DataLoaders
    # Drop last for training to maintain stable batch statistics
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
