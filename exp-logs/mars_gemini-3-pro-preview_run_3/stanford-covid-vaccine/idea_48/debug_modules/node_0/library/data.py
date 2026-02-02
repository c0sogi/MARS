import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import save_cache, load_cache

# ==========================================
# Mappings
# ==========================================
TOKEN2INT_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN2INT_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN2INT_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_pair_index_and_mask(structure):
    """
    Parses dot-bracket structure string to generate pair indices and masks.
    Returns:
        pair_index: np.array of shape (L,), containing index of paired base.
                    Unpaired bases are set to 0 (safe placeholder).
        pair_mask: np.array of shape (L,), 1.0 if paired, 0.0 if unpaired.
    """
    length = len(structure)
    pair_index = np.zeros(length, dtype=np.int64)  # Default to 0
    pair_mask = np.zeros(length, dtype=np.float32)

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


def preprocess_data(df, is_test=False):
    """
    Converts DataFrame columns into numpy arrays for model input.
    """
    num_samples = len(df)
    seq_len = Config.seq_len
    input_dim = Config.input_dim

    # Initialize arrays
    # Inputs: (N, L, 14)
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    # Pair Index: (N, L)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    # Pair Mask: (N, L)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    # Targets: (N, L, 5)
    targets = np.zeros((num_samples, seq_len, Config.num_classes), dtype=np.float32)
    # IDs
    ids = df["id"].values

    # Target columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # 1. Sequence Encoding (0-3)
        seq = row["sequence"]
        for i, char in enumerate(seq):
            if i < seq_len and char in TOKEN2INT_SEQ:
                inputs[idx, i, TOKEN2INT_SEQ[char]] = 1.0

        # 2. Structure Encoding (4-6)
        struct = row["structure"]
        for i, char in enumerate(struct):
            if i < seq_len and char in TOKEN2INT_STRUCT:
                inputs[idx, i, 4 + TOKEN2INT_STRUCT[char]] = 1.0

        # 3. Loop Type Encoding (7-13)
        loop = row["predicted_loop_type"]
        for i, char in enumerate(loop):
            if i < seq_len and char in TOKEN2INT_LOOP:
                inputs[idx, i, 7 + TOKEN2INT_LOOP[char]] = 1.0

        # 4. Pair Index & Mask
        p_idx, p_mask = get_pair_index_and_mask(struct)
        pair_indices[idx, :] = p_idx
        pair_masks[idx, :] = p_mask

        # 5. Targets (if not test)
        if not is_test:
            for k, col in enumerate(target_cols):
                val_list = row[col]
                # val_list is a list or numpy array. Length is usually 68.
                # We copy it into the (107,) buffer.
                length = min(len(val_list), seq_len)
                targets[idx, :length, k] = val_list[:length]

    return {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


def load_dataset(mode="train", load_cached_data=True, debug_samples=None):
    """
    Loads data for a specific mode ('train', 'val', 'test').
    Handles caching and processing from Parquet files.
    """
    # Determine paths
    if mode == "train":
        metadata_path = Config.train_metadata_path
        cache_path = Config.train_cache_path
        is_test = False
    elif mode == "val":
        metadata_path = Config.val_metadata_path
        cache_path = Config.val_cache_path
        is_test = False
    elif mode == "test":
        metadata_path = Config.test_metadata_path
        cache_path = Config.test_cache_path
        is_test = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Try Loading Cache
    if load_cached_data:
        data_dict = load_cache(cache_path)
        if data_dict is not None:
            print(f"Loaded {mode} data from cache: {cache_path}")
            # Handle debug subsampling on cached data
            if debug_samples is not None:
                for k in data_dict:
                    data_dict[k] = data_dict[k][:debug_samples]
            return data_dict

    # 2. Process from Scratch
    print(f"Processing {mode} data from {metadata_path}...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)

    if debug_samples is not None:
        df = df.iloc[:debug_samples]

    data_dict = preprocess_data(df, is_test=is_test)

    # 3. Save Cache (only if not debugging, to avoid overwriting full cache with partial data)
    # Or save to a separate debug cache? For simplicity, we only save if full dataset.
    if debug_samples is None:
        save_cache(cache_path, data_dict)
        print(f"Saved {mode} data to cache: {cache_path}")

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.from_numpy(data_dict["inputs"]).float()
        self.pair_indices = torch.from_numpy(data_dict["pair_indices"]).long()
        self.pair_masks = torch.from_numpy(data_dict["pair_masks"]).float()
        self.targets = torch.from_numpy(data_dict["targets"]).float()
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],  # (107, 14)
            "pair_indices": self.pair_indices[idx],  # (107,)
            "pair_masks": self.pair_masks[idx],  # (107,)
            "targets": self.targets[idx],  # (107, 5)
            "id": self.ids[idx],
        }


def get_dataloaders(debug_samples=None, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load Data
    train_data = load_dataset("train", load_cached_data, debug_samples)
    val_data = load_dataset("val", load_cached_data, debug_samples)
    test_data = load_dataset("test", load_cached_data, debug_samples)

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True if Config.device == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True if Config.device == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True if Config.device == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
