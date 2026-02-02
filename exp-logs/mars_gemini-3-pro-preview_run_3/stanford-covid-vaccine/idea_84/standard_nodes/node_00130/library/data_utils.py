import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCTURE_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# ==========================================
# Helper Functions
# ==========================================


def get_pair_index(structure):
    """
    Parses a dot-bracket structure string and returns an array of pair indices.
    If index i is paired with j, arr[i] = j.
    If index i is unpaired, arr[i] = -1.
    """
    n = len(structure)
    pair_index = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_index[i] = j
                pair_index[j] = i
    return pair_index


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns shape (Length, VocabSize)
    """
    arr = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_dataframe(df):
    """
    Process dataframe into numpy arrays for features, pair indices, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Features: (N, L, 14) -> 4 (Seq) + 3 (Struct) + 7 (Loop)
    features = np.zeros((num_samples, seq_len, Config.INPUT_DIM), dtype=np.float32)

    # Pair Indices: (N, L)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: (N, L, 5) - Initialize with zeros
    # We will fill this if the columns exist in the dataframe
    targets = None
    has_targets = all(col in df.columns for col in Config.TARGET_COLS)

    if has_targets:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    ids = df["id"].values

    for i, row in df.iterrows():
        # 1. Sequence One-Hot (4)
        seq_oh = one_hot_encode(row["sequence"], NUCLEOTIDE_MAP, 4)

        # 2. Structure One-Hot (3)
        struct_oh = one_hot_encode(row["structure"], STRUCTURE_MAP, 3)

        # 3. Loop Type One-Hot (7)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_TYPE_MAP, 7)

        # Concatenate features along channel dimension
        features[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. Pair Indices
        pair_indices[i] = get_pair_index(row["structure"])

        # 5. Targets (if available)
        if has_targets:
            # Targets are provided as lists (length usually 68).
            # We pad them to seq_len (107) with zeros (already initialized).
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    # Safety check to not exceed sequence length
                    valid_len = min(length, seq_len)
                    targets[i, :valid_len, t_idx] = val_list[:valid_len]

    return {
        "features": features,
        "pair_indices": pair_indices,
        "targets": targets,
        "ids": ids,
    }


# ==========================================
# Dataset Class
# ==========================================


class RNADataset(Dataset):
    def __init__(self, features, pair_indices, targets=None, ids=None):
        self.features = features
        self.pair_indices = pair_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features: (L, 14)
        # pair_indices: (L,)
        # targets: (L, 5) or None

        item = {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


# ==========================================
# Data Loading & Caching
# ==========================================


def load_data(split, load_cached_data=True, max_samples=None):
    """
    Loads data for a specific split (train, val, test).

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache.
        max_samples (int, optional): If set, loads only a subset (disables caching).

    Returns:
        dict: Dictionary containing numpy arrays for features, pair_indices, targets, ids.
    """
    cache_filename = f"{split}_data.npz"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # If max_samples is set, we skip loading from main cache to avoid using full data
    # and we skip saving to main cache to avoid overwriting it with partial data.
    use_cache = load_cached_data and (max_samples is None)

    # 1. Try Loading from Cache
    if use_cache and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            result = {
                "features": data["features"],
                "pair_indices": data["pair_indices"],
                "ids": data["ids"],
            }
            # Load targets if they exist (train/val)
            if (
                "targets" in data and data["targets"].ndim > 0
            ):  # check if not None/empty
                result["targets"] = data["targets"]
            else:
                result["targets"] = None
            return result
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Processing from scratch.")

    # 2. Process from Scratch
    print(f"Processing {split} data from scratch...")

    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_parquet(path)

    # Handle subsampling for debugging
    if max_samples is not None:
        print(f"Subsampling {split} to {max_samples} samples.")
        df = df.head(max_samples).reset_index(drop=True)

    processed = process_dataframe(df)

    # 3. Save to Cache (only if not subsampled)
    if use_cache:
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        save_dict = {
            "features": processed["features"],
            "pair_indices": processed["pair_indices"],
            "ids": processed["ids"],
        }
        if processed["targets"] is not None:
            save_dict["targets"] = processed["targets"]

        np.savez(cache_path, **save_dict)
        print(f"Saved {split} data to cache: {cache_path}")

    return processed


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_samples=None,
):
    """
    Creates and returns dataloaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.
        max_samples (int, optional): Limit number of samples for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_data = load_data("train", load_cached_data, max_samples)
    val_data = load_data("val", load_cached_data, max_samples)
    test_data = load_data("test", load_cached_data, max_samples)

    # Create Datasets
    train_dataset = RNADataset(
        train_data["features"],
        train_data["pair_indices"],
        train_data["targets"],
        train_data["ids"],
    )

    val_dataset = RNADataset(
        val_data["features"],
        val_data["pair_indices"],
        val_data["targets"],
        val_data["ids"],
    )

    test_dataset = RNADataset(
        test_data["features"],
        test_data["pair_indices"],
        None,  # No targets for test
        test_data["ids"],
    )

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
