import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import config

# =========================================================================
# Encoding Maps
# =========================================================================
NUC_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def get_pair_indices(structure):
    """
    Parses a dot-bracket structure string to find base pairs.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        indices (np.ndarray): (L,) array where indices[i] = j if (i, j) are paired.
                              If unpaired, indices[i] = 0 (dummy index).
        mask (np.ndarray): (L,) array, 1.0 if paired, 0.0 if unpaired.
    """
    L = len(structure)
    indices = np.zeros(L, dtype=np.int64)
    mask = np.zeros(L, dtype=np.float32)

    stack = []
    pairs = {}

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i

    for i in range(L):
        if i in pairs:
            indices[i] = pairs[i]
            mask[i] = 1.0
        else:
            indices[i] = 0  # Point to 0, mask will zero out the contribution
            mask[i] = 0.0

    return indices, mask


def one_hot_encode(seq, map_dict, num_classes):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    L = len(seq)
    encoding = np.zeros((L, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in map_dict:
            encoding[i, map_dict[char]] = 1.0
    return encoding


def process_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into dictionary of numpy arrays.
    """
    num_samples = len(df)
    seq_len = config.SEQ_LEN
    input_dim = config.INPUT_DIM  # 14

    # Pre-allocate arrays
    features = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    if not is_test:
        # Targets: (N, 68, 5)
        targets = np.zeros(
            (num_samples, config.SEQ_SCORED, config.NUM_TARGETS), dtype=np.float32
        )
    else:
        targets = None

    for idx, row in df.iterrows():
        # 1. Input Features
        # Sequence (4 channels)
        seq_enc = one_hot_encode(row["sequence"], NUC_MAP, 4)
        # Structure (3 channels)
        struct_enc = one_hot_encode(row["structure"], STRUCT_MAP, 3)
        # Loop Type (7 channels)
        loop_enc = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate along channel dimension: (L, 14)
        features[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 2. Structural Adjacency
        p_idx, p_mask = get_pair_indices(row["structure"])
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets (Training/Validation only)
        if not is_test:
            for t_i, col in enumerate(config.TARGET_COLS):
                val_list = row[col]
                # Ensure we only take the scored positions
                length = min(len(val_list), config.SEQ_SCORED)
                targets[idx, :length, t_i] = val_list[:length]

    return {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


def load_or_process_data(split_name, parquet_path, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from Parquet and caches it.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split_name}_cache.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split_name} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            result = {
                "features": data["features"],
                "pair_indices": data["pair_indices"],
                "pair_masks": data["pair_masks"],
                "ids": data["ids"],
            }

            if "targets" in data:
                result["targets"] = data["targets"]
            else:
                result["targets"] = None

            return result
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing from scratch.")

    # Process from scratch
    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    is_test = split_name == "test"

    processed = process_dataframe(df, is_test=is_test)

    # Save to cache
    save_dict = {
        "features": processed["features"],
        "pair_indices": processed["pair_indices"],
        "pair_masks": processed["pair_masks"],
        "ids": processed["ids"],
    }

    if processed["targets"] is not None:
        save_dict["targets"] = processed["targets"]

    np.savez(cache_path, **save_dict)
    print(f"Saved {split_name} data to cache: {cache_path}")

    return processed


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.features = torch.from_numpy(data_dict["features"]).float()
        self.pair_indices = torch.from_numpy(data_dict["pair_indices"]).long()
        self.pair_masks = torch.from_numpy(data_dict["pair_masks"]).float()
        self.ids = data_dict["ids"]

        if data_dict["targets"] is not None:
            self.targets = torch.from_numpy(data_dict["targets"]).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            "sequence": self.features[idx],  # (107, 14)
            "pair_indices": self.pair_indices[idx],  # (107,)
            "pair_mask": self.pair_masks[idx],  # (107,)
            "id": self.ids[idx],
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]  # (68, 5)

        return item


# =========================================================================
# DataLoader Factory
# =========================================================================


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Generates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from .npz cache.
        batch_size (int, optional): Batch size override.
        num_workers (int, optional): Num workers override.

    Returns:
        train_loader, val_loader, test_loader
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    nw = num_workers if num_workers is not None else config.NUM_WORKERS

    # Load raw data dictionaries
    train_data = load_or_process_data("train", config.TRAIN_PATH, load_cached_data)
    val_data = load_or_process_data("val", config.VAL_PATH, load_cached_data)
    test_data = load_or_process_data("test", config.TEST_PATH, load_cached_data)

    # Initialize Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    return train_loader, val_loader, test_loader
