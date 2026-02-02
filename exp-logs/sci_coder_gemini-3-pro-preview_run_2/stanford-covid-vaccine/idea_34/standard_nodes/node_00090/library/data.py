import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_CACHE,
    VAL_CACHE,
    TEST_CACHE,
    SEQ_LENGTH,
    TARGET_COLS,
    BATCH_SIZE,
    NUM_WORKERS,
)
from library.utils import parse_list_column

# =============================================================================
# ENCODING MAPS
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# Dimensions
DIM_SEQ = 4
DIM_STRUCT = 3
DIM_LOOP = 7
DIM_PARTNER = 4  # Partner identity is also a base (A, G, C, U)


def get_partner_indices(structure):
    """
    Parses dot-bracket structure to find partner indices.
    Returns an array where arr[i] is the index of the partner of base i.
    If unpaired, arr[i] is -1.
    """
    partner_indices = np.full(len(structure), -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i

    return partner_indices


def one_hot_encode(seq, mapping, depth):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    encoding = np.zeros((len(seq), depth), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def preprocess_data(csv_path, is_test=False):
    """
    Loads CSV, generates features and targets, and returns numpy arrays.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Pre-allocate arrays
    n_samples = len(df)

    # Feature dimensions: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    total_features = DIM_SEQ + DIM_STRUCT + DIM_LOOP + DIM_PARTNER

    inputs = np.zeros((n_samples, SEQ_LENGTH, total_features), dtype=np.float32)
    partner_indices_arr = np.zeros((n_samples, SEQ_LENGTH), dtype=np.int32)
    targets = np.zeros((n_samples, SEQ_LENGTH, len(TARGET_COLS)), dtype=np.float32)
    masks = np.zeros((n_samples, SEQ_LENGTH), dtype=np.float32)
    ids = []

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Basic One-Hot Encodings
        ohe_seq = one_hot_encode(sequence, SEQ_MAP, DIM_SEQ)
        ohe_struct = one_hot_encode(structure, STRUCT_MAP, DIM_STRUCT)
        ohe_loop = one_hot_encode(loop_type, LOOP_MAP, DIM_LOOP)

        # 2. Partner Indices
        p_indices = get_partner_indices(structure)
        partner_indices_arr[idx] = p_indices

        # 3. Partner Identity
        # Create one-hot of the partner base. If unpaired (-1), it remains all zeros.
        ohe_partner = np.zeros((SEQ_LENGTH, DIM_PARTNER), dtype=np.float32)
        for i, p_idx in enumerate(p_indices):
            if p_idx != -1:
                partner_base = sequence[p_idx]
                if partner_base in SEQ_MAP:
                    ohe_partner[i, SEQ_MAP[partner_base]] = 1.0

        # Concatenate all features
        # Shape: (107, 18)
        inputs[idx] = np.concatenate(
            [ohe_seq, ohe_struct, ohe_loop, ohe_partner], axis=1
        )

        # 4. Targets and Masks
        if not is_test:
            # Parse targets
            # Targets are provided for the first 'seq_scored' positions (usually 68)
            # We pad them to 107
            seq_scored = row.get("seq_scored", 68)

            for t_i, col in enumerate(TARGET_COLS):
                val_arr = parse_list_column(row[col])
                length = len(val_arr)
                if length > 0:
                    # Fill available data
                    limit = min(length, SEQ_LENGTH)
                    targets[idx, :limit, t_i] = val_arr[:limit]

            # Create mask
            # 1.0 for scored positions, 0.0 for others
            masks[idx, :seq_scored] = 1.0
        else:
            # For test set, we still might want a mask based on seq_scored for submission formatting
            seq_scored = row.get("seq_scored", 68)
            masks[idx, :seq_scored] = 1.0

        ids.append(row["id"])

    return inputs, partner_indices_arr, targets, masks, np.array(ids)


def load_and_cache_data(split, load_cached_data=True):
    """
    Loads data from cache if available, otherwise preprocesses and caches it.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays.
    """
    # Determine paths based on split
    if split == "train":
        csv_path = TRAIN_CSV
        cache_path = TRAIN_CACHE
        is_test = False
    elif split == "val":
        csv_path = VAL_CSV
        cache_path = VAL_CACHE
        is_test = False
    elif split == "test":
        csv_path = TEST_CSV
        cache_path = TEST_CACHE
        is_test = True
    else:
        raise ValueError(f"Invalid split: {split}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            "inputs": data["inputs"],
            "partner_indices": data["partner_indices"],
            "targets": data["targets"],
            "masks": data["masks"],
            "ids": data["ids"],
        }

    # Preprocess from scratch
    print(f"Preprocessing {split} data from {csv_path}...")
    inputs, partner_indices, targets, masks, ids = preprocess_data(
        csv_path, is_test=is_test
    )

    # Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    np.savez(
        cache_path,
        inputs=inputs,
        partner_indices=partner_indices,
        targets=targets,
        masks=masks,
        ids=ids,
    )

    return {
        "inputs": inputs,
        "partner_indices": partner_indices,
        "targets": targets,
        "masks": masks,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        data = load_and_cache_data(split, load_cached_data)

        self.inputs = torch.from_numpy(data["inputs"])
        self.partner_indices = torch.from_numpy(data["partner_indices"]).long()
        self.targets = torch.from_numpy(data["targets"])
        self.masks = torch.from_numpy(data["masks"])
        self.ids = data["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Return tuple: (inputs, partner_indices, targets, mask)
        # inputs: (107, 18)
        # partner_indices: (107,)
        # targets: (107, 5)
        # mask: (107,)
        return (
            self.inputs[idx],
            self.partner_indices[idx],
            self.targets[idx],
            self.masks[idx],
        )


def get_dataloaders(load_cached_data=True, batch_size=BATCH_SIZE):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    train_ds = RNADataset("train", load_cached_data=load_cached_data)
    val_ds = RNADataset("val", load_cached_data=load_cached_data)
    test_ds = RNADataset("test", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
