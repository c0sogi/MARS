import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_TRAIN,
    CACHE_VAL,
    CACHE_TEST,
    SEQ_LEN,
    SCORED_LEN,
    TARGET_COLS,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import parse_list_column, get_structure_adj

# =============================================================================
# MAPPINGS
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def one_hot(seq, mapping, num_classes):
    """
    Converts a sequence string into a one-hot numpy array.
    Shape: (Length, Num_Classes)
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def get_features(sequence, structure, loop_type):
    """
    Generates the input feature tensor for a single sample.

    Channels:
    - 0-3: Sequence One-Hot (A, G, U, C)
    - 4-6: Structure One-Hot ((, ), .)
    - 7-13: Loop Type One-Hot (S, M, I, B, H, E, X)
    - 14-18: Partner Identity One-Hot (A, G, U, C, None)

    Returns:
        features: (19, L) float32 array
        partner_indices: (L,) int64 array
    """
    length = len(sequence)

    # 1. Basic One-Hot Encodings
    seq_oh = one_hot(sequence, SEQ_MAP, 4)  # (L, 4)
    struct_oh = one_hot(structure, STRUCT_MAP, 3)  # (L, 3)
    loop_oh = one_hot(loop_type, LOOP_MAP, 7)  # (L, 7)

    # 2. Partner Indices
    partner_indices = get_structure_adj(structure)  # (L,)

    # 3. Partner Identity
    # 5 channels: A, G, U, C, No-Partner
    partner_identity_oh = np.zeros((length, 5), dtype=np.float32)

    for i in range(length):
        partner_idx = partner_indices[i]
        if partner_idx != -1:
            # Get the base at the partner index
            partner_base = sequence[partner_idx]
            if partner_base in SEQ_MAP:
                partner_identity_oh[i, SEQ_MAP[partner_base]] = 1.0
        else:
            # No partner (index 4)
            partner_identity_oh[i, 4] = 1.0

    # 4. Concatenate
    # Stack along channel dimension: (L, C) -> (C, L)
    # Total channels: 4 + 3 + 7 + 5 = 19
    combined = np.concatenate([seq_oh, struct_oh, loop_oh, partner_identity_oh], axis=1)
    features = combined.transpose(1, 0)  # (19, L)

    return features, partner_indices


def process_data(csv_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from CSV, generates features/targets, and handles caching.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path)
        return data["features"], data["partner_indices"], data["targets"], data["ids"]

    # 2. Process from Scratch
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    features_list = []
    partner_indices_list = []
    targets_list = []
    ids_list = []

    for _, row in df.iterrows():
        # Input Features
        feat, p_idx = get_features(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        features_list.append(feat)
        partner_indices_list.append(p_idx)
        ids_list.append(row["id"])

        # Targets
        # Initialize (5, 107) with zeros
        target_matrix = np.zeros((len(TARGET_COLS), SEQ_LEN), dtype=np.float32)

        if mode in ["train", "val"]:
            # Parse targets (available for first 68 positions)
            for i, col in enumerate(TARGET_COLS):
                val_arr = parse_list_column(row[col])
                # Safety check for length
                valid_len = min(len(val_arr), SEQ_LEN)
                target_matrix[i, :valid_len] = val_arr[:valid_len]

        targets_list.append(target_matrix)

    # Convert to numpy arrays
    features = np.array(features_list, dtype=np.float32)  # (N, 19, 107)
    partner_indices = np.array(partner_indices_list, dtype=np.int64)  # (N, 107)
    targets = np.array(targets_list, dtype=np.float32)  # (N, 5, 107)
    ids = np.array(ids_list)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        partner_indices=partner_indices,
        targets=targets,
        ids=ids,
    )
    print(f"Saved processed data to {cache_path}")

    return features, partner_indices, targets, ids


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, targets, ids):
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensors
        x = torch.from_numpy(self.features[idx])
        p_idx = torch.from_numpy(self.partner_indices[idx])
        y = torch.from_numpy(self.targets[idx])

        return x, p_idx, y


def get_loaders(load_cached_data=True, debug=False):
    """
    Generates DataLoaders for train, val, and test sets.
    """
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Train ---
    X_train, P_train, Y_train, ids_train = process_data(
        TRAIN_CSV, CACHE_TRAIN, mode="train", load_cached_data=load_cached_data
    )
    if debug:
        X_train, P_train, Y_train, ids_train = (
            X_train[:100],
            P_train[:100],
            Y_train[:100],
            ids_train[:100],
        )

    train_ds = RNADataset(X_train, P_train, Y_train, ids_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- Val ---
    X_val, P_val, Y_val, ids_val = process_data(
        VAL_CSV, CACHE_VAL, mode="val", load_cached_data=load_cached_data
    )
    if debug:
        X_val, P_val, Y_val, ids_val = X_val[:20], P_val[:20], Y_val[:20], ids_val[:20]

    val_ds = RNADataset(X_val, P_val, Y_val, ids_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    X_test, P_test, Y_test, ids_test = process_data(
        TEST_CSV, CACHE_TEST, mode="test", load_cached_data=load_cached_data
    )
    # Note: Y_test contains zeros, used as placeholder

    test_ds = RNADataset(X_test, P_test, Y_test, ids_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
