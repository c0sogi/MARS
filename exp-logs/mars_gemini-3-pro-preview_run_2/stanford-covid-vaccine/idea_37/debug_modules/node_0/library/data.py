import os
import ast
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
    PREPROCESSED_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SUBSET_SIZE,
)

# =============================================================================
# Vocabularies & Constants
# =============================================================================
SEQ_VOCAB = ["A", "G", "C", "U"]
STRUCT_VOCAB = ["(", ".", ")"]
LOOP_VOCAB = ["S", "M", "I", "B", "H", "E", "X"]

# Maps for fast lookup
SEQ_TO_INT = {c: i for i, c in enumerate(SEQ_VOCAB)}
STRUCT_TO_INT = {c: i for i, c in enumerate(STRUCT_VOCAB)}
LOOP_TO_INT = {c: i for i, c in enumerate(LOOP_VOCAB)}

# =============================================================================
# Helper Functions
# =============================================================================


def get_partner_map(structure):
    """
    Parses dot-bracket structure to find pairing partners.
    Returns:
        partner_indices: (L,) array. If i is paired with j, arr[i] = j.
                         If unpaired, arr[i] = i (safe dummy index).
        pairing_mask: (L,) array. 1 if paired, 0 if unpaired.
    """
    L = len(structure)
    partner_indices = np.arange(L)  # Default to self (safe for gather)
    pairing_mask = np.zeros(L, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i
                pairing_mask[i] = 1.0
                pairing_mask[j] = 1.0

    return partner_indices, pairing_mask


def one_hot_encode(seq, vocab_map, length):
    """
    One-hot encodes a sequence string.
    """
    res = np.zeros((length, len(vocab_map)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in vocab_map:
            res[i, vocab_map[char]] = 1.0
    return res


def get_partner_identity(sequence, partner_indices, pairing_mask, length):
    """
    Creates a one-hot encoding of the partner's base identity.
    If unpaired (mask=0), returns a zero vector.
    """
    # Dimensions: Length x 4 (A, G, C, U)
    res = np.zeros((length, len(SEQ_VOCAB)), dtype=np.float32)

    # Convert sequence to indices once
    seq_indices = [SEQ_TO_INT.get(c, -1) for c in sequence]

    for i in range(length):
        if pairing_mask[i] == 1.0:
            j = partner_indices[i]
            # Check bounds and validity
            if 0 <= j < length and 0 <= j < len(seq_indices):
                base_idx = seq_indices[j]
                if base_idx != -1:
                    res[i, base_idx] = 1.0
    return res


def parse_targets(row, target_cols, length):
    """
    Parses target columns from stringified lists in the dataframe.
    """
    # Shape: (Length, 5)
    targets = np.zeros((length, len(target_cols)), dtype=np.float32)

    for idx, col in enumerate(target_cols):
        val_str = row[col]
        try:
            # ast.literal_eval is safer than eval
            val_list = ast.literal_eval(val_str)
            # Targets are usually length 68, but we pad to 107 (SEQ_LENGTH)
            # The competition scoring only cares about the first 68,
            # but we keep the array full length for tensor consistency.
            valid_len = min(len(val_list), length)
            targets[:valid_len, idx] = val_list[:valid_len]
        except (ValueError, SyntaxError):
            # Handle cases where data might be missing or malformed
            pass

    return targets


# =============================================================================
# Preprocessing Logic
# =============================================================================


def preprocess_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from CSV, processes features, and caches to .npz.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            return (
                data["features"],
                data["partner_indices"],
                data["pairing_mask"],
                data["targets"],
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load CSV
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if DEBUG:
        df = df.head(DEBUG_SUBSET_SIZE)
        print(f"DEBUG MODE: Reduced dataset to {len(df)} samples.")

    num_samples = len(df)

    # 3. Initialize Arrays
    # Feature dim = 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
    feat_dim = len(SEQ_VOCAB) + len(STRUCT_VOCAB) + len(LOOP_VOCAB) + len(SEQ_VOCAB)

    features = np.zeros((num_samples, SEQ_LENGTH, feat_dim), dtype=np.float32)
    partner_indices = np.zeros((num_samples, SEQ_LENGTH), dtype=np.int64)
    pairing_mask = np.zeros((num_samples, SEQ_LENGTH), dtype=np.float32)
    targets = np.zeros((num_samples, SEQ_LENGTH, len(TARGET_COLS)), dtype=np.float32)
    ids = df["id"].values

    # 4. Iterate and Process
    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # A. Basic One-Hot Features
        oh_seq = one_hot_encode(seq, SEQ_TO_INT, SEQ_LENGTH)
        oh_struct = one_hot_encode(struct, STRUCT_TO_INT, SEQ_LENGTH)
        oh_loop = one_hot_encode(loop, LOOP_TO_INT, SEQ_LENGTH)

        # B. Partner Info
        p_idx, p_mask = get_partner_map(struct)
        # Pad or truncate to SEQ_LENGTH
        if len(p_idx) > SEQ_LENGTH:
            p_idx = p_idx[:SEQ_LENGTH]
            p_mask = p_mask[:SEQ_LENGTH]
        elif len(p_idx) < SEQ_LENGTH:
            # Pad with self-indices and 0 mask
            pad_len = SEQ_LENGTH - len(p_idx)
            pad_indices = np.arange(len(p_idx), SEQ_LENGTH)
            p_idx = np.concatenate([p_idx, pad_indices])
            p_mask = np.concatenate([p_mask, np.zeros(pad_len)])

        partner_indices[idx] = p_idx
        pairing_mask[idx] = p_mask

        # C. Partner Identity Feature
        oh_partner = get_partner_identity(seq, p_idx, p_mask, SEQ_LENGTH)

        # D. Concatenate Features
        # [Seq, Struct, Loop, PartnerID]
        sample_feat = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        features[idx] = sample_feat

        # E. Targets
        if not is_test:
            targets[idx] = parse_targets(row, TARGET_COLS, SEQ_LENGTH)

    # 5. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        partner_indices=partner_indices,
        pairing_mask=pairing_mask,
        targets=targets,
        ids=ids,
    )
    print(f"Saved processed data to {cache_path}")

    return features, partner_indices, pairing_mask, targets, ids


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, pairing_mask, targets, ids=None):
        self.features = torch.from_numpy(features).float()
        self.partner_indices = torch.from_numpy(partner_indices).long()
        self.pairing_mask = torch.from_numpy(pairing_mask).float()
        self.targets = torch.from_numpy(targets).float()
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Returns:
        # x: (SeqLen, FeatDim)
        # p_idx: (SeqLen,)
        # p_mask: (SeqLen,)
        # y: (SeqLen, 5)
        return (
            self.features[idx],
            self.partner_indices[idx],
            self.pairing_mask[idx],
            self.targets[idx],
        )


# =============================================================================
# DataLoader Generators
# =============================================================================


def get_train_loader(load_cached_data=True):
    features, p_idx, p_mask, targets, _ = preprocess_data(
        TRAIN_CSV, TRAIN_CACHE, load_cached_data=load_cached_data, is_test=False
    )
    dataset = RNADataset(features, p_idx, p_mask, targets)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def get_val_loader(load_cached_data=True):
    features, p_idx, p_mask, targets, _ = preprocess_data(
        VAL_CSV, VAL_CACHE, load_cached_data=load_cached_data, is_test=False
    )
    dataset = RNADataset(features, p_idx, p_mask, targets)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def get_test_loader(load_cached_data=True):
    features, p_idx, p_mask, targets, ids = preprocess_data(
        TEST_CSV, TEST_CACHE, load_cached_data=load_cached_data, is_test=True
    )
    dataset = RNADataset(features, p_idx, p_mask, targets, ids=ids)
    return (
        DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        ids,
    )
