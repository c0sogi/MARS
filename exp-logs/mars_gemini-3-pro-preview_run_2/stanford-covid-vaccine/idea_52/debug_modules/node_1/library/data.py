import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# ==========================================
# Constants & Maps
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_indices(structure):
    """
    Parses dot-bracket structure to find partner indices.
    Returns an array of shape (L,) where arr[i] is the index of the pair of i,
    or -1 if i is unpaired.
    """
    length = len(structure)
    indices = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
    return indices


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping.
    Returns shape (L, vocab_size).
    """
    length = len(seq)
    encoding = np.zeros((length, vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def preprocess_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from CSV, generates features, and caches them.

    Args:
        csv_path: Path to the source CSV file.
        cache_path: Path to the .npz cache file.
        load_cached_data: Whether to try loading from cache.
        is_test: If True, targets are not expected/processed (dummies used).

    Returns:
        features: (N, L, 18)
        partner_indices: (N, L)
        targets: (N, L, 5)
        ids: (N,)
    """
    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["features"],
                data["partner_indices"],
                data["targets"],
                data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process Data
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Pre-allocate lists
    all_features = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    # Target columns to parse
    target_cols = Config.ALL_TARGET_COLS

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]
        length = len(sequence)  # Should be 107

        # --- Feature Generation ---

        # 1. Base One-Hot encodings
        oh_seq = one_hot_encode(sequence, SEQ_MAP, 4)
        oh_struct = one_hot_encode(structure, STRUCT_MAP, 3)
        oh_loop = one_hot_encode(loop_type, LOOP_MAP, 7)

        # 2. Partner Indices
        p_indices = get_partner_indices(structure)

        # 3. Partner Identity
        # Create a (L, 4) tensor representing the identity of the paired base
        partner_identity = np.zeros((length, 4), dtype=np.float32)

        # Vectorized assignment for partner identity
        # Find positions that are paired
        paired_mask = p_indices != -1
        # Get indices of the partners
        partners = p_indices[paired_mask]
        # Assign the sequence encoding of the partners to the current positions
        partner_identity[paired_mask] = oh_seq[partners]

        # 4. Concatenate all features
        # (L, 4+3+7+4) = (L, 18)
        sample_features = np.concatenate(
            [oh_seq, oh_struct, oh_loop, partner_identity], axis=1
        )
        all_features.append(sample_features)
        all_partner_indices.append(p_indices)

        # --- Target Generation ---
        if not is_test:
            sample_targets = []
            for col in target_cols:
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(row[col])
                except (ValueError, SyntaxError):
                    val_list = []

                # Convert to array
                val_arr = np.array(val_list, dtype=np.float32)

                # Pad to sequence length (107)
                # Raw targets are usually length 68
                padded_target = np.zeros(length, dtype=np.float32)
                if len(val_arr) > 0:
                    valid_len = min(len(val_arr), length)
                    padded_target[:valid_len] = val_arr[:valid_len]

                sample_targets.append(padded_target)

            # Stack targets -> (L, 5)
            all_targets.append(np.stack(sample_targets, axis=1))
        else:
            # Dummy targets for test set
            all_targets.append(np.zeros((length, 5), dtype=np.float32))

    # Convert to numpy arrays
    features_np = np.array(all_features, dtype=np.float32)
    partner_indices_np = np.array(all_partner_indices, dtype=np.int32)
    targets_np = np.array(all_targets, dtype=np.float32)

    # 3. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features_np,
        partner_indices=partner_indices_np,
        targets=targets_np,
        ids=all_ids,
    )
    print(f"Saved processed data to {cache_path}")

    return features_np, partner_indices_np, targets_np, all_ids


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, targets, ids):
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Features: (L, C) -> Transpose to (C, L) for PyTorch Conv1d
        # shape: (18, 107)
        feat = torch.tensor(self.features[idx], dtype=torch.float32).transpose(0, 1)

        # Partner Indices: (L,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # Targets: (L, 5)
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)

        return feat, p_idx, tgt
