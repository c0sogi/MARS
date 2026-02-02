import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import parse_list_column

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================


def get_structure_adj(structure, seq_length):
    """
    Parses dot-bracket structure to find partner indices.
    Returns:
        partner_indices: Array of shape (L,) where value is index of partner or -1.
    """
    stack = []
    partner_indices = np.full(seq_length, -1, dtype=np.int32)

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i
    return partner_indices


def one_hot_encode(seq, vocab):
    """
    One-hot encodes a sequence string based on a vocab dictionary.
    Returns:
        one_hot: Array of shape (L, len(vocab))
    """
    mapping = {char: i for i, char in enumerate(vocab)}
    seq_len = len(seq)
    vocab_size = len(vocab)
    one_hot = np.zeros((seq_len, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def process_data(csv_path, is_test=False, load_cached_data=True, cache_name="data.npz"):
    """
    Loads metadata, processes features, and caches results.

    Args:
        csv_path (str): Path to the CSV file (train/val/test).
        is_test (bool): Whether this is the test set (no targets).
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_name (str): Filename for the cache.

    Returns:
        dict: Dictionary containing processed arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Convert NpzFile to dict to ensure it's accessible after close
            return {key: data[key] for key in data}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Vocabularies
    seq_vocab = "AGUC"
    struct_vocab = "()."
    loop_vocab = "SMIBHEX"

    # Containers
    ids = []
    features = []
    partner_indices_list = []
    targets = []

    # Target columns in the CSV
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        sample_id = row["id"]

        # A. Base One-Hot Features
        ohe_seq = one_hot_encode(seq, seq_vocab)  # (L, 4)
        ohe_struct = one_hot_encode(struct, struct_vocab)  # (L, 3)
        ohe_loop = one_hot_encode(loop, loop_vocab)  # (L, 7)

        # B. Partner Indices
        p_idx = get_structure_adj(struct, Config.SEQ_LENGTH)

        # C. Partner Identity (Explicit Injection)
        # If i is paired with j, get one-hot of base at j. Else 0 vector.
        ohe_partner = np.zeros((Config.SEQ_LENGTH, 4), dtype=np.float32)
        for i, partner_i in enumerate(p_idx):
            if partner_i != -1:
                ohe_partner[i] = ohe_seq[partner_i]

        # D. Concatenate all features
        # Shape: (L, 4+3+7+4) = (L, 18)
        sample_features = np.concatenate(
            [ohe_seq, ohe_struct, ohe_loop, ohe_partner], axis=1
        )

        ids.append(sample_id)
        features.append(sample_features)
        partner_indices_list.append(p_idx)

        # E. Process Targets (Train/Val only)
        if not is_test:
            sample_targets = []
            for col in target_cols:
                # Use helper from utils to parse stringified list and pad
                val_array = parse_list_column(row[col], length=Config.SEQ_LENGTH)
                sample_targets.append(val_array)

            # Stack targets: (5, L) -> Transpose to (L, 5) if preferred,
            # but usually (5, L) is better for Conv1d output matching.
            # However, standard dataset usually returns (Channels, Length).
            # Let's stack as (5, L).
            targets.append(np.stack(sample_targets, axis=0))

    # Convert lists to numpy arrays
    features = np.array(features, dtype=np.float32)  # (N, L, 18)
    partner_indices_list = np.array(partner_indices_list, dtype=np.int32)  # (N, L)
    ids = np.array(ids)

    data_dict = {
        "ids": ids,
        "features": features,
        "partner_indices": partner_indices_list,
    }

    if not is_test:
        targets = np.array(targets, dtype=np.float32)  # (N, 5, L)
        data_dict["targets"] = targets

    # 3. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


# ==================================================================================
# DATASET CLASS
# ==================================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        """
        PyTorch Dataset for RNA data.

        Args:
            data_dict (dict): Dictionary returned by process_data.
            is_test (bool): If True, __getitem__ returns ids instead of targets.
        """
        self.features = data_dict["features"]
        self.partner_indices = data_dict["partner_indices"]
        self.ids = data_dict["ids"]
        self.is_test = is_test

        if not is_test:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features stored as (L, C) in numpy array from process_data logic above?
        # Wait, in process_data: sample_features is concatenated axis=1 -> (L, 18).
        # PyTorch Conv1d expects (C, L).

        # Get feature: (L, 18)
        feat_np = self.features[idx]
        # Convert to tensor and permute to (18, L)
        feat = torch.tensor(feat_np, dtype=torch.float32).permute(1, 0)

        # Partner indices: (L,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.is_test:
            return feat, p_idx, self.ids[idx]
        else:
            # Targets stored as (5, L) in process_data
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return feat, p_idx, target
