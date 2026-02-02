import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import parse_list_column


class RNADataset(Dataset):
    def __init__(self, features, partner_indices, targets=None):
        """
        PyTorch Dataset for RNA data.

        Args:
            features (np.ndarray): Input features of shape (N, Channels, Seq_Len).
            partner_indices (np.ndarray): Partner indices of shape (N, Seq_Len).
            targets (np.ndarray, optional): Target values of shape (N, Seq_Len, 5).
        """
        self.features = features
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features: [C, L]
        # partner_indices: [L]
        # targets: [L, 5] (if available)

        x = torch.tensor(self.features[idx], dtype=torch.float32)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_idx, y
        else:
            return x, p_idx


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find base pairs.
    Returns a mapping {index: partner_index}. Unpaired bases are not in the dict.
    """
    pairs = {}
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def process_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Processes RNA data: generates one-hot encodings, partner features, and targets.
    Uses caching to avoid re-computation.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_path (str): Path to the .npz cache file.
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        tuple: (features, partner_indices, targets_or_ids)
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            if is_test:
                return data["features"], data["partner_indices"], data["ids"]
            else:
                return data["features"], data["partner_indices"], data["targets"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Debug subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    # Mappings
    seq_map = {c: i for i, c in enumerate("AGCU")}
    struct_map = {c: i for i, c in enumerate("().")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    features_list = []
    partner_indices_list = []
    targets_list = []
    ids_list = df["id"].values if "id" in df.columns else []

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        length = len(seq)

        # 1. Sequence One-Hot (4 channels)
        seq_oh = np.zeros((4, length), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in seq_map:
                seq_oh[seq_map[char], i] = 1.0

        # 2. Structure One-Hot (3 channels)
        struct_oh = np.zeros((3, length), dtype=np.float32)
        for i, char in enumerate(struct):
            if char in struct_map:
                struct_oh[struct_map[char], i] = 1.0

        # 3. Loop Type One-Hot (7 channels)
        loop_oh = np.zeros((7, length), dtype=np.float32)
        for i, char in enumerate(loop):
            if char in loop_map:
                loop_oh[loop_map[char], i] = 1.0

        # 4. Partner Identity (4 channels) & Indices
        pairs = get_structure_pairs(struct)
        partner_oh = np.zeros((4, length), dtype=np.float32)
        p_indices = np.full(length, -1, dtype=np.int32)

        for i in range(length):
            if i in pairs:
                j = pairs[i]
                p_indices[i] = j
                partner_char = seq[j]
                if partner_char in seq_map:
                    partner_oh[seq_map[partner_char], i] = 1.0

        # Concatenate all features: [18, Length]
        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner) = 18
        sample_features = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_oh], axis=0
        )
        features_list.append(sample_features)
        partner_indices_list.append(p_indices)

        # Targets
        if not is_test:
            # Parse stringified lists
            t_react = parse_list_column(row["reactivity"])
            t_mg_ph10 = parse_list_column(row["deg_Mg_pH10"])
            t_ph10 = parse_list_column(row["deg_pH10"])
            t_mg_50c = parse_list_column(row["deg_Mg_50C"])
            t_50c = parse_list_column(row["deg_50C"])

            # Stack: [Length, 5]
            # Initialize with zeros (padding for positions > 68)
            sample_targets = np.zeros((length, 5), dtype=np.float32)

            # Fill available data (usually first 68 positions)
            valid_len = len(t_react)
            if valid_len > 0:
                sample_targets[:valid_len, 0] = t_react
                sample_targets[:valid_len, 1] = t_mg_ph10
                sample_targets[:valid_len, 2] = t_ph10
                sample_targets[:valid_len, 3] = t_mg_50c
                sample_targets[:valid_len, 4] = t_50c

            targets_list.append(sample_targets)

    # Convert to numpy arrays
    features = np.array(features_list)
    partner_indices = np.array(partner_indices_list)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if is_test:
        np.savez(
            cache_path, features=features, partner_indices=partner_indices, ids=ids_list
        )
        return features, partner_indices, ids_list
    else:
        targets = np.array(targets_list)
        np.savez(
            cache_path,
            features=features,
            partner_indices=partner_indices,
            targets=targets,
        )
        return features, partner_indices, targets
