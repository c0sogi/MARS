import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def get_pair_map(structure):
    """
    Parses a dot-bracket structure string to identify base pairs.
    Returns a dictionary mapping index -> paired_index.
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


def structure_to_adj(structure, seq_len):
    """
    Converts a dot-bracket structure string into an adjacency matrix.
    Edges include:
    1. Backbone connections (i, i+1)
    2. Hydrogen bonds (base pairs)

    Args:
        structure (str): Dot-bracket string.
        seq_len (int): Length of sequence.

    Returns:
        np.ndarray: Adjacency matrix of shape (seq_len, seq_len).
    """
    adj = np.zeros((seq_len, seq_len), dtype=np.float32)

    # 1. Backbone connections
    # Nodes i and i+1 are connected
    indices = np.arange(seq_len - 1)
    adj[indices, indices + 1] = 1.0
    adj[indices + 1, indices] = 1.0

    # 2. Hydrogen bonds
    pairs = get_pair_map(structure)
    for i, j in pairs.items():
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


def get_distance_indices(structure, seq_len):
    """
    Generates Discrete Pairing Indices.
    Maps signed distance d = j - i to a discrete index.
    Unpaired bases have distance 0.
    Indices are offset by 128 to handle negative values.

    Args:
        structure (str): Dot-bracket structure.
        seq_len (int): Sequence length.

    Returns:
        np.ndarray: Indices of shape (seq_len,).
    """
    pairs = get_pair_map(structure)
    indices = np.zeros(seq_len, dtype=np.int64)

    # Offset to handle negative distances (range approx -107 to 107)
    # 0 -> 128
    OFFSET = 128

    for i in range(seq_len):
        if i in pairs:
            dist = pairs[i] - i
            # Clamp to vocabulary limits just in case
            idx = dist + OFFSET
            idx = max(0, min(idx, Config.VOCAB_SIZE_PAIR - 1))
            indices[i] = idx
        else:
            indices[i] = OFFSET  # Distance 0

    return indices


def extract_features(df, split_name, load_cached=True):
    """
    Orchestrates the feature extraction process for a dataframe.
    Computes Pair Indices for all samples.
    Handles caching to disk.

    Args:
        df (pd.DataFrame): Dataframe containing 'structure' and 'seq_length'.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing:
            - 'pair_indices': torch.Tensor (N, L)
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_features_v2.pt")

    # 1. Try to load from cache
    if load_cached and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        try:
            data = torch.load(cache_file)
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute features
    print(f"Computing geometric features for {split_name} set ({len(df)} samples)...")

    pair_indices_list = []

    structures = df["structure"].values

    for idx, struct in enumerate(structures):
        slen = len(struct)

        # Pair Indices (Discrete)
        p_idx = get_distance_indices(struct, slen)
        pair_indices_list.append(p_idx)

    # Stack into tensors
    pair_indices_tensor = torch.tensor(np.array(pair_indices_list), dtype=torch.long)

    result = {"pair_indices": pair_indices_tensor}

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    try:
        torch.save(result, cache_file)
        print(f"Saved features to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return result
