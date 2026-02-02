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


def compute_rwpe(adj_matrix, steps):
    """
    Computes Random Walk Positional Encoding (RWPE) features.
    Calculates the diagonal elements of the transition matrix powers T^k.

    T = D^-1 * A

    Args:
        adj_matrix (np.ndarray): Adjacency matrix (seq_len, seq_len).
        steps (list[int]): List of steps k to compute T^k diagonals for.

    Returns:
        np.ndarray: RWPE features of shape (seq_len, len(steps)).
    """
    seq_len = adj_matrix.shape[0]

    # Compute Degree Matrix D
    degrees = np.sum(adj_matrix, axis=1)

    # Handle isolated nodes (though rare in RNA backbone) to avoid div by zero
    # If degree is 0, we can leave it as 0 or set to 1 to avoid NaN.
    # For RNA, backbone ensures degree >= 1 usually.
    degrees[degrees == 0] = 1.0

    # Compute Transition Matrix T = D^-1 A
    # D_inv is diagonal matrix with 1/degree
    d_inv = np.diag(1.0 / degrees)
    T = np.matmul(d_inv, adj_matrix)

    rwpe_features = []

    # We need specific powers. Since steps are typically [1, 2, 4, 8, 16],
    # we can optimize by repeated squaring if the steps are powers of 2,
    # but for generality, we'll compute necessary powers.
    # Given the steps are small and fixed in Config, we can just compute.

    # Current power of T
    T_curr = T
    current_k = 1

    # Sort steps to compute incrementally
    sorted_steps = sorted(steps)
    max_step = sorted_steps[-1]

    # Dictionary to store diagonals for requested steps
    diagonals = {}

    # Iteratively multiply
    # Note: Matrix multiplication is O(N^3). N=107 is small.
    # We can just compute powers directly or incrementally.
    # Incremental: T^k = T^{k-1} @ T

    T_power = T.copy()
    if 1 in sorted_steps:
        diagonals[1] = np.diag(T_power)

    for k in range(2, max_step + 1):
        T_power = np.matmul(T_power, T)
        if k in sorted_steps:
            diagonals[k] = np.diag(T_power)

    # Collect results in order of 'steps'
    for k in steps:
        rwpe_features.append(diagonals[k])

    # Stack -> (len(steps), seq_len) -> transpose to (seq_len, len(steps))
    return np.stack(rwpe_features, axis=1)


def get_sinusoidal_encoding(structure, seq_len, embed_dim):
    """
    Generates Signed Sinusoidal Pairing Encodings.
    Encodes the signed distance d = j - i for paired bases (i, j).
    Unpaired bases have distance 0.

    Args:
        structure (str): Dot-bracket structure.
        seq_len (int): Sequence length.
        embed_dim (int): Output embedding dimension.

    Returns:
        np.ndarray: Encodings of shape (seq_len, embed_dim).
    """
    pairs = get_pair_map(structure)

    # Calculate signed distances
    # If i is paired with j, dist = j - i
    # If unpaired, dist = 0
    distances = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len):
        if i in pairs:
            distances[i] = pairs[i] - i
        else:
            distances[i] = 0.0

    # Compute Sinusoidal Encoding
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    position = distances[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

    pe = np.zeros((seq_len, embed_dim), dtype=np.float32)

    # Argument for sin/cos
    # (seq_len, 1) * (1, embed_dim/2) -> (seq_len, embed_dim/2)
    args = position * div_term

    pe[:, 0::2] = np.sin(args)
    pe[:, 1::2] = np.cos(args)

    return pe


def extract_features(df, split_name, load_cached=True):
    """
    Orchestrates the feature extraction process for a dataframe.
    Computes RWPE and Pair Encodings for all samples.
    Handles caching to disk.

    Args:
        df (pd.DataFrame): Dataframe containing 'structure' and 'seq_length'.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing:
            - 'rwpe': torch.Tensor (N, L, n_steps)
            - 'pair_enc': torch.Tensor (N, L, embed_dim)
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{split_name}_features.pt")

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

    rwpe_list = []
    pair_enc_list = []

    # Use Config values
    rwpe_steps = Config.RWPE_STEPS
    pair_dim = Config.EMBED_DIM_PAIR

    structures = df["structure"].values
    # Ensure seq_length is consistent or use column if variable (though Config says 107)
    # We'll use the actual length from the dataframe to be safe, or Config.SEQ_LENGTH

    for idx, struct in enumerate(structures):
        # Determine length (should be 107)
        slen = len(struct)

        # A. Adjacency & RWPE
        adj = structure_to_adj(struct, slen)
        rwpe = compute_rwpe(adj, rwpe_steps)  # (L, n_steps)

        # B. Pair Encoding
        p_enc = get_sinusoidal_encoding(struct, slen, pair_dim)  # (L, pair_dim)

        rwpe_list.append(rwpe)
        pair_enc_list.append(p_enc)

    # Stack into tensors
    rwpe_tensor = torch.tensor(np.array(rwpe_list), dtype=torch.float32)
    pair_enc_tensor = torch.tensor(np.array(pair_enc_list), dtype=torch.float32)

    result = {"rwpe": rwpe_tensor, "pair_enc": pair_enc_tensor}

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    try:
        torch.save(result, cache_file)
        print(f"Saved features to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return result
