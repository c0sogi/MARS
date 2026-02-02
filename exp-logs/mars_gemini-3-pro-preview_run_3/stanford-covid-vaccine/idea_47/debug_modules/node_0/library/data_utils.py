import os
import numpy as np
import pandas as pd
from library.config import Config

# =========================================================================
# Feature Mappings
# =========================================================================
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCTURE_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_adjacency_indices(structure_str):
    """
    Parses a dot-bracket structure string to identify paired bases.

    Args:
        structure_str (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: An array of shape (seq_len,) where arr[i] is the index
                    of the base paired with i. If i is unpaired, arr[i] = -1.
    """
    seq_len = len(structure_str)
    indices = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
            else:
                # Unbalanced closing parenthesis, should not happen in valid data
                pass

    return indices


def process_sequence(sequence, structure, predicted_loop_type):
    """
    Generates a one-hot encoded feature matrix for a given RNA sequence.

    Channels:
        0-3: Nucleotide (A, G, C, U)
        4-6: Structure ((, ), .)
        7-13: Loop Type (S, M, I, B, H, E, X)

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket structure.
        predicted_loop_type (str): Loop type string.

    Returns:
        np.ndarray: Feature matrix of shape (seq_len, 14).
    """
    seq_len = len(sequence)
    # 4 (Seq) + 3 (Struct) + 7 (Loop) = 14 channels
    features = np.zeros((seq_len, 14), dtype=np.float32)

    for i in range(seq_len):
        # Nucleotide
        nt = sequence[i]
        if nt in NUCLEOTIDE_MAP:
            features[i, NUCLEOTIDE_MAP[nt]] = 1.0

        # Structure
        st = structure[i]
        if st in STRUCTURE_MAP:
            features[i, 4 + STRUCTURE_MAP[st]] = 1.0

        # Loop Type
        lt = predicted_loop_type[i]
        if lt in LOOP_TYPE_MAP:
            features[i, 7 + LOOP_TYPE_MAP[lt]] = 1.0

    return features


def load_data(split="train", load_cached_data=True):
    """
    Loads and processes the dataset for the specified split.
    Uses caching to speed up subsequent loads.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary containing:
            - 'features': np.ndarray (N, 107, 14)
            - 'adjacency': np.ndarray (N, 107)
            - 'ids': np.ndarray (N,)
            - 'targets': np.ndarray (N, 68, 5) [Only for train/val]
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif split == "val":
        meta_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif split == "test":
        meta_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Invalid split: {split}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data.")

    # 2. Process from scratch
    print(f"Processing {split} data from {meta_path}...")

    # Load metadata (Parquet preserves list structures)
    df = pd.read_parquet(meta_path)

    # Initialize containers
    features_list = []
    adjacency_list = []
    ids_list = []
    targets_list = []

    # Iterate over rows
    for idx, row in df.iterrows():
        # Extract ID
        ids_list.append(row["id"])

        # Extract Features
        feat = process_sequence(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        features_list.append(feat)

        # Extract Adjacency
        adj = get_adjacency_indices(row["structure"])
        adjacency_list.append(adj)

        # Extract Targets (only for train/val)
        if split in ["train", "val"]:
            # Targets are lists in the dataframe columns
            # We stack them to shape (seq_scored, num_targets) -> (68, 5)
            t_arrays = [
                np.array(row[col], dtype=np.float32) for col in Config.TARGET_COLS
            ]
            # Check length consistency (should be 68)
            # Stack: list of (68,) -> (5, 68) -> transpose to (68, 5)
            t_matrix = np.stack(t_arrays, axis=0).T
            targets_list.append(t_matrix)

    # Convert to numpy arrays
    data_dict = {
        "features": np.array(features_list, dtype=np.float32),
        "adjacency": np.array(adjacency_list, dtype=np.int32),
        "ids": np.array(ids_list, dtype=str),
    }

    if split in ["train", "val"]:
        data_dict["targets"] = np.array(targets_list, dtype=np.float32)

    # 3. Save to cache
    print(f"Saving {split} data to cache: {cache_path}")
    np.save(cache_path, data_dict)

    return data_dict
