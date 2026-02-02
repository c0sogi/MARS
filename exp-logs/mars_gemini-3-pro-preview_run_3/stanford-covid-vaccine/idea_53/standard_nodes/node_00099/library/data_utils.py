import os
import numpy as np
import pandas as pd
from library.config import Config

# =========================================================================
# Constants & Mappings
# =========================================================================
NUC_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string into an adjacency index array.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (L,) where arr[i] is the index of the base
                    paired with i. Returns -1 if i is unpaired.
    """
    length = len(structure)
    adj = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i

    return adj


def one_hot_encode(df):
    """
    Encodes sequence, structure, and loop type into a one-hot tensor.

    Args:
        df (pd.DataFrame): Dataframe containing 'sequence', 'structure',
                           and 'predicted_loop_type' columns.

    Returns:
        np.ndarray: Tensor of shape (N, 107, 14).
                    Channels 0-3: Sequence (A, G, C, U)
                    Channels 4-6: Structure ((, ), .)
                    Channels 7-13: Loop Type (S, M, I, B, H, E, X)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM

    # Pre-allocate output array
    features = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)

    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values

    for i in range(num_samples):
        seq = sequences[i]
        struc = structures[i]
        loop = loop_types[i]

        for j in range(seq_len):
            # Sequence (0-3)
            if j < len(seq):
                char = seq[j]
                if char in NUC_MAP:
                    features[i, j, NUC_MAP[char]] = 1.0

            # Structure (4-6)
            if j < len(struc):
                char = struc[j]
                if char in STRUCT_MAP:
                    features[i, j, 4 + STRUCT_MAP[char]] = 1.0

            # Loop Type (7-13)
            if j < len(loop):
                char = loop[j]
                if char in LOOP_MAP:
                    features[i, j, 7 + LOOP_MAP[char]] = 1.0

    return features


def process_data(df, data_type):
    """
    Processes raw dataframe into model-ready numpy arrays.

    Args:
        df (pd.DataFrame): Raw dataframe.
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed arrays.
    """
    # 1. Input Features (One-Hot)
    inputs = one_hot_encode(df)

    # 2. Structure Adjacency (Pair Indices)
    # Process each structure string to get pair indices
    structures = df["structure"].values
    pair_indices_list = [get_structure_adj(s) for s in structures]
    pair_indices = np.array(pair_indices_list, dtype=np.int32)

    # 3. Targets (if available)
    targets = None
    if data_type in ["train", "val"]:
        # Extract target columns
        target_arrays = []
        for col in Config.TARGET_COLS:
            # Convert column of lists to numpy array (N, 68)
            # We use np.vstack to handle the list of lists correctly
            col_data = np.vstack(df[col].values).astype(np.float32)
            target_arrays.append(col_data)

        # Stack along the last dimension -> (N, 68, 5)
        targets = np.stack(target_arrays, axis=-1)

    # 4. IDs
    ids = df["id"].values

    return {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "targets": targets,
        "ids": ids,
    }


def load_and_cache_data(data_type, load_cached_data=True, debug=False):
    """
    Loads data from cache or parquet, processes it, and caches the result.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, loads a small subset and bypasses main cache.

    Returns:
        dict: Dictionary containing 'inputs', 'pair_indices', 'targets', 'ids'.
    """
    # Determine file paths
    if data_type == "train":
        parquet_path = Config.TRAIN_PARQUET
        cache_path = Config.TRAIN_CACHE
    elif data_type == "val":
        parquet_path = Config.VAL_PARQUET
        cache_path = Config.VAL_CACHE
    elif data_type == "test":
        parquet_path = Config.TEST_PARQUET
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    # Ensure working directory exists (redundant with Config but safe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Debug mode: Bypass cache loading to ensure we get the subset
    if debug:
        print(f"[{data_type}] Debug mode: Loading raw parquet subset...")
        df = pd.read_parquet(parquet_path)
        df = df.head(50)  # Small subset for debugging
        data_dict = process_data(df, data_type)
        return data_dict

    # Normal mode: Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"[{data_type}] Loading cached data from {cache_path}...")
            # Allow pickle is required for loading dictionary/object arrays
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return data_dict
        except Exception as e:
            print(f"[{data_type}] Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"[{data_type}] Processing raw data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    data_dict = process_data(df, data_type)

    # Save to cache
    print(f"[{data_type}] Saving processed data to {cache_path}...")
    np.save(cache_path, data_dict)

    return data_dict
