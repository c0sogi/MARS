import os
import numpy as np
import pandas as pd
import torch
from library.config import Config

# =============================================================================
# Mappings
# =============================================================================
# Sequence: 4 bases
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}

# Structure: 3 types
# Note: In dot-bracket, '(' and ')' denote pairs, '.' denotes unpaired.
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}

# Predicted Loop Type: 7 types
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_to_indices(structure_str):
    """
    Parses a dot-bracket structure string into an adjacency index array.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (L,) where arr[i] is the index of the base
                    paired with i. If i is unpaired, arr[i] = -1.
    """
    n = len(structure_str)
    indices = np.full(n, -1, dtype=np.int32)
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


def one_hot_encode(sequence, structure, loop_type):
    """
    Encodes sequence, structure, and loop type into a single feature tensor.

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket structure.
        loop_type (str): Predicted loop type string.

    Returns:
        np.ndarray: Float32 array of shape (Seq_Len, 14).
                    Channels 0-3: Sequence (A, G, C, U)
                    Channels 4-6: Structure ((, ), .)
                    Channels 7-13: Loop Type (S, M, I, B, H, E, X)
    """
    seq_len = len(sequence)
    # Total channels = 4 + 3 + 7 = 14
    encoding = np.zeros((seq_len, 14), dtype=np.float32)

    for i in range(seq_len):
        # Sequence
        s_char = sequence[i]
        if s_char in SEQ_MAP:
            encoding[i, SEQ_MAP[s_char]] = 1.0

        # Structure
        st_char = structure[i]
        if st_char in STRUCT_MAP:
            encoding[i, 4 + STRUCT_MAP[st_char]] = 1.0

        # Loop Type
        l_char = loop_type[i]
        if l_char in LOOP_MAP:
            encoding[i, 7 + LOOP_MAP[l_char]] = 1.0

    return encoding


def process_dataframe(df, is_test=False):
    """
    Internal function to process a dataframe into numpy arrays.
    """
    ids = df[Config.ID_COL].values

    # Pre-allocate lists
    features_list = []
    adjacency_list = []

    # Iterate and process features
    for idx, row in df.iterrows():
        seq = row[Config.SEQUENCE_COL]
        struct = row[Config.STRUCTURE_COL]
        loop = row[Config.LOOP_TYPE_COL]

        # 1. One-Hot Encoding
        feat = one_hot_encode(seq, struct, loop)
        features_list.append(feat)

        # 2. Adjacency Parsing
        adj = parse_structure_to_indices(struct)
        adjacency_list.append(adj)

    features = np.array(features_list, dtype=np.float32)  # (N, L, 14)
    adjacency = np.array(adjacency_list, dtype=np.int32)  # (N, L)

    targets = None
    if not is_test:
        # Extract targets. The parquet file stores them as lists/arrays in cells.
        # We need to stack them into a (N, L, 5) tensor.
        # Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

        target_arrays = []
        for col in Config.TARGET_COLS:
            # df[col] is a Series of lists/arrays.
            # np.vstack or np.array(list(...)) converts to 2D array (N, 68)
            col_data = np.array(df[col].tolist(), dtype=np.float32)
            target_arrays.append(col_data)

        # Stack along the last dimension -> (N, 68, 5)
        targets = np.stack(target_arrays, axis=-1)

    return {
        "ids": ids,
        "features": features,
        "adjacency": adjacency,
        "targets": targets,
    }


def load_dataset(split="train", load_cached_data=True):
    """
    Loads the dataset for a specific split (train, val, test).
    Handles caching to .npz files to speed up subsequent loads.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: Dictionary containing:
            - 'ids': np.ndarray of strings
            - 'features': np.ndarray (N, 107, 14)
            - 'adjacency': np.ndarray (N, 107)
            - 'targets': np.ndarray (N, 68, 5) or None if test
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "ids": data["ids"],
                "features": data["features"],
                "adjacency": data["adjacency"],
                "targets": (
                    data["targets"] if "targets" in data and split != "test" else None
                ),
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load from Parquet Metadata
    print(f"Processing {split} data from metadata...")

    if split == "train":
        file_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        file_path = Config.VAL_DATA_PATH
    elif split == "test":
        file_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metadata file not found: {file_path}")

    df = pd.read_parquet(file_path)

    # 3. Process Data
    processed_data = process_dataframe(df, is_test=(split == "test"))

    # 4. Save to Cache
    print(f"Saving {split} data to cache: {cache_path}")
    save_dict = {
        "ids": processed_data["ids"],
        "features": processed_data["features"],
        "adjacency": processed_data["adjacency"],
    }

    if processed_data["targets"] is not None:
        save_dict["targets"] = processed_data["targets"]

    np.savez_compressed(cache_path, **save_dict)

    return processed_data
