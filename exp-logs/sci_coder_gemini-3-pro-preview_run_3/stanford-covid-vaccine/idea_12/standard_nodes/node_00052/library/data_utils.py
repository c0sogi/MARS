import os
import hashlib
import numpy as np
import pandas as pd
from library.config import Config


def parse_structure_pairs(structure):
    """
    Parses a dot-bracket structure string into an index mapping array.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array where index i maps to paired index j, or i if unpaired.
    """
    n = len(structure)
    pairs = np.arange(n)  # Initialize with self-mapping (unpaired)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
            else:
                # Unbalanced closing bracket handling (though rare in clean data)
                pass

    return pairs


def one_hot_encode(sequence, structure, predicted_loop_type):
    """
    One-hot encodes sequence, structure, and loop type.

    Args:
        sequence (str): RNA sequence (AGUC).
        structure (str): Dot-bracket structure.
        predicted_loop_type (str): Loop type string.

    Returns:
        np.ndarray: Shape (seq_len, 14) float32 tensor.
    """
    seq_len = len(sequence)
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    encoding = np.zeros((seq_len, 14), dtype=np.float32)

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 4, ")": 5, ".": 6}
    loop_map = {"S": 7, "M": 8, "I": 9, "B": 10, "H": 11, "E": 12, "X": 13}

    for i in range(seq_len):
        # Sequence
        s_char = sequence[i]
        if s_char in seq_map:
            encoding[i, seq_map[s_char]] = 1.0

        # Structure
        st_char = structure[i]
        if st_char in struct_map:
            encoding[i, struct_map[st_char]] = 1.0

        # Loop Type
        l_char = predicted_loop_type[i]
        if l_char in loop_map:
            encoding[i, loop_map[l_char]] = 1.0

    return encoding


def process_data(df, split):
    """
    Processes the dataframe into numpy arrays.

    Args:
        df (pd.DataFrame): Input dataframe.
        split (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed arrays.
    """
    ids = []
    inputs = []
    pair_indices = []
    targets = []

    # Target columns
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        ids.append(row["id"])

        # Inputs
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. One-hot encoding
        enc = one_hot_encode(seq, struct, loop)
        inputs.append(enc)

        # 2. Pair indices
        pairs = parse_structure_pairs(struct)
        pair_indices.append(pairs)

        # 3. Targets (only for train/val)
        if split in ["train", "val"]:
            # Targets are lists of floats in the parquet file
            # Shape: (seq_scored, 5) -> (68, 5)
            row_targets = []
            for col in target_cols:
                val = row[col]
                # Ensure it's a list/array
                if isinstance(val, (list, np.ndarray)):
                    row_targets.append(val)
                else:
                    # Fallback for safety
                    row_targets.append([0.0] * 68)

            # Stack to (5, 68) then transpose to (68, 5)
            t = np.array(row_targets, dtype=np.float32).T
            targets.append(t)

    # Convert lists to numpy arrays
    inputs = np.array(inputs, dtype=np.float32)  # (N, 107, 14)
    pair_indices = np.array(pair_indices, dtype=np.int64)  # (N, 107)
    ids = np.array(ids)

    result = {"ids": ids, "inputs": inputs, "pair_indices": pair_indices}

    if split in ["train", "val"]:
        targets = np.array(targets, dtype=np.float32)  # (N, 68, 5)
        result["targets"] = targets

    return result


def load_and_cache_data(config, split, load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches the result.

    Args:
        config (Config): Configuration object.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    # Determine file paths
    if split == "train":
        input_path = config.train_path
        cache_base = config.train_cache_base
    elif split == "val":
        input_path = config.val_path
        cache_base = config.val_cache_base
    elif split == "test":
        input_path = config.test_path
        cache_base = config.test_cache_base
    else:
        raise ValueError(f"Unknown split: {split}")

    # Generate hash for cache versioning
    # Hash includes input path and relevant config params to invalidate cache if params change
    hash_content = (
        f"{input_path}_{config.seq_len}_{config.input_channels}_{config.scored_cols}"
    )
    file_hash = hashlib.md5(hash_content.encode()).hexdigest()
    cache_path = f"{cache_base}_{file_hash}.npz"

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=True is needed for string arrays (ids)
            data = np.load(cache_path, allow_pickle=True)
            # Return as dict
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # Process from scratch
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)

    # Process
    processed_data = process_data(df, split)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **processed_data)

    return processed_data
