import os
import numpy as np
import pandas as pd
from library.config import Config

# Global Mappings for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.array: Array of length len(structure).
                  index[i] = j if base i is paired with base j.
                  index[i] = -1 if base i is unpaired.
    """
    length = len(structure)
    pairs = np.full(length, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pairs[start] = i
                pairs[i] = start

    return pairs


def encode_sequence_features(sequence, structure, loop_type):
    """
    Generates the base feature vector for a single sequence (without spatial augmentation).

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket structure.
        loop_type (str): Loop type string.

    Returns:
        np.array: Shape (seq_len, 14) - One-hot encoded features.
    """
    length = len(sequence)

    # Initialize feature arrays
    # Dimensions: Seq(4) + Struct(3) + Loop(7) = 14
    f_seq = np.zeros((length, 4), dtype=np.float32)
    f_struct = np.zeros((length, 3), dtype=np.float32)
    f_loop = np.zeros((length, 7), dtype=np.float32)

    # Fill One-Hot Encodings
    for i in range(length):
        # Sequence
        char_s = sequence[i]
        if char_s in SEQ_MAP:
            f_seq[i, SEQ_MAP[char_s]] = 1.0

        # Structure
        char_st = structure[i]
        if char_st in STRUCT_MAP:
            f_struct[i, STRUCT_MAP[char_st]] = 1.0

        # Loop Type
        char_l = loop_type[i]
        if char_l in LOOP_MAP:
            f_loop[i, LOOP_MAP[char_l]] = 1.0

    # Concatenate
    return np.concatenate([f_seq, f_struct, f_loop], axis=1)


def build_augmented_features(df):
    """
    Constructs the feature tensor for the entire dataframe.
    Simplified to remove spatial augmentation (Cite solution_lesson_node_00033).

    Args:
        df (pd.DataFrame): Dataframe containing 'sequence', 'structure', 'predicted_loop_type'.

    Returns:
        np.array: Shape (N_samples, seq_len, 14).
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH
    input_dim = 14  # Base dimension

    # Final container
    X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)

    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values

    for idx in range(n_samples):
        seq = sequences[idx]
        struct = structures[idx]
        loop = loop_types[idx]

        # Get Base Features (L, 14)
        base_feats = encode_sequence_features(seq, struct, loop)

        # Assign to tensor
        X[idx] = base_feats

    return X


def process_targets(df):
    """
    Extracts targets and calculates weights based on errors.

    Args:
        df (pd.DataFrame): Dataframe with target and error columns.

    Returns:
        y (np.array): Targets (N, seq_scored, 5)
        w (np.array): Weights (N, seq_scored, 5)
    """
    # Extract targets
    # Targets are stored as lists in the dataframe columns
    # We stack them to create (N, 68) arrays for each target type
    target_arrays = []
    for col in Config.TARGET_COLS:
        # Convert column of lists to 2D numpy array
        # Some rows might be strings if loaded incorrectly, but parquet should handle lists.
        # We assume lists of floats.
        data = np.vstack(df[col].values)
        target_arrays.append(data)

    # Stack along last axis -> (N, 68, 5)
    y = np.stack(target_arrays, axis=2).astype(np.float32)

    # Extract errors and compute weights
    # Weight = 1 / (error^2 + epsilon)
    weight_arrays = []
    for col in Config.ERROR_COLS:
        data = np.vstack(df[col].values)
        # Compute weight
        w_col = 1.0 / (data**2 + Config.LOSS_EPSILON)
        weight_arrays.append(w_col)

    w = np.stack(weight_arrays, axis=2).astype(np.float32)

    return y, w


def load_dataset(split, load_cached_data=True):
    """
    Main entry point to load and process data for a specific split.
    Handles caching mechanism.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing:
            'X': Feature tensor (N, 107, 28)
            'y': Target tensor (N, 68, 5) (Train/Val only)
            'w': Weight tensor (N, 68, 5) (Train/Val only)
            'ids': List of sample IDs
    """
    # Determine paths
    cache_file = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache: {cache_file}")
        try:
            data = np.load(cache_file, allow_pickle=True)
            result = {"X": data["X"], "ids": data["ids"]}
            if split != "test":
                result["y"] = data["y"]
                result["w"] = data["w"]
            return result
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {split} data from metadata...")

    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load metadata
    df = pd.read_parquet(meta_path)

    # Debugging subset
    if Config.DEBUG_SUBSET_SIZE is not None and split == "train":
        print(f"DEBUG: Using subset of {Config.DEBUG_SUBSET_SIZE} samples.")
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

    # Build Features
    X = build_augmented_features(df)
    ids = df["id"].values

    result = {"X": X, "ids": ids}

    # Build Targets (if not test)
    if split != "test":
        y, w = process_targets(df)
        result["y"] = y
        result["w"] = w

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    print(f"Saving {split} data to cache: {cache_file}")
    if split != "test":
        np.savez_compressed(
            cache_file, X=result["X"], y=result["y"], w=result["w"], ids=result["ids"]
        )
    else:
        np.savez_compressed(cache_file, X=result["X"], ids=result["ids"])

    return result
