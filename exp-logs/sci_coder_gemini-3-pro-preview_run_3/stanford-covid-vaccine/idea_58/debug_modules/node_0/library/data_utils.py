import os
import numpy as np
import pandas as pd
from library.config import Config


def get_structure_adj(structure_str):
    """
    Parses a dot-bracket structure string to generate indices of paired bases.

    Args:
        structure_str (str): Dot-bracket string (e.g., '...((...))...').

    Returns:
        indices (np.ndarray): Array of shape (L,) where indices[i] = j if i is paired with j.
                              If unpaired, indices[i] = 0 (safe index for gather, masked later).
        mask (np.ndarray): Array of shape (L,) where mask[i] = 1.0 if paired, 0.0 otherwise.
    """
    length = len(structure_str)
    indices = np.zeros(length, dtype=np.int32)
    mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
            else:
                # Handle potentially unbalanced brackets gracefully
                pass

    return indices, mask


def encode_sequence_features(seq_str, struct_str, loop_str):
    """
    One-hot encodes sequence, structure, and loop type into a unified tensor.

    Channels:
    0-3: A, G, C, U
    4-6: ., (, )
    7-13: S, M, I, B, H, E, X
    Total: 14 channels

    Args:
        seq_str (str): Sequence string.
        struct_str (str): Structure string.
        loop_str (str): Predicted loop type string.

    Returns:
        np.ndarray: Float32 array of shape (L, 14).
    """
    length = len(seq_str)
    encoding = np.zeros((length, 14), dtype=np.float32)

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 4, "(": 5, ")": 6}
    loop_map = {"S": 7, "M": 8, "I": 9, "B": 10, "H": 11, "E": 12, "X": 13}

    for i in range(length):
        # Sequence
        s_char = seq_str[i]
        if s_char in seq_map:
            encoding[i, seq_map[s_char]] = 1.0

        # Structure
        st_char = struct_str[i]
        if st_char in struct_map:
            encoding[i, struct_map[st_char]] = 1.0

        # Loop Type
        l_char = loop_str[i]
        if l_char in loop_map:
            encoding[i, loop_map[l_char]] = 1.0

    return encoding


def process_data(df, is_test=False):
    """
    Converts a pandas DataFrame into a dictionary of numpy arrays.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): If True, targets are not extracted.

    Returns:
        dict: Dictionary containing 'features', 'bpp_indices', 'bpp_mask', 'ids', and optionally 'targets'.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Pre-allocate arrays
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    bpp_mask = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Extract Targets if not test
    targets = None
    if not is_test:
        target_cols = Config.TARGET_COLS
        targets_list = []
        for col in target_cols:
            # Each element in the column is a list/array. vstack converts to (N, 68)
            col_data = np.vstack(df[col].values)
            targets_list.append(col_data)

        # Stack along the last dimension -> (N, 68, 5)
        targets = np.stack(targets_list, axis=2).astype(np.float32)

    # Extract Features
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    for i in range(num_samples):
        features[i] = encode_sequence_features(sequences[i], structures[i], loops[i])
        idx, msk = get_structure_adj(structures[i])
        bpp_indices[i] = idx
        bpp_mask[i] = msk

    data_dict = {
        "features": features,
        "bpp_indices": bpp_indices,
        "bpp_mask": bpp_mask,
        "ids": ids,
    }

    if targets is not None:
        data_dict["targets"] = targets

    return data_dict


def load_dataset(mode="train", load_cached_data=True, max_samples=None):
    """
    Loads the dataset for the specified mode, handling caching and slicing.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached .npz files.
        max_samples (int, optional): Number of samples to return (for debugging).

    Returns:
        dict: Dictionary containing the dataset arrays.
    """
    # Determine paths based on mode
    if mode == "train":
        parquet_path = Config.TRAIN_PATH
        cache_path = Config.TRAIN_CACHE
        is_test = False
    elif mode == "val":
        parquet_path = Config.VAL_PATH
        cache_path = Config.VAL_CACHE
        is_test = False
    elif mode == "test":
        parquet_path = Config.TEST_PATH
        cache_path = Config.TEST_CACHE
        is_test = True
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    data = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load npz file
            loaded = np.load(cache_path)
            # Convert NpzFile to dict to force loading into memory
            data = {key: loaded[key] for key in loaded.files}
        except Exception as e:
            print(f"Warning: Failed to load cache from {cache_path}. Error: {e}")
            data = None

    # 2. Process from scratch if needed
    if data is None:
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Metadata file not found at {parquet_path}")

        df = pd.read_parquet(parquet_path)
        data = process_data(df, is_test=is_test)

        # Save to cache
        np.savez(cache_path, **data)

    # 3. Handle slicing (max_samples)
    if max_samples is not None and max_samples < len(data["ids"]):
        sliced_data = {}
        for key, arr in data.items():
            sliced_data[key] = arr[:max_samples]
        return sliced_data

    return data
