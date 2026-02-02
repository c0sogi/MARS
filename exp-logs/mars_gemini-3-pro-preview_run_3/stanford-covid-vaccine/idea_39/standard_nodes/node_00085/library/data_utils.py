import os
import hashlib
import numpy as np
import pandas as pd
from library.config import Config


def get_couples(structure):
    """
    Parses a dot-bracket structure string to identify base pairs for the
    Decoupled Structural Interaction Module.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        tuple:
            - pair_indices (np.ndarray): Array of shape (L,) where index i contains the index j
              it is paired with. If unpaired, defaults to i (self-loop) to ensure valid gather indices.
            - pair_mask (np.ndarray): Array of shape (L,) containing 1.0 if paired, 0.0 otherwise.
              Used for Strict Output Masking.
    """
    seq_len = len(structure)
    # Default to self-index to prevent out-of-bounds errors during gather
    pair_indices = np.arange(seq_len, dtype=np.int64)
    pair_mask = np.zeros(seq_len, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Record bidirectional relationship
                pair_indices[i] = j
                pair_indices[j] = i
                # Set mask to 1 for valid pairs
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0

    return pair_indices, pair_mask


def one_hot_encode(sequence, structure, loop_type, config):
    """
    Encodes sequence, structure, and loop type into a one-hot tensor.

    Channels:
    - Sequence (4): A, G, U, C
    - Structure (3): ., (, )
    - Loop Type (7): S, M, I, B, H, E, X
    Total: 14 channels

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket structure.
        loop_type (str): Predicted loop type.
        config (Config): Configuration object with mappings.

    Returns:
        np.ndarray: Feature tensor of shape (seq_len, input_dim).
    """
    seq_len = config.seq_len
    # Initialize tensor (N, 107, 14)
    features = np.zeros((seq_len, config.input_dim), dtype=np.float32)

    # Mapping dictionaries
    token2int = config.token2int
    struct2int = config.struct2int
    loop2int = config.loop2int

    # Channel offsets
    offset_struct = config.num_tokens
    offset_loop = config.num_tokens + config.num_struct

    # Fill tensor
    for i in range(seq_len):
        # Sequence
        if i < len(sequence):
            char = sequence[i]
            if char in token2int:
                features[i, token2int[char]] = 1.0

        # Structure
        if i < len(structure):
            char = structure[i]
            if char in struct2int:
                features[i, offset_struct + struct2int[char]] = 1.0

        # Loop Type
        if i < len(loop_type):
            char = loop_type[i]
            if char in loop2int:
                features[i, offset_loop + loop2int[char]] = 1.0

    return features


def process_dataframe(df, config, is_test=False):
    """
    Processes a dataframe into numpy arrays for model input.

    Args:
        df (pd.DataFrame): Input dataframe.
        config (Config): Configuration object.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        dict: Dictionary containing 'inputs', 'bpp_indices', 'bpp_mask', 'ids',
              and optionally 'targets'.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, config.input_dim), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    bpp_mask = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = []

    # Targets are only present for train/val
    targets = None
    if not is_test:
        # Targets shape: (N, 68, 5)
        targets = np.zeros(
            (num_samples, config.pred_len, config.num_classes), dtype=np.float32
        )

    # Iterate and process
    for i, (_, row) in enumerate(df.iterrows()):
        # 1. Feature Encoding
        inputs[i] = one_hot_encode(
            row["sequence"], row["structure"], row["predicted_loop_type"], config
        )

        # 2. Adjacency Parsing
        p_idx, p_mask = get_couples(row["structure"])
        bpp_indices[i] = p_idx
        bpp_mask[i] = p_mask

        ids.append(row["id"])

        # 3. Target Extraction (if not test)
        if not is_test:
            for t_idx, col in enumerate(config.target_cols):
                val_list = row[col]
                # Ensure it's a list or array
                if isinstance(val_list, (list, np.ndarray)):
                    length = min(len(val_list), config.pred_len)
                    targets[i, :length, t_idx] = val_list[:length]

    # Convert IDs to numpy array of strings
    ids = np.array(ids, dtype=str)

    result = {
        "inputs": inputs,
        "bpp_indices": bpp_indices,
        "bpp_mask": bpp_mask,
        "ids": ids,
    }

    if not is_test:
        result["targets"] = targets

    return result


def get_config_hash(config):
    """
    Generates a hash based on relevant config parameters to ensure cache validity.
    Changes in dimensions or target columns will invalidate old caches.
    """
    params = (
        f"{config.seq_len}_{config.pred_len}_{config.input_dim}_{config.target_cols}"
    )
    return hashlib.md5(params.encode()).hexdigest()


def load_or_process_data(data_type, config, load_cached_data=True):
    """
    Loads data from cache or processes it from metadata Parquet files.

    Args:
        data_type (str): 'train', 'val', or 'test'.
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed data arrays.
    """
    # Determine source path
    if data_type == "train":
        source_path = config.train_metadata_path
    elif data_type == "val":
        source_path = config.val_metadata_path
    elif data_type == "test":
        source_path = config.test_metadata_path
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # Generate cache filename based on config hash
    config_hash = get_config_hash(config)
    cache_filename = f"{data_type}_data_{config_hash}.npz"
    cache_path = os.path.join(config.cache_dir, cache_filename)

    # Ensure cache directory exists
    os.makedirs(config.cache_dir, exist_ok=True)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {data_type} data from cache: {cache_path}")
        try:
            # Load npz file
            loaded = np.load(cache_path, allow_pickle=True)
            data = {key: loaded[key] for key in loaded.files}

            # Handle max_train_samples for debugging (slicing cached data)
            if data_type == "train" and config.max_train_samples is not None:
                limit = config.max_train_samples
                if len(data["inputs"]) > limit:
                    print(f"Slicing cached data to {limit} samples.")
                    data["inputs"] = data["inputs"][:limit]
                    data["bpp_indices"] = data["bpp_indices"][:limit]
                    data["bpp_mask"] = data["bpp_mask"][:limit]
                    data["ids"] = data["ids"][:limit]
                    if "targets" in data:
                        data["targets"] = data["targets"][:limit]

            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {data_type} data from {source_path}...")

    # Load raw data from Parquet
    df = pd.read_parquet(source_path)

    # Apply limit for debugging BEFORE processing to save time
    if data_type == "train" and config.max_train_samples is not None:
        print(f"Limiting processing to {config.max_train_samples} samples.")
        df = df.iloc[: config.max_train_samples]

    is_test = data_type == "test"
    data = process_dataframe(df, config, is_test=is_test)

    # 3. Save to Cache
    print(f"Saving processed {data_type} data to {cache_path}...")
    np.savez_compressed(cache_path, **data)

    return data
