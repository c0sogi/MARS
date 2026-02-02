import os
import ast
import numpy as np
import pandas as pd
from library.config import Config

# Mapping dictionaries for one-hot encoding
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_array(x):
    """
    Parses a string representation of a list into a numpy array.
    Handles potential malformed strings gracefully.
    """
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string to find pairing partners.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of length L. index[i] = j means base i is paired with base j.
                    index[i] = -1 means base i is unpaired.
    """
    length = len(structure)
    indices = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i

    return indices


def get_one_hot_features(sequence, structure, loop_type, partner_indices):
    """
    Generates the input feature tensor for a single sample.

    Features (18 channels):
    - Sequence (4): A, G, U, C
    - Structure (3): (, ), .
    - Loop Type (7): S, M, I, B, H, E, X
    - Partner Identity (4): A, G, U, C (of the paired base)

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket structure.
        loop_type (str): Predicted loop type string.
        partner_indices (np.ndarray): Array mapping index i to partner j.

    Returns:
        np.ndarray: Feature matrix of shape (Seq_Len, 18).
    """
    length = len(sequence)
    features = np.zeros((length, Config.INPUT_CHANNELS), dtype=np.float32)

    for i in range(length):
        # 1. Sequence One-Hot (0-3)
        char_seq = sequence[i]
        if char_seq in SEQ_MAP:
            features[i, SEQ_MAP[char_seq]] = 1.0

        # 2. Structure One-Hot (4-6)
        char_struct = structure[i]
        if char_struct in STRUCT_MAP:
            features[i, 4 + STRUCT_MAP[char_struct]] = 1.0

        # 3. Loop Type One-Hot (7-13)
        char_loop = loop_type[i]
        if char_loop in LOOP_MAP:
            features[i, 7 + LOOP_MAP[char_loop]] = 1.0

        # 4. Partner Identity (14-17)
        # If paired, encode the sequence identity of the partner.
        # If unpaired (partner_idx == -1), this remains 0.
        partner_idx = partner_indices[i]
        if partner_idx != -1:
            partner_char = sequence[partner_idx]
            if partner_char in SEQ_MAP:
                features[i, 14 + SEQ_MAP[partner_char]] = 1.0

    return features


def process_data(csv_path, cache_path, mode="train", load_cached_data=True):
    """
    Main data processing function. Loads CSV, generates features/targets,
    and handles caching logic.

    Args:
        csv_path (str): Path to the metadata CSV file.
        cache_path (str): Path to the .npz cache file.
        mode (str): 'train' (includes targets) or 'test' (no targets).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'inputs', 'partner_indices', 'ids', and optionally 'targets'.
    """
    # 1. Load Metadata
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Subset for debugging if configured
    if Config.SUBSET_SIZE is not None:
        print(f"Subsetting data to first {Config.SUBSET_SIZE} samples.")
        df = df.iloc[: Config.SUBSET_SIZE].reset_index(drop=True)

    # 2. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Checking cache at {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Cite debug_lesson_1: Validate Cache Integrity
            if len(data["ids"]) == len(df):
                print(f"Loading cached data (Size: {len(df)})...")
                return {key: data[key] for key in data.files}
            else:
                print(
                    f"Cache size mismatch (Found: {len(data['ids'])}, Expected: {len(df)}). Reprocessing..."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 3. Initialize Lists
    all_inputs = []
    all_partner_indices = []
    all_ids = df["id"].values

    # Target arrays (only for train/val)
    all_targets = []

    # 4. Iterate and Process
    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Generate Partner Indices
        p_indices = get_structure_indices(struct)

        # Generate Input Features
        feats = get_one_hot_features(seq, struct, loop, p_indices)

        all_inputs.append(feats)
        all_partner_indices.append(p_indices)

        # Process Targets if training
        if mode == "train":
            # Collect all 5 targets
            # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Note: The CSV stores them as stringified lists
            t_list = []
            for t_col in Config.ALL_TARGETS:
                val_array = parse_array(row[t_col])
                # Ensure length matches PRED_LEN (68)
                if len(val_array) < Config.PRED_LEN:
                    # Pad with nans or zeros if missing (should not happen in clean data)
                    padded = np.full(Config.PRED_LEN, np.nan, dtype=np.float32)
                    padded[: len(val_array)] = val_array
                    t_list.append(padded)
                else:
                    t_list.append(val_array[: Config.PRED_LEN])

            # Stack targets for this sample: Shape (68, 5)
            sample_targets = np.stack(t_list, axis=1)
            all_targets.append(sample_targets)

    # 5. Convert to Numpy Arrays
    inputs_np = np.array(all_inputs, dtype=np.float32)  # (N, 107, 18)
    partner_indices_np = np.array(all_partner_indices, dtype=np.int32)  # (N, 107)

    result = {
        "inputs": inputs_np,
        "partner_indices": partner_indices_np,
        "ids": all_ids,
    }

    if mode == "train":
        targets_np = np.array(all_targets, dtype=np.float32)  # (N, 68, 5)
        result["targets"] = targets_np

    # 6. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **result)

    return result


def load_train_data(load_cached_data=True):
    return process_data(
        Config.TRAIN_METADATA,
        Config.TRAIN_CACHE,
        mode="train",
        load_cached_data=load_cached_data,
    )


def load_val_data(load_cached_data=True):
    return process_data(
        Config.VAL_METADATA,
        Config.VAL_CACHE,
        mode="train",
        load_cached_data=load_cached_data,
    )


def load_test_data(load_cached_data=True):
    return process_data(
        Config.TEST_METADATA,
        Config.TEST_CACHE,
        mode="test",
        load_cached_data=load_cached_data,
    )
