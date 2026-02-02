import os
import ast
import numpy as np
import pandas as pd
from library import config

# Dictionaries for One-Hot Encoding
SEQ_MAP = {"A": [1, 0, 0, 0], "G": [0, 1, 0, 0], "C": [0, 0, 1, 0], "U": [0, 0, 0, 1]}

STRUCT_MAP = {"(": [1, 0, 0], ")": [0, 1, 0], ".": [0, 0, 1]}

LOOP_MAP = {
    "S": [1, 0, 0, 0, 0, 0, 0],
    "M": [0, 1, 0, 0, 0, 0, 0],
    "I": [0, 0, 1, 0, 0, 0, 0],
    "B": [0, 0, 0, 1, 0, 0, 0],
    "H": [0, 0, 0, 0, 1, 0, 0],
    "E": [0, 0, 0, 0, 0, 1, 0],
    "X": [0, 0, 0, 0, 0, 0, 1],
}


def parse_dot_bracket(structure):
    """
    Parses a dot-bracket structure string to find pairing partners.
    Returns:
        partner_indices (np.ndarray): Array of shape (L,) where arr[i] is the index
                                      of the base paired with i, or -1 if unpaired.
    """
    n = len(structure)
    partner_indices = np.full(n, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i

    return partner_indices


def get_partner_identity(sequence, partner_indices):
    """
    Creates a one-hot encoding of the partner base.
    If a base is unpaired (partner_index == -1), returns a zero vector.
    """
    seq_len = len(sequence)
    # 4 channels for A, G, C, U
    partner_identity = np.zeros((seq_len, 4), dtype=np.float32)

    for i, partner_idx in enumerate(partner_indices):
        if partner_idx != -1:
            partner_base = sequence[partner_idx]
            if partner_base in SEQ_MAP:
                partner_identity[i] = SEQ_MAP[partner_base]

    return partner_identity


def str_to_onehot(string_seq, mapping, length):
    """
    Converts a string sequence to a one-hot encoded numpy array.
    """
    encoded = []
    for char in string_seq:
        encoded.append(mapping.get(char, [0] * len(list(mapping.values())[0])))

    # Pad or truncate if necessary (though data should be fixed length)
    curr_len = len(encoded)
    if curr_len < length:
        padding = [[0] * len(list(mapping.values())[0])] * (length - curr_len)
        encoded.extend(padding)
    elif curr_len > length:
        encoded = encoded[:length]

    return np.array(encoded, dtype=np.float32)


def process_csv(csv_path, is_test=False, debug=False):
    """
    Reads CSV, processes features and targets, and returns numpy arrays.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if debug:
        df = df.head(config.DEBUG_SUBSET_SIZE).copy()
        print(f"DEBUG MODE: Processing subset of {len(df)} samples from {csv_path}")

    # Initialize lists to store processed data
    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = []

    # Columns to parse for targets
    target_cols = config.ALL_TARGET_COLS

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        sample_id = row["id"]

        # 1. Feature Engineering
        # Base One-Hot
        seq_oh = str_to_onehot(seq, SEQ_MAP, config.SEQ_LEN)
        # Structure One-Hot
        struct_oh = str_to_onehot(struct, STRUCT_MAP, config.SEQ_LEN)
        # Loop Type One-Hot
        loop_oh = str_to_onehot(loop, LOOP_MAP, config.SEQ_LEN)

        # Partner Map
        partner_idx = parse_dot_bracket(struct)

        # Partner Identity One-Hot
        partner_ident_oh = get_partner_identity(seq, partner_idx)

        # Concatenate all features: (L, 4+3+7+4) = (L, 18)
        # Channels: [Seq(4), Struct(3), Loop(7), PartnerIdentity(4)]
        sample_input = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_ident_oh], axis=1
        )

        all_inputs.append(sample_input)
        all_partner_indices.append(partner_idx)
        all_ids.append(sample_id)

        # 2. Target Processing (if not test)
        if not is_test:
            sample_targets = []
            for col in target_cols:
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(row[col])
                except (ValueError, SyntaxError):
                    # Fallback for malformed data, though unlikely in clean metadata
                    val_list = [0.0] * config.SCORED_LEN

                # Pad to SEQ_LEN (107)
                # Targets are provided for first 68 bases
                full_target = np.zeros(config.SEQ_LEN, dtype=np.float32)
                length = min(len(val_list), config.SEQ_LEN)
                full_target[:length] = val_list[:length]
                sample_targets.append(full_target)

            # Shape: (5, 107) -> Transpose to (107, 5)
            all_targets.append(np.stack(sample_targets, axis=1))

    # Convert to numpy arrays
    X = np.array(all_inputs, dtype=np.float32)  # (N, 107, 18)
    P = np.array(all_partner_indices, dtype=np.int32)  # (N, 107)
    Ids = np.array(all_ids)  # (N,)

    if not is_test:
        Y = np.array(all_targets, dtype=np.float32)  # (N, 107, 5)
    else:
        # Create dummy targets for test set to maintain consistent API
        Y = np.zeros((len(df), config.SEQ_LEN, len(target_cols)), dtype=np.float32)

    return {"inputs": X, "partner_indices": P, "targets": Y, "ids": Ids}


def get_dataset(split="train", load_cached_data=True, debug=False):
    """
    Main interface to get the dataset. Handles caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to use a small subset for debugging.

    Returns:
        dict: Dictionary containing 'inputs', 'partner_indices', 'targets', 'ids'.
    """
    # Determine paths based on split
    if split == "train":
        csv_path = config.TRAIN_CSV
        cache_path = config.CACHE_TRAIN
        is_test = False
    elif split == "val":
        csv_path = config.VAL_CSV
        cache_path = config.CACHE_VAL
        is_test = False
    elif split == "test":
        csv_path = config.TEST_CSV
        cache_path = config.CACHE_TEST
        is_test = True
    else:
        raise ValueError(f"Invalid split: {split}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_path) and not debug:
        try:
            print(f"Loading cached {split} data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": data["inputs"],
                "partner_indices": data["partner_indices"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # Process data from scratch
    print(f"Processing {split} data from {csv_path}...")
    data_dict = process_csv(csv_path, is_test=is_test, debug=debug)

    # Save to cache (only if not in debug mode, to avoid overwriting full cache with subset)
    if not debug:
        print(f"Saving {split} data to cache {cache_path}...")
        np.savez_compressed(
            cache_path,
            inputs=data_dict["inputs"],
            partner_indices=data_dict["partner_indices"],
            targets=data_dict["targets"],
            ids=data_dict["ids"],
        )

    return data_dict
