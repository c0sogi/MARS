import os
import ast
import numpy as np
import pandas as pd
from library.config import Config

# =========================================================================
# Vocabulary Definitions
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_couples(structure):
    """
    Converts a dot-bracket structure string into a pair index array.
    Returns:
        pairs (np.ndarray): Array of shape (L,) where pairs[i] = j if i pairs with j,
                            and -1 if i is unpaired.
    """
    L = len(structure)
    pairs = np.full(L, -1, dtype=np.int32)
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


def one_hot(seq, mapping):
    """
    One-hot encodes a sequence string based on the provided mapping.
    """
    L = len(seq)
    N_classes = len(mapping)
    encoded = np.zeros((L, N_classes), dtype=np.float32)

    for i, char in enumerate(seq):
        if char in mapping:
            encoded[i, mapping[char]] = 1.0

    return encoded


def get_partner_identity(seq_onehot, pair_indices):
    """
    Generates the partner identity features.
    If base i is paired with j, output[i] == seq_onehot[j].
    If unpaired, output[i] is all zeros.
    """
    L, D = seq_onehot.shape
    partner_id = np.zeros((L, D), dtype=np.float32)

    # Mask for paired bases (where index != -1)
    mask = pair_indices != -1

    # Get indices of partners
    partners = pair_indices[mask]

    # Assign features
    partner_id[mask] = seq_onehot[partners]

    return partner_id


def process_data(df, mode="train"):
    """
    Processes a dataframe into dense numpy arrays for training/inference.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Feature dimensions
    dim_seq = len(SEQ_MAP)
    dim_struct = len(STRUCT_MAP)
    dim_loop = len(LOOP_MAP)
    dim_partner = dim_seq  # Partner identity has same dim as sequence
    total_input_dim = dim_seq + dim_struct + dim_loop + dim_partner

    # Initialize arrays
    # Inputs: (N, L, Channels)
    inputs = np.zeros((num_samples, seq_len, total_input_dim), dtype=np.float32)
    # Pair Indices: (N, L)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    # IDs: (N,)
    ids = df["id"].values

    # Targets: (N, L, 5)
    # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # Loop through samples
    for idx, row in df.iterrows():
        # 1. Parse Sequences
        seq_str = row["sequence"]
        struct_str = row["structure"]
        loop_str = row["predicted_loop_type"]

        # 2. Generate Basic Features
        oh_seq = one_hot(seq_str, SEQ_MAP)
        oh_struct = one_hot(struct_str, STRUCT_MAP)
        oh_loop = one_hot(loop_str, LOOP_MAP)

        # 3. Generate Structural/Partner Features
        pairs = get_couples(struct_str)
        oh_partner = get_partner_identity(oh_seq, pairs)

        # 4. Concatenate Inputs
        # Order: Sequence, Structure, Loop, Partner
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=-1)
        inputs[idx] = sample_input
        pair_indices[idx] = pairs

        # 5. Process Targets (if available)
        if mode in ["train", "val"]:
            for t_i, col in enumerate(target_cols):
                # Parse stringified list
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                except:
                    val_list = []

                # Assign to the first 68 positions
                # The remaining positions (68-107) remain 0.0 (Boundary Anchoring)
                length = min(len(val_list), seq_len)
                targets[idx, :length, t_i] = val_list[:length]

        # For test mode, targets remain 0.0

    return {
        "inputs": inputs,
        "targets": targets,
        "pair_indices": pair_indices,
        "ids": ids,
    }


def get_data(mode="train", load_cached_data=True):
    """
    Main data loading function with caching mechanism.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npz cache.

    Returns:
        dict: Dictionary containing 'inputs', 'targets', 'pair_indices', 'ids'.
    """
    # Determine paths based on mode
    if mode == "train":
        csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        cache_path = Config.TRAIN_CACHE_PATH
    elif mode == "val":
        csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        cache_path = Config.VAL_CACHE_PATH
    elif mode == "test":
        csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": data["inputs"],
                "targets": data["targets"],
                "pair_indices": data["pair_indices"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    processed_data = process_data(df, mode=mode)

    # 3. Save to Cache
    print(f"Saving {mode} data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=processed_data["inputs"],
        targets=processed_data["targets"],
        pair_indices=processed_data["pair_indices"],
        ids=processed_data["ids"],
    )

    return processed_data
