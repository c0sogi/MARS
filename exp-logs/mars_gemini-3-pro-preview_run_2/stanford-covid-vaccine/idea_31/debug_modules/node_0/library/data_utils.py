import os
import ast
import numpy as np
import pandas as pd
from library.config import Config

# =========================================================================
# Vocabulary Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_idx(structure):
    """
    Parses a dot-bracket structure string to find the index of the paired base
    for each position.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (L,) where arr[i] is the index of the partner
                    of base i. Returns -1 if the base is unpaired.
    """
    length = len(structure)
    partner = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner[i] = j
                partner[j] = i
    return partner


def one_hot_encode(seq, map_dict, length):
    """
    Converts a sequence string into a One-Hot Encoded numpy array.

    Args:
        seq (str): Input sequence.
        map_dict (dict): Dictionary mapping characters to indices.
        length (int): Desired length of the output (truncates or handles short seqs).

    Returns:
        np.ndarray: Array of shape (length, len(map_dict)).
    """
    vocab_size = len(map_dict)
    arr = np.zeros((length, vocab_size), dtype=np.float32)

    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in map_dict:
            arr[i, map_dict[char]] = 1.0

    return arr


def process_data(mode, load_cached_data=True):
    """
    Loads raw metadata, processes features (including Partner Identity),
    parses targets, and caches the result as .npz files.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from existing .npz cache.

    Returns:
        dict: Dictionary containing:
            - 'inputs': np.ndarray (N, SeqLen, Channels)
            - 'targets': np.ndarray (N, SeqLen, NumTargets)
            - 'ids': np.ndarray (N,)
            - 'partner_indices': np.ndarray (N, SeqLen)
    """
    # 1. Determine File Paths
    if mode == "train":
        csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        csv_path = os.path.join(Config.METADATA_DIR, "val.csv")
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        csv_path = os.path.join(Config.METADATA_DIR, "test.csv")
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": data["inputs"],
                "targets": data["targets"],
                "ids": data["ids"],
                "partner_indices": data["partner_indices"],
            }
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing from scratch...")

    # 3. Process from Scratch
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    all_inputs = []
    all_targets = []
    all_ids = []
    all_partner_indices = []

    seq_len = Config.SEQ_LEN

    for _, row in df.iterrows():
        # --- Feature Engineering ---

        # A. Basic One-Hot Encodings
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        # B. Partner Indices
        partner_idx = get_partner_idx(row["structure"])

        # C. Partner Identity (Explicit Injection)
        # Initialize with zeros
        partner_identity = np.zeros((seq_len, len(SEQ_MAP)), dtype=np.float32)

        # Identify paired positions
        paired_mask = partner_idx != -1
        valid_partners = partner_idx[paired_mask]

        # Assign the sequence one-hot of the partner to the current position
        # If i is paired with j, partner_identity[i] = seq_oh[j]
        partner_identity[paired_mask] = seq_oh[valid_partners]

        # D. Concatenate Static Features
        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18 channels
        # Note: Recycling channels are dynamic and added during training/inference
        sample_input = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_identity], axis=1
        )

        all_inputs.append(sample_input)
        all_ids.append(row["id"])
        all_partner_indices.append(partner_idx)

        # --- Target Processing ---
        if mode in ["train", "val"]:
            sample_targets = []
            for col in Config.TARGET_COLS:
                # Parse string representation of list
                try:
                    val_list = ast.literal_eval(row[col])
                except (ValueError, SyntaxError):
                    val_list = []

                # Pad to full sequence length (107)
                # Raw targets are typically length 68
                padded_col = np.zeros(seq_len, dtype=np.float32)
                current_len = len(val_list)
                if current_len > 0:
                    # Copy available data
                    # Ensure we don't exceed seq_len if data is weird
                    copy_len = min(current_len, seq_len)
                    padded_col[:copy_len] = val_list[:copy_len]

                sample_targets.append(padded_col)

            # Stack to get (SeqLen, 5)
            all_targets.append(np.stack(sample_targets, axis=1))

    # 4. Convert to Numpy Arrays
    inputs_arr = np.array(all_inputs, dtype=np.float32)
    partner_indices_arr = np.array(all_partner_indices, dtype=np.int32)
    ids_arr = np.array(all_ids)

    if mode in ["train", "val"]:
        targets_arr = np.array(all_targets, dtype=np.float32)
    else:
        # For test set, create dummy targets of correct shape
        targets_arr = np.zeros(
            (len(df), seq_len, len(Config.TARGET_COLS)), dtype=np.float32
        )

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print(f"Saving {mode} data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs_arr,
        targets=targets_arr,
        ids=ids_arr,
        partner_indices=partner_indices_arr,
    )

    return {
        "inputs": inputs_arr,
        "targets": targets_arr,
        "ids": ids_arr,
        "partner_indices": partner_indices_arr,
    }
