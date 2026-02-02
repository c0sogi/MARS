import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# ------------------------------------------------------------------------------
# Mappings
# ------------------------------------------------------------------------------
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# Inverse map for partner identity lookup
IDX_TO_BASE_MAP = {v: k for k, v in SEQ_MAP.items()}


def get_partner_indices(structure):
    """
    Parses a dot-bracket structure string to find paired indices.
    Returns a numpy array of shape (seq_len,) where arr[i] is the index of the
    base paired with i, or -1 if unpaired.
    """
    partner = np.full(len(structure), -1, dtype=np.int32)
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


def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    Returns shape (length, num_classes).
    """
    num_classes = len(mapping)
    encoding = np.zeros((length, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def get_partner_identity(sequence, partner_indices):
    """
    Generates one-hot encoding of the partner base.
    If partner_indices[i] == -1 (unpaired), returns a zero vector.
    """
    length = len(sequence)
    num_classes = len(SEQ_MAP)
    encoding = np.zeros((length, num_classes), dtype=np.float32)

    seq_indices = [SEQ_MAP.get(c, -1) for c in sequence]

    for i, p_idx in enumerate(partner_indices):
        if p_idx != -1:
            # Get the base index of the partner
            partner_base_idx = seq_indices[p_idx]
            if partner_base_idx != -1:
                encoding[i, partner_base_idx] = 1.0

    return encoding


def process_dataframe(df, is_test=False):
    """
    Processes a dataframe into numpy arrays for inputs, partner_indices, and targets.
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Input feature dimensions:
    # Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18 channels
    input_dim = 4 + 3 + 7 + 4

    inputs = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices_all = np.zeros((n_samples, seq_len), dtype=np.int32)
    targets = np.zeros((n_samples, seq_len, 5), dtype=np.float32)

    # Target columns in order
    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        # 1. Basic Features
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        # 2. Structural Topology
        p_indices = get_partner_indices(row["structure"])

        # 3. Partner Identity (Explicit Injection)
        if Config.USE_PARTNER_IDENTITY:
            partner_id_oh = get_partner_identity(row["sequence"], p_indices)
        else:
            partner_id_oh = np.zeros((seq_len, 4), dtype=np.float32)

        # Concatenate all features
        # Shape: (Seq_Len, 18)
        sample_input = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_id_oh], axis=1
        )

        inputs[idx] = sample_input
        partner_indices_all[idx] = p_indices

        # 4. Targets
        if not is_test:
            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    # Parse stringified list "[0.1, 0.2, ...]"
                    val_list = ast.literal_eval(val_str)
                    # Pad or truncate to seq_len (though usually length is seq_scored)
                    # We fill the rest with zeros or NaNs.
                    # The loss function handles masking via seq_scored.
                    length_to_fill = min(len(val_list), seq_len)
                    targets[idx, :length_to_fill, t_i] = val_list[:length_to_fill]
                except (ValueError, SyntaxError):
                    # Handle cases where parsing fails (should not happen in clean data)
                    pass

    return inputs, partner_indices_all, targets


def preprocess_data(mode="train", load_cached_data=True, debug_subset=None):
    """
    Main function to load data, process it, and cache it.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.
        debug_subset (int, optional): If set, only process this many samples.

    Returns:
        tuple: (inputs, partner_indices, targets, ids)
    """
    # Determine paths based on mode
    if mode == "train":
        csv_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        csv_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        csv_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            targets = data["targets"]
            ids = data["ids"]

            # If debugging, slice the cached data
            if debug_subset is not None:
                inputs = inputs[:debug_subset]
                partner_indices = partner_indices[:debug_subset]
                targets = targets[:debug_subset]
                ids = ids[:debug_subset]

            return inputs, partner_indices, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    if debug_subset is not None:
        df = df.head(debug_subset)

    is_test = mode == "test"
    inputs, partner_indices, targets = process_dataframe(df, is_test=is_test)
    ids = df["id"].values

    # Save to cache (only if not debugging, to avoid overwriting full cache with subset)
    if debug_subset is None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            inputs=inputs,
            partner_indices=partner_indices,
            targets=targets,
            ids=ids,
        )
        print(f"Saved {mode} data to cache: {cache_path}")

    return inputs, partner_indices, targets, ids


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, mode="train", load_cached_data=True, debug_subset=None):
        self.inputs, self.partner_indices, self.targets, self.ids = preprocess_data(
            mode=mode, load_cached_data=load_cached_data, debug_subset=debug_subset
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs: (Seq_Len, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,)
        # We keep -1 for unpaired here; logic in model handles masking/clamping
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # Targets: (Seq_Len, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "inputs": x,
            "partner_indices": p_idx,
            "targets": y,
            "id": self.ids[idx],
        }
