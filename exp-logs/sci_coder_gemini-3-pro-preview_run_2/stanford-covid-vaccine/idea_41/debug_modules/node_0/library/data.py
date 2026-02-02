import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# Feature Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_partner_map(structure):
    """
    Parses a dot-bracket structure string to find the index of the paired base.
    Returns a numpy array of shape (L,) where arr[i] is the index of the base paired with i.
    If i is unpaired, arr[i] = -1.
    """
    L = len(structure)
    partner_map = np.full(L, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_map[i] = j
                partner_map[j] = i

    return partner_map


def get_partner_identity(sequence, partner_map):
    """
    Returns a one-hot encoded array (L, 4) representing the identity of the paired base.
    If unpaired, returns a zero vector.
    """
    L = len(sequence)
    identity = np.zeros((L, 4), dtype=np.float32)

    # Convert sequence to indices
    seq_indices = np.array([SEQ_MAP.get(c, -1) for c in sequence])

    # For every position i, if it has a partner j, get seq_indices[j]
    # We can do this with masking
    paired_mask = partner_map != -1
    partners = partner_map[paired_mask]

    if len(partners) > 0:
        partner_bases = seq_indices[partners]
        # Set one-hot
        # valid bases are 0-3. -1 indicates unknown char (shouldn't happen in clean data)
        valid_idx = partner_bases != -1

        # We need to map back to the original indices i
        # The 'identity' array at row 'i' should be one-hot of base at 'partner_map[i]'

        # Create a temporary one-hot for the whole sequence
        seq_onehot = np.zeros((L, 4), dtype=np.float32)
        for i, val in enumerate(seq_indices):
            if val != -1:
                seq_onehot[i, val] = 1.0

        # Gather
        identity[paired_mask] = seq_onehot[partners]

    return identity


def one_hot_encode(seq, mapping, length):
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None):
        """
        inputs: (N, L, C) - Concatenated features
        partner_indices: (N, L) - Indices of paired bases (-1 if unpaired)
        targets: (N, L, 5) - Target values (padded with 0 for unscored positions)
        """
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Features
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices
        # For gather operations, -1 is invalid. We replace -1 with the index itself (self-loop)
        # or 0. The model usually uses a mask to ignore unpaired interactions.
        # Here we replace -1 with the current position index 'i' so gather retrieves self,
        # and we generate a mask indicating if it was truly paired.
        p_idx_raw = self.partner_indices[idx]
        L = len(p_idx_raw)

        # Create safe indices: if -1, set to range(0, L)
        safe_indices = p_idx_raw.copy()
        unpaired_mask = safe_indices == -1
        safe_indices[unpaired_mask] = np.arange(L)[unpaired_mask]

        p_idx = torch.tensor(safe_indices, dtype=torch.long)

        # Pairing mask: 1 if paired, 0 if unpaired
        p_mask = torch.tensor((~unpaired_mask).astype(np.float32), dtype=torch.float32)

        # Targets
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # Dummy targets for test set
            y = torch.zeros((L, Config.NUM_TARGETS), dtype=torch.float32)

        return x, p_idx, p_mask, y


def process_dataframe(df):
    """
    Process a dataframe into numpy arrays for features and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Features: Seq(4) + Struct(3) + Loop(7) + PartnerId(4) = 18
    num_features = 4 + 3 + 7 + 4
    inputs = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: 5 channels
    # Initialize with 0. We will fill the first 68 positions.
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    has_targets = "reactivity" in df.columns

    for idx, row in df.iterrows():
        # 1. Base Features
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        # 2. Partner Info
        p_map = get_partner_map(row["structure"])
        p_id_oh = get_partner_identity(row["sequence"], p_map)

        # Concatenate features
        # Shape: (L, 18)
        sample_feat = np.concatenate([seq_oh, struct_oh, loop_oh, p_id_oh], axis=1)
        inputs[idx] = sample_feat
        partner_indices[idx] = p_map

        # 3. Targets
        if has_targets:
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Note: The order in Config.SCORED_COLS_INDICES suggests specific scoring,
            # but we load all 5 in the standard order for the model output.
            # Standard order in CSV lists: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

            # Helper to parse list string
            def parse_col(col_name):
                val_str = row[col_name]
                try:
                    # It might be a list already if loaded from JSON directly,
                    # but from CSV metadata it's a string representation of a list
                    if isinstance(val_str, str):
                        return np.array(ast.literal_eval(val_str), dtype=np.float32)
                    elif isinstance(val_str, list):
                        return np.array(val_str, dtype=np.float32)
                    return np.zeros(Config.SCORED_LENGTH, dtype=np.float32)
                except:
                    return np.zeros(Config.SCORED_LENGTH, dtype=np.float32)

            t_react = parse_col("reactivity")
            t_mg_ph10 = parse_col("deg_Mg_pH10")
            t_ph10 = parse_col("deg_pH10")
            t_mg_50c = parse_col("deg_Mg_50C")
            t_50c = parse_col("deg_50C")

            # Stack (5, 68) -> Transpose to (68, 5)
            # Ensure lengths match seq_scored (68)
            sl = Config.SCORED_LENGTH
            sample_targets = np.stack(
                [t_react[:sl], t_mg_ph10[:sl], t_ph10[:sl], t_mg_50c[:sl], t_50c[:sl]],
                axis=1,
            )

            # Place into (107, 5) array
            targets[idx, :sl, :] = sample_targets

    return inputs, partner_indices, targets


def load_data(split="train", load_cached_data=True, debug=False):
    """
    Loads data for the specified split ('train', 'val', 'test').
    Handles caching to .npz files.
    """
    # Determine paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif split == "val":
        meta_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif split == "test":
        meta_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            targets = data["targets"]

            if debug:
                inputs = inputs[: Config.DEBUG_SIZE]
                partner_indices = partner_indices[: Config.DEBUG_SIZE]
                targets = targets[: Config.DEBUG_SIZE]

            return RNADataset(inputs, partner_indices, targets)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from Scratch
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)

    if debug:
        df = df.head(Config.DEBUG_SIZE)

    inputs, partner_indices, targets = process_dataframe(df)

    # 3. Save Cache (only if not debugging, to avoid overwriting full cache with partial data)
    if not debug:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path, inputs=inputs, partner_indices=partner_indices, targets=targets
        )

    return RNADataset(inputs, partner_indices, targets)
