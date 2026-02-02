import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants and Mappings
# ==========================================
# Sequence: 4 bases
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}

# Structure: 3 types
STRUCT_MAP = {".": 0, "(": 1, ")": 2}

# Loop Type: 7 types
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure to find base pairs.
    Returns:
        pairs: dict mapping index -> partner_index
    """
    pairs = {}
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


def one_hot_encode(seq, mapping, num_classes):
    """
    Helper to one-hot encode a sequence string based on a mapping.
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets):
        """
        Args:
            inputs: (N, SeqLen, Channels) - Combined features
            partner_indices: (N, SeqLen) - Indices of paired bases (-1 if unpaired)
            targets: (N, SeqLen, 5) - Ground truth (padded with 0.0)
        """
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return x, p_idx, y


# ==========================================
# Preprocessing & Caching
# ==========================================
def preprocess_and_cache(mode="train", load_cached_data=True):
    """
    Loads raw data, processes features/targets, and caches as .npz.

    Args:
        mode: 'train', 'val', or 'test'
        load_cached_data: If True, attempts to load from cache first.

    Returns:
        RNADataset object
    """
    # Determine paths
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        cache_path = Config.CACHE_TRAIN
    elif mode == "val":
        csv_path = Config.VAL_CSV
        cache_path = Config.CACHE_VAL
    elif mode == "test":
        csv_path = Config.TEST_CSV
        cache_path = Config.CACHE_TEST
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{mode.upper()}] Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return RNADataset(
                inputs=data["inputs"],
                partner_indices=data["partner_indices"],
                targets=data["targets"],
            )
        except Exception as e:
            print(f"[{mode.upper()}] Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"[{mode.upper()}] Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Pre-allocate lists
    all_inputs = []
    all_partner_indices = []
    all_targets = []

    # Target columns to parse
    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]
        length = len(seq)

        # --- Feature Engineering ---
        # 1. Basic One-Hot Encodings
        oh_seq = one_hot_encode(seq, SEQ_MAP, 4)  # (L, 4)
        oh_struct = one_hot_encode(struct, STRUCT_MAP, 3)  # (L, 3)
        oh_loop = one_hot_encode(loop, LOOP_MAP, 7)  # (L, 7)

        # 2. Partner Info
        pairs = get_structure_pairs(struct)
        partner_idx_arr = np.full(length, -1, dtype=np.int32)
        partner_identity = np.zeros((length, 4), dtype=np.float32)

        for i in range(length):
            if i in pairs:
                j = pairs[i]
                partner_idx_arr[i] = j
                # Partner identity is the one-hot of the base at index j
                partner_char = seq[j]
                if partner_char in SEQ_MAP:
                    partner_identity[i, SEQ_MAP[partner_char]] = 1.0

        # Concatenate all features: 4 + 3 + 7 + 4 = 18 channels
        # (Seq, Struct, Loop, PartnerID)
        sample_input = np.concatenate(
            [oh_seq, oh_struct, oh_loop, partner_identity], axis=1
        )

        # --- Target Engineering ---
        sample_targets = np.zeros((length, Config.NUM_TARGETS), dtype=np.float32)

        if mode != "test":
            for t_i, col in enumerate(target_cols):
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(row[col])
                    # Fill valid positions (usually first 68)
                    # The rest remain 0.0 (Anchoring)
                    valid_len = min(len(val_list), length)
                    sample_targets[:valid_len, t_i] = val_list[:valid_len]
                except (ValueError, SyntaxError):
                    # Handle cases where data might be missing or malformed
                    pass

        all_inputs.append(sample_input)
        all_partner_indices.append(partner_idx_arr)
        all_targets.append(sample_targets)

    # Convert to numpy arrays
    inputs_np = np.array(all_inputs, dtype=np.float32)
    partner_indices_np = np.array(all_partner_indices, dtype=np.int64)
    targets_np = np.array(all_targets, dtype=np.float32)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=inputs_np,
        partner_indices=partner_indices_np,
        targets=targets_np,
    )
    print(f"[{mode.upper()}] Saved processed data to {cache_path}")

    return RNADataset(inputs_np, partner_indices_np, targets_np)


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    train_ds = preprocess_and_cache("train")
    val_ds = preprocess_and_cache("val")
    test_ds = preprocess_and_cache("test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
