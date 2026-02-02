import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array of size len(structure) where:
        arr[i] = j if i is paired with j
        arr[i] = -1 if i is unpaired
    """
    pairs = np.full(len(structure), -1, dtype=np.int32)
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


def one_hot(indices, depth):
    """
    Manual one-hot encoding for numpy arrays.
    indices: Array of integers.
    depth: Depth of encoding.
    """
    n = len(indices)
    out = np.zeros((n, depth), dtype=np.float32)
    # Filter valid indices to prevent errors
    valid = (indices >= 0) & (indices < depth)
    out[np.arange(n)[valid], indices[valid]] = 1.0
    return out


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None):
        """
        Args:
            inputs: (N, Seq_Len, Input_Dim) float array
            partner_indices: (N, Seq_Len) int array with -1 for unpaired
            targets: (N, Seq_Len, Num_Targets) float array (optional)
        """
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 1. Input Features
        # Shape: (107, 19)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # 2. Partner Indices and Mask
        # Raw indices have -1 for unpaired.
        p_raw = self.partner_indices[idx]

        # Create Mask: 1.0 if paired, 0.0 if unpaired
        # Shape: (107,)
        p_mask = torch.tensor((p_raw != -1), dtype=torch.float32)

        # Prepare Indices for Gather: Replace -1 with 0 to avoid index errors in the model.
        # The mask will ensure these dummy 0-indexed values are ignored/zeroed out later.
        p_idx_safe = p_raw.copy()
        p_idx_safe[p_idx_safe == -1] = 0
        p_idx = torch.tensor(p_idx_safe, dtype=torch.long)

        # 3. Targets
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # Dummy targets for test set
            y = torch.zeros((Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32)

        return x, p_idx, p_mask, y


# =========================================================================
# Processing Pipeline
# =========================================================================


def process_data(csv_path, cache_path, is_test=False, load_cached_data=True):
    """
    Reads CSV, generates features and targets, handles caching.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            return data["inputs"], data["partner_indices"], data["targets"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load CSV
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 3. Initialize containers
    n_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 19

    all_inputs = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    all_partner_indices = np.zeros((n_samples, seq_len), dtype=np.int32)
    all_targets = np.zeros((n_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # 4. Process Loop
    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # --- Feature Engineering ---

        # A. Sequence (4 channels)
        seq_idxs = np.array([SEQ_MAP.get(c, 0) for c in sequence])
        feat_seq = one_hot(seq_idxs, 4)

        # B. Structure (3 channels)
        struct_idxs = np.array([STRUCT_MAP.get(c, 2) for c in structure])
        feat_struct = one_hot(struct_idxs, 3)

        # C. Loop Type (7 channels)
        loop_idxs = np.array([LOOP_MAP.get(c, 6) for c in loop_type])
        feat_loop = one_hot(loop_idxs, 7)

        # D. Partner Identity (5 channels) & Partner Indices
        pairs = get_structure_pairs(structure)
        all_partner_indices[idx] = pairs

        # Partner ID Indices: 0-3 for bases, 4 for None
        partner_id_idxs = np.full(seq_len, 4, dtype=np.int32)  # Default to 4 (None)

        # Identify paired positions
        paired_mask = pairs != -1
        if np.any(paired_mask):
            # Get indices of partners
            partners = pairs[paired_mask]
            # Get base identity of partners
            partner_bases = seq_idxs[partners]
            # Assign
            partner_id_idxs[paired_mask] = partner_bases

        feat_partner = one_hot(partner_id_idxs, 5)

        # Concatenate Features
        # Shape: (107, 19)
        features = np.concatenate(
            [feat_seq, feat_struct, feat_loop, feat_partner], axis=1
        )
        all_inputs[idx] = features

        # --- Target Processing ---
        if not is_test:
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                    # Targets are usually length 68, pad to 107
                    # We use 0.0 padding; loss function handles masking
                    valid_len = len(val_list)
                    all_targets[idx, :valid_len, t_i] = val_list
                except:
                    pass  # Keep zeros if parsing fails or data missing

    # 5. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=all_inputs,
        partner_indices=all_partner_indices,
        targets=all_targets,
    )
    print(f"Saved processed data to {cache_path}")

    return all_inputs, all_partner_indices, all_targets


def get_loaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Val, and Test sets.
    """
    # Train
    train_inputs, train_pidx, train_targets = process_data(
        Config.TRAIN_CSV,
        Config.TRAIN_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )
    train_ds = RNADataset(train_inputs, train_pidx, train_targets)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val
    val_inputs, val_pidx, val_targets = process_data(
        Config.VAL_CSV,
        Config.VAL_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )
    val_ds = RNADataset(val_inputs, val_pidx, val_targets)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test
    test_inputs, test_pidx, test_targets = process_data(
        Config.TEST_CSV,
        Config.TEST_CACHE,
        is_test=True,
        load_cached_data=load_cached_data,
    )
    test_ds = RNADataset(test_inputs, test_pidx, test_targets)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
