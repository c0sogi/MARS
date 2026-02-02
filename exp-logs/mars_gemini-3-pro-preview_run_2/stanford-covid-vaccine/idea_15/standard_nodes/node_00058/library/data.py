import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Mappings
# =============================================================================
TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGCU")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate(".()")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}

# =============================================================================
# Helper Functions
# =============================================================================


def get_couples(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] is the index of the base paired with i,
    or -1 if unpaired.
    """
    pairs = np.full(len(structure), -1, dtype=int)
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


def parse_list_column(x):
    """Parses a stringified list into a numpy array."""
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


def process_data(df, is_test=False):
    """
    Generates features, neighbor indices, and targets from a dataframe.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 19

    # Pre-allocate arrays
    # Inputs: (N, 107, 19)
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    # Neighbor Indices: (N, 107, 1) -> [partner] (Cite Lesson 00023)
    neighbor_indices = np.full((num_samples, seq_len, 1), -1, dtype=np.int32)
    # Targets: (N, 107, 5)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    # IDs
    ids = df["id"].values

    # Target columns to process
    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        # Adjust index for numpy array (idx might be non-sequential if filtered)
        # We use enumerate in the outer loop or reset index before.
        # Here we assume df index is not reliable for array pos, so we use a counter.
        pass

    # Reset index to ensure safe iteration
    df = df.reset_index(drop=True)

    for i, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Basic One-Hot Encoding
        # Sequence (0-3)
        for j, char in enumerate(sequence):
            if char in TOKEN2INT_SEQ:
                inputs[i, j, TOKEN2INT_SEQ[char]] = 1.0

        # Structure (4-6)
        for j, char in enumerate(structure):
            if char in TOKEN2INT_STRUCT:
                inputs[i, j, 4 + TOKEN2INT_STRUCT[char]] = 1.0

        # Loop Type (7-13)
        for j, char in enumerate(loop_type):
            if char in TOKEN2INT_LOOP:
                inputs[i, j, 7 + TOKEN2INT_LOOP[char]] = 1.0

        # 2. Partner Identity & Neighbor Indices
        pairs = get_couples(structure)

        for j in range(seq_len):
            partner_idx = pairs[j]

            if partner_idx != -1:
                # Partner Identity (14-18)
                # 14:A, 15:G, 16:C, 17:U, 18:None
                partner_base = sequence[partner_idx]
                if partner_base in TOKEN2INT_SEQ:
                    inputs[i, j, 14 + TOKEN2INT_SEQ[partner_base]] = 1.0

                # Neighbor Indices [partner]
                neighbor_indices[i, j, 0] = partner_idx
            else:
                # Unpaired: Set "None" channel for partner identity
                inputs[i, j, 18] = 1.0
                # Neighbor indices remain -1

        # 3. Targets
        if not is_test:
            for k, col in enumerate(target_cols):
                # Parse string list
                val_arr = parse_list_column(row[col])
                # Pad to seq_len (usually data is 68 long)
                length = len(val_arr)
                if length > 0:
                    targets[i, :length, k] = val_arr

    return inputs, neighbor_indices, targets, ids


def load_or_generate_data(csv_path, cache_key, is_test=False, debug=False):
    """
    Loads data from cache if available, otherwise processes CSV and caches.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_key)

    if os.path.exists(cache_path):
        # Load from cache
        data = np.load(cache_path, allow_pickle=True)
        inputs = data["inputs"]
        neighbor_indices = data["neighbor_indices"]
        targets = data["targets"]
        ids = data["ids"]
    else:
        # Generate
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        inputs, neighbor_indices, targets, ids = process_data(df, is_test=is_test)

        # Save to cache
        np.savez_compressed(
            cache_path,
            inputs=inputs,
            neighbor_indices=neighbor_indices,
            targets=targets,
            ids=ids,
        )

    if debug:
        # Subsample
        inputs = inputs[:32]
        neighbor_indices = neighbor_indices[:32]
        targets = targets[:32]
        ids = ids[:32]

    return inputs, neighbor_indices, targets, ids


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, inputs, neighbor_indices, targets, ids, is_test=False):
        self.inputs = inputs
        self.neighbor_indices = neighbor_indices
        self.targets = targets
        self.ids = ids
        self.is_test = is_test

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        idx_map = torch.tensor(self.neighbor_indices[idx], dtype=torch.long)

        if self.is_test:
            # Return dummy targets or zeros for test
            y = torch.zeros((Config.SEQ_LEN, 5), dtype=torch.float32)
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return x, idx_map, y


# =============================================================================
# Data Loaders
# =============================================================================


def get_loaders(debug=False):
    """
    Returns train, val, and test dataloaders.
    """
    # Train
    train_inputs, train_idx, train_targets, train_ids = load_or_generate_data(
        Config.TRAIN_CSV, Config.TRAIN_CACHE_KEY, is_test=False, debug=debug
    )
    train_dataset = RNADataset(
        train_inputs, train_idx, train_targets, train_ids, is_test=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Val
    val_inputs, val_idx, val_targets, val_ids = load_or_generate_data(
        Config.VAL_CSV, Config.VAL_CACHE_KEY, is_test=False, debug=debug
    )
    val_dataset = RNADataset(val_inputs, val_idx, val_targets, val_ids, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test
    test_inputs, test_idx, test_targets, test_ids = load_or_generate_data(
        Config.TEST_CSV, Config.TEST_CACHE_KEY, is_test=True, debug=debug
    )
    test_dataset = RNADataset(
        test_inputs, test_idx, test_targets, test_ids, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Return IDs for test set submission generation
    return train_loader, val_loader, test_loader, test_ids
