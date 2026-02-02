import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEQ_LEN,
    SCORED_LEN,
    NUM_TARGETS,
)

# =============================================================================
# CONSTANTS & MAPPINGS
# =============================================================================
TOKEN2INT_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN2INT_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN2INT_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_couples(structure):
    """
    Determines the pairing partners for each base in the RNA structure.
    Returns an array where arr[i] is the index of the base paired with i,
    or -1 if i is unpaired.
    """
    partners = np.full(len(structure), -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i
    return partners


def preprocess_data(df, is_test=False):
    """
    Generates features and targets from the dataframe.

    Features:
    - Sequence One-Hot (4 channels)
    - Structure One-Hot (3 channels)
    - Loop Type One-Hot (7 channels)
    - Partner Identity One-Hot (4 channels)

    Targets:
    - 5 channels, padded to 107 length with 0.0.
    """
    # Initialize lists to hold processed data
    sequences_enc = []
    structures_enc = []
    loops_enc = []
    partner_indices_list = []
    partner_identities_enc = []
    targets_list = []
    ids_list = []

    # Target columns in order
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # 1. Basic Encodings
        # Sequence
        seq_vec = np.zeros((SEQ_LEN, 4), dtype=np.float32)
        for i, char in enumerate(seq):
            if char in TOKEN2INT_SEQ:
                seq_vec[i, TOKEN2INT_SEQ[char]] = 1.0

        # Structure
        struct_vec = np.zeros((SEQ_LEN, 3), dtype=np.float32)
        for i, char in enumerate(struct):
            if char in TOKEN2INT_STRUCT:
                struct_vec[i, TOKEN2INT_STRUCT[char]] = 1.0

        # Loop Type
        loop_vec = np.zeros((SEQ_LEN, 7), dtype=np.float32)
        for i, char in enumerate(loop):
            if char in TOKEN2INT_LOOP:
                loop_vec[i, TOKEN2INT_LOOP[char]] = 1.0

        # 2. Partner Indices & Identity
        partners = get_couples(struct)
        partner_indices_list.append(partners)

        # Partner Identity: If i is paired with j, get one-hot of seq[j]
        partner_id_vec = np.zeros((SEQ_LEN, 4), dtype=np.float32)
        for i, p_idx in enumerate(partners):
            if p_idx != -1:
                # Copy the one-hot vector of the partner
                partner_id_vec[i] = seq_vec[p_idx]

        # Collect Features
        sequences_enc.append(seq_vec)
        structures_enc.append(struct_vec)
        loops_enc.append(loop_vec)
        partner_identities_enc.append(partner_id_vec)
        ids_list.append(row["id"])

        # 3. Targets (Training/Validation only)
        if not is_test:
            sample_targets = []
            for col in target_cols:
                val_str = row[col]
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(val_str)
                except:
                    val_list = [0.0] * SCORED_LEN  # Fallback

                # Pad to SEQ_LEN (107) with 0.0 for Boundary Anchoring
                # The provided list is usually length 68.
                padded_target = np.zeros(SEQ_LEN, dtype=np.float32)
                current_len = len(val_list)
                padded_target[:current_len] = val_list
                # Tail remains 0.0

                sample_targets.append(padded_target)

            # Stack to (SEQ_LEN, 5)
            targets_list.append(np.stack(sample_targets, axis=1))

    # Convert to numpy arrays
    # Concatenate features along channel dimension: 4 + 3 + 7 + 4 = 18 channels
    X_seq = np.array(sequences_enc)
    X_struct = np.array(structures_enc)
    X_loop = np.array(loops_enc)
    X_partner = np.array(partner_identities_enc)

    inputs = np.concatenate([X_seq, X_struct, X_loop, X_partner], axis=2)
    partner_indices = np.array(partner_indices_list)
    ids = np.array(ids_list)

    if is_test:
        targets = np.zeros((len(df), SEQ_LEN, NUM_TARGETS), dtype=np.float32)
    else:
        targets = np.array(targets_list)

    return inputs, partner_indices, targets, ids


def load_or_process_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from cache if available and requested.
    Otherwise, loads from CSV, processes, and saves to cache.
    """
    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return data["inputs"], data["partner_indices"], data["targets"], data["ids"]

    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)
    inputs, partner_indices, targets, ids = preprocess_data(df, is_test=is_test)

    print(f"Saving data to cache {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs,
        partner_indices=partner_indices,
        targets=targets,
        ids=ids,
    )

    return inputs, partner_indices, targets, ids


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.partner_indices = torch.tensor(partner_indices, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.partner_indices[idx], self.targets[idx]


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Val, and Test sets.
    """
    # 1. Load Data (Train)
    train_inputs, train_pi, train_targets, train_ids = load_or_process_data(
        TRAIN_METADATA_PATH, TRAIN_CACHE_PATH, load_cached_data, is_test=False
    )

    # 2. Load Data (Val)
    val_inputs, val_pi, val_targets, val_ids = load_or_process_data(
        VAL_METADATA_PATH, VAL_CACHE_PATH, load_cached_data, is_test=False
    )

    # 3. Load Data (Test)
    test_inputs, test_pi, test_targets, test_ids = load_or_process_data(
        TEST_METADATA_PATH, TEST_CACHE_PATH, load_cached_data, is_test=True
    )

    # Debug Subsetting
    if debug:
        subset_size = 100
        print(f"DEBUG MODE: Truncating datasets to {subset_size} samples.")
        train_inputs = train_inputs[:subset_size]
        train_pi = train_pi[:subset_size]
        train_targets = train_targets[:subset_size]
        train_ids = train_ids[:subset_size]

        val_inputs = val_inputs[:subset_size]
        val_pi = val_pi[:subset_size]
        val_targets = val_targets[:subset_size]
        val_ids = val_ids[:subset_size]

        test_inputs = test_inputs[:subset_size]
        test_pi = test_pi[:subset_size]
        test_targets = test_targets[:subset_size]
        test_ids = test_ids[:subset_size]

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_pi, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_pi, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_pi, test_targets, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
