import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Mappings for One-Hot Encoding
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_indices(structure):
    """
    Parses dot-bracket structure to find paired indices.
    Returns an array where arr[i] is the index of the base paired with i,
    or -1 if unpaired.
    """
    partner = np.full(len(structure), -1, dtype=np.int64)
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


def one_hot_encode(seq, map_dict, num_classes):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in map_dict:
            arr[i, map_dict[char]] = 1.0
    return arr


def preprocess_data(df, is_test=False):
    """
    Generates features and targets from the dataframe.

    Features (18 channels):
    - Sequence (4)
    - Structure (3)
    - Loop Type (7)
    - Partner Identity (4)

    Returns:
        inputs: (N, SeqLen, 18)
        partner_indices: (N, SeqLen)
        targets: (N, SeqLen, 5)
        ids: List of IDs
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, Config.IN_CHANNELS), dtype=np.float32)
    partner_indices_all = np.zeros((num_samples, seq_len), dtype=np.int64)
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    ids = df["id"].tolist()

    for idx, row in df.iterrows():
        # 1. Basic Features
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # Ensure length consistency
        if len(sequence) != seq_len:
            # Should not happen based on dataset description, but safety check
            continue

        # One-Hot Encodings
        oh_seq = one_hot_encode(sequence, SEQ_MAP, 4)
        oh_struct = one_hot_encode(structure, STRUCT_MAP, 3)
        oh_loop = one_hot_encode(loop_type, LOOP_MAP, 7)

        # 2. Partner Indices & Identity
        p_indices = get_structure_indices(structure)
        partner_indices_all[idx] = p_indices

        # Construct Partner Identity Feature
        # If i is paired with j, feature at i is one-hot of sequence[j]
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i, p_idx in enumerate(p_indices):
            if p_idx != -1:
                partner_base = sequence[p_idx]
                if partner_base in SEQ_MAP:
                    oh_partner[i, SEQ_MAP[partner_base]] = 1.0

        # Concatenate Features: 4 + 3 + 7 + 4 = 18
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        inputs[idx] = sample_input

        # 3. Targets
        if not is_test:
            for t_i, col in enumerate(Config.TARGET_COLS):
                # Parse stringified list
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                    # Target provided for first 68 bases
                    length_provided = len(val_list)
                    # Fill the provided part
                    targets[idx, :length_provided, t_i] = np.array(
                        val_list, dtype=np.float32
                    )
                    # Remaining part (68-107) stays 0.0 (padded)
                except Exception:
                    # In case of parsing error, leave as zeros
                    pass

    return inputs, partner_indices_all, targets, ids


def load_or_generate_data(csv_path, cache_path, is_test=False, debug=False):
    """
    Loads data from cache if available, otherwise processes CSV and caches result.
    """
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        inputs = data["inputs"]
        partner_indices = data["partner_indices"]
        targets = data["targets"]
        ids = data["ids"]
    else:
        print(f"Cache not found. Processing {csv_path}...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        inputs, partner_indices, targets, ids = preprocess_data(df, is_test=is_test)

        # Save to cache
        print(f"Saving data to {cache_path}...")
        np.savez_compressed(
            cache_path,
            inputs=inputs,
            partner_indices=partner_indices,
            targets=targets,
            ids=ids,
        )

    # Debug Subsetting
    if debug:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(inputs))
        inputs = inputs[:subset_size]
        partner_indices = partner_indices[:subset_size]
        targets = targets[:subset_size]
        ids = ids[:subset_size]
        print(f"DEBUG MODE: Subsetting to {subset_size} samples.")

    return inputs, partner_indices, targets, ids


class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, ids):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        # Inputs: (SeqLen, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: (SeqLen,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # Targets: (SeqLen, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return x, p_idx, y


def get_dataloaders(debug=False):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    # 1. Train Data
    train_inputs, train_pidx, train_targets, train_ids = load_or_generate_data(
        Config.TRAIN_METADATA, Config.TRAIN_CACHE, is_test=False, debug=debug
    )
    train_dataset = RNADataset(train_inputs, train_pidx, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # 2. Val Data
    val_inputs, val_pidx, val_targets, val_ids = load_or_generate_data(
        Config.VAL_METADATA, Config.VAL_CACHE, is_test=False, debug=debug
    )
    val_dataset = RNADataset(val_inputs, val_pidx, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # 3. Test Data
    test_inputs, test_pidx, test_targets, test_ids = load_or_generate_data(
        Config.TEST_METADATA, Config.TEST_CACHE, is_test=True, debug=debug
    )
    test_dataset = RNADataset(test_inputs, test_pidx, test_targets, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # Return IDs for submission generation later
    return train_loader, val_loader, test_loader, test_ids
