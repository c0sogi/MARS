import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.

    Returns:
        inputs: (Seq_Len, Channels) - One-Hot encoded features
        partner_indices: (Seq_Len,) - Indices of paired bases (or self if unpaired)
        targets: (Seq_Scored, Num_Targets) - Ground truth values (only for train/val)
    """

    def __init__(self, inputs, partner_indices, targets=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_structure_pairs(structure):
    """
    Parses dot-bracket structure string to find base pairs.

    Args:
        structure (str): Dot-bracket notation (e.g., "((..))")

    Returns:
        np.array: Array of length len(structure).
                  arr[i] = j if base i is paired with base j.
                  arr[i] = i if base i is unpaired (maps to self).
    """
    seq_len = len(structure)
    pairs = np.arange(seq_len)  # Initialize with self-indices (unpaired assumption)
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


def get_one_hot_encoding(sequence, structure, loop_type):
    """
    Generates concatenated One-Hot encoding for Sequence, Structure, and Loop Type.

    Channels:
        Sequence (4): A, G, C, U
        Structure (3): ., (, )
        Loop Type (7): S, M, I, B, H, E, X

    Total Channels: 14
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    seq_len = len(sequence)

    # Initialize feature maps
    enc_seq = np.zeros((seq_len, 4), dtype=np.float32)
    enc_struct = np.zeros((seq_len, 3), dtype=np.float32)
    enc_loop = np.zeros((seq_len, 7), dtype=np.float32)

    # Fill feature maps
    for i, char in enumerate(sequence):
        if char in seq_map:
            enc_seq[i, seq_map[char]] = 1.0

    for i, char in enumerate(structure):
        if char in struct_map:
            enc_struct[i, struct_map[char]] = 1.0

    for i, char in enumerate(loop_type):
        if char in loop_map:
            enc_loop[i, loop_map[char]] = 1.0

    # Concatenate along channel dimension
    return np.concatenate([enc_seq, enc_struct, enc_loop], axis=1)


def process_data(csv_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from CSV, processes it into features/targets, and caches the result.
    Implements explicit cache invalidation via unique filenames in Config.
    """
    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            # Load targets if they exist in cache and we aren't in test mode (or if test cache has them for some reason)
            targets = data["targets"] if "targets" in data else None
            return inputs, partner_indices, targets
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from source.")

    # 2. Process from scratch
    print(f"Processing data from {csv_path}")
    df = pd.read_csv(csv_path)

    inputs_list = []
    partners_list = []
    targets_list = []

    # Target columns to parse
    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        # Extract raw strings
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Generate Features
        # 1. One-Hot Encoding
        one_hot = get_one_hot_encoding(seq, struct, loop)
        inputs_list.append(one_hot)

        # 2. Partner Index Map
        partners = get_structure_pairs(struct)
        partners_list.append(partners)

        # 3. Targets (Train/Val only)
        if not is_test:
            sample_targets = []
            for col in target_cols:
                val_str = row[col]
                try:
                    # Parse string representation of list "[0.1, 0.2, ...]"
                    val_list = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    # Fallback for malformed data (though metadata should be clean)
                    val_list = [0.0] * Config.SEQ_SCORED
                sample_targets.append(val_list)

            # Convert to numpy and transpose to (Seq_Len, Num_Targets)
            # Input lists are length 68, so result is (68, 5)
            sample_targets_arr = np.array(sample_targets, dtype=np.float32).T
            targets_list.append(sample_targets_arr)

    # Convert lists to numpy arrays
    inputs_arr = np.array(inputs_list, dtype=np.float32)  # (N, 107, 14)
    partners_arr = np.array(partners_list, dtype=np.int64)  # (N, 107)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not is_test:
        targets_arr = np.array(targets_list, dtype=np.float32)  # (N, 68, 5)
        np.savez(
            cache_path,
            inputs=inputs_arr,
            partner_indices=partners_arr,
            targets=targets_arr,
        )
        return inputs_arr, partners_arr, targets_arr
    else:
        np.savez(cache_path, inputs=inputs_arr, partner_indices=partners_arr)
        return inputs_arr, partners_arr, None


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Generates DataLoaders for Train, Validation, and Test sets.

    Args:
        debug (bool): If True, subsets data for rapid prototyping.
        load_cached_data (bool): If True, attempts to load preprocessed .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Retrieve paths from Config
    train_csv = Config.TRAIN_CSV
    val_csv = Config.VAL_CSV
    test_csv = Config.TEST_CSV

    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # Process Data
    train_inputs, train_partners, train_targets = process_data(
        train_csv, train_cache, load_cached_data, is_test=False
    )
    val_inputs, val_partners, val_targets = process_data(
        val_csv, val_cache, load_cached_data, is_test=False
    )
    test_inputs, test_partners, _ = process_data(
        test_csv, test_cache, load_cached_data, is_test=True
    )

    # Debugging Subset
    if debug:
        print(f"DEBUG MODE: Reducing dataset size to {Config.DEBUG_SUBSET_SIZE}")
        subset = Config.DEBUG_SUBSET_SIZE
        train_inputs = train_inputs[:subset]
        train_partners = train_partners[:subset]
        train_targets = train_targets[:subset]

        val_inputs = val_inputs[:subset]
        val_partners = val_partners[:subset]
        val_targets = val_targets[:subset]

        test_inputs = test_inputs[:subset]
        test_partners = test_partners[:subset]

    # Initialize Datasets
    train_dataset = RNADataset(train_inputs, train_partners, train_targets)
    val_dataset = RNADataset(val_inputs, val_partners, val_targets)
    test_dataset = RNADataset(test_inputs, test_partners, None)

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
