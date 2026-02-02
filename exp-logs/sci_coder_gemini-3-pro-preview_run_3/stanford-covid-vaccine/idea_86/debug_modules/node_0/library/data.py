import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
TOKEN_TO_INDEX_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_TO_INDEX_STRUCT = {".": 0, "(": 1, ")": 2}
TOKEN_TO_INDEX_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# ==========================================
# Helper Functions
# ==========================================
def parse_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.

    Args:
        structure (str): Dot-bracket string (e.g., "..((..))..").

    Returns:
        pair_indices (np.ndarray): Shape (seq_len,). pair_indices[i] = j if i pairs with j.
                                   If unpaired, value is -1.
    """
    seq_len = len(structure)
    pair_indices = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_index = stack.pop()
                pair_indices[start_index] = i
                pair_indices[i] = start_index

    return pair_indices


def one_hot_encode(sequence, mapping, num_classes):
    """
    One-hot encodes a sequence string based on a mapping dictionary.

    Args:
        sequence (str): Input string.
        mapping (dict): Dictionary mapping char to index.
        num_classes (int): Total number of classes (channels).

    Returns:
        np.ndarray: Shape (len(sequence), num_classes).
    """
    seq_len = len(sequence)
    encoding = np.zeros((seq_len, num_classes), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def preprocess_dataset(df, has_targets=True):
    """
    Converts a dataframe into numpy arrays for inputs, structural info, and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        has_targets (bool): Whether to extract target columns.

    Returns:
        tuple: (inputs, pair_indices, pair_mask, targets, ids)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 14

    # Initialize arrays
    # Inputs: (N, 107, 14)
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)

    # Pair Indices: (N, 107) - stores index j for pair (i, j).
    # We will set unpaired indices to 0 for safe gathering, but mask them out.
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)

    # Pair Mask: (N, 107) - 1.0 if paired, 0.0 if unpaired.
    pair_mask = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: (N, 68, 5)
    targets = None
    if has_targets:
        targets = np.zeros((num_samples, Config.SEQ_SCORED, 5), dtype=np.float32)

    ids = []

    for idx, row in df.iterrows():
        # 1. Inputs
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Safety check for length
        if len(seq) != seq_len:
            continue

        # One-hot encoding
        oh_seq = one_hot_encode(seq, TOKEN_TO_INDEX_SEQ, 4)
        oh_struct = one_hot_encode(struct, TOKEN_TO_INDEX_STRUCT, 3)
        oh_loop = one_hot_encode(loop, TOKEN_TO_INDEX_LOOP, 7)

        # Concatenate: (107, 14)
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop], axis=1)
        inputs[idx] = sample_input

        # 2. Structural Pairs
        p_indices = parse_structure_pairs(struct)

        # Create mask: 1 where p_indices != -1
        mask = (p_indices != -1).astype(np.float32)
        pair_mask[idx] = mask

        # Create safe indices for gather: replace -1 with 0
        # The mask will zero out the gathered value anyway.
        safe_indices = p_indices.copy()
        safe_indices[safe_indices == -1] = 0
        pair_indices[idx] = safe_indices

        # 3. Targets
        if has_targets:
            # Targets are lists in the dataframe
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            try:
                t_reactivity = np.array(row["reactivity"], dtype=np.float32)
                t_deg_Mg_pH10 = np.array(row["deg_Mg_pH10"], dtype=np.float32)
                t_deg_pH10 = np.array(row["deg_pH10"], dtype=np.float32)
                t_deg_Mg_50C = np.array(row["deg_Mg_50C"], dtype=np.float32)
                t_deg_50C = np.array(row["deg_50C"], dtype=np.float32)

                # Stack: (68, 5)
                sample_targets = np.stack(
                    [t_reactivity, t_deg_Mg_pH10, t_deg_pH10, t_deg_Mg_50C, t_deg_50C],
                    axis=1,
                )

                targets[idx] = sample_targets
            except Exception:
                # Fallback for potential malformed data, though metadata check passed
                pass

        ids.append(row["id"])

    return inputs, pair_indices, pair_mask, targets, np.array(ids)


def get_cache_path(split_name):
    return os.path.join(Config.CACHE_DIR, f"{split_name}_data.npz")


def load_or_process_data(
    split_name, parquet_path, load_cached_data=True, debug=False, debug_size=100
):
    """
    Loads data from cache or processes from Parquet.
    """
    cache_path = get_cache_path(split_name)

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            pair_indices = data["pair_indices"]
            pair_mask = data["pair_mask"]
            targets = data["targets"] if "targets" in data else None
            ids = data["ids"]

            # Handle Debug Slicing after loading cache
            if debug:
                print(f"Debug mode: Slicing {split_name} to {debug_size} samples.")
                inputs = inputs[:debug_size]
                pair_indices = pair_indices[:debug_size]
                pair_mask = pair_mask[:debug_size]
                if targets is not None:
                    targets = targets[:debug_size]
                ids = ids[:debug_size]

            return inputs, pair_indices, pair_mask, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Reset index
    df = df.reset_index(drop=True)

    if debug:
        print(f"Debug mode: Slicing raw dataframe to {debug_size} samples.")
        df = df.iloc[:debug_size].reset_index(drop=True)

    # Check for targets presence
    has_targets = "reactivity" in df.columns

    inputs, pair_indices, pair_mask, targets, ids = preprocess_dataset(
        df, has_targets=has_targets
    )

    # Save to cache (only if not debugging)
    if not debug:
        print(f"Saving {split_name} data to cache: {cache_path}")
        save_dict = {
            "inputs": inputs,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "ids": ids,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_path, **save_dict)

    return inputs, pair_indices, pair_mask, targets, ids


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, pair_mask, targets=None, ids=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.pair_mask = torch.tensor(pair_mask, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_mask": self.pair_mask[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


# ==========================================
# Main Loader Function
# ==========================================
def get_dataloaders(
    load_cached_data=True, debug=False, debug_size=100, batch_size=32, num_workers=2
):
    """
    Main function to get DataLoaders for Train, Val, and Test.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Train Data
    train_inputs, train_pairs, train_mask, train_targets, train_ids = (
        load_or_process_data(
            "train", Config.TRAIN_DATA_PATH, load_cached_data, debug, debug_size
        )
    )
    train_dataset = RNADataset(
        train_inputs, train_pairs, train_mask, train_targets, train_ids
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Val Data
    val_inputs, val_pairs, val_mask, val_targets, val_ids = load_or_process_data(
        "val", Config.VAL_DATA_PATH, load_cached_data, debug, debug_size
    )
    val_dataset = RNADataset(val_inputs, val_pairs, val_mask, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Test Data
    test_inputs, test_pairs, test_mask, test_targets, test_ids = load_or_process_data(
        "test", Config.TEST_DATA_PATH, load_cached_data, debug, debug_size
    )
    test_dataset = RNADataset(
        test_inputs, test_pairs, test_mask, test_targets, test_ids
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
