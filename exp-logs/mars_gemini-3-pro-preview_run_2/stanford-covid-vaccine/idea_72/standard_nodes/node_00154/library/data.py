import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config

# =============================================================================
# CONSTANTS & MAPPINGS
# =============================================================================

SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_couples(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a mapping array where arr[i] = j means i is paired with j.
    If arr[i] = -1, i is unpaired.
    """
    mapping = np.full(len(structure), -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                mapping[i] = j
                mapping[j] = i
    return mapping


def one_hot_encode(seq, mapping, vocab_size):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    Returns array of shape (Length, Vocab_Size).
    """
    arr = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


# =============================================================================
# DATA PROCESSING
# =============================================================================


def process_data(csv_path, mode="train"):
    """
    Reads the CSV, generates features and targets.

    Args:
        csv_path (str): Path to the CSV file.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing numpy arrays for inputs, partner_indices, targets, ids.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debugging: Reduce dataset size if configured
    if config.DEBUG:
        df = df.head(config.DEBUG_SUBSET_SIZE)

    ids = df["id"].values

    # Pre-allocate lists
    all_inputs = []
    all_partner_indices = []
    all_targets = []

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]
        seq_len = len(sequence)

        # --- Feature Generation ---

        # 1. Basic One-Hot Encodings
        seq_oh = one_hot_encode(sequence, SEQ_MAP, config.VOCAB_SIZE)  # (L, 4)
        struct_oh = one_hot_encode(
            structure, STRUCT_MAP, config.STRUCTURE_VOCAB_SIZE
        )  # (L, 3)
        loop_oh = one_hot_encode(loop_type, LOOP_MAP, config.LOOP_VOCAB_SIZE)  # (L, 7)

        # 2. Partner Information
        partner_idx = get_couples(structure)

        # 3. Explicit Partner Identity
        # Create a tensor representing the sequence identity of the paired base.
        # If unpaired, this remains a zero vector.
        partner_seq_oh = np.zeros((seq_len, config.VOCAB_SIZE), dtype=np.float32)

        # Mask for paired bases
        paired_mask = partner_idx != -1
        if np.any(paired_mask):
            valid_partners = partner_idx[paired_mask]
            partner_seq_oh[paired_mask] = seq_oh[valid_partners]

        # Concatenate all features
        # Total Channels = 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerSeq) = 18
        sample_input = np.concatenate(
            [seq_oh, struct_oh, loop_oh, partner_seq_oh], axis=1
        )

        all_inputs.append(sample_input)
        all_partner_indices.append(partner_idx)

        # --- Target Generation ---

        # Initialize targets with zeros (Boundary Anchoring for positions 68-107)
        sample_target = np.zeros((seq_len, config.NUM_TARGETS), dtype=np.float32)

        if mode in ["train", "val"]:
            for t_i, col_name in enumerate(config.TARGET_COLS):
                val_str = row[col_name]
                try:
                    # Parse stringified list "[0.1, 0.2, ...]"
                    val_list = ast.literal_eval(val_str)
                    val_arr = np.array(val_list, dtype=np.float32)

                    # Fill only the scored positions (0 to 68)
                    limit = min(len(val_arr), config.SEQ_SCORED)
                    sample_target[:limit, t_i] = val_arr[:limit]
                except (ValueError, SyntaxError):
                    # In case of parsing error, leave as zeros
                    pass

        all_targets.append(sample_target)

    # Convert lists to numpy arrays
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_partner_indices = np.array(all_partner_indices, dtype=np.int64)
    all_targets = np.array(all_targets, dtype=np.float32)

    return {
        "inputs": all_inputs,
        "partner_indices": all_partner_indices,
        "targets": all_targets,
        "ids": ids,
    }


def get_dataset_data(mode, load_cached_data=True):
    """
    Manages caching and loading of processed data.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Data dictionary containing inputs, targets, etc.
    """
    # Determine paths
    if mode == "train":
        cache_path = config.CACHE_TRAIN
        csv_path = config.TRAIN_CSV
    elif mode == "val":
        cache_path = config.CACHE_VAL
        csv_path = config.VAL_CSV
    elif mode == "test":
        cache_path = config.CACHE_TEST
        csv_path = config.TEST_CSV
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": data["inputs"],
                "partner_indices": data["partner_indices"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache ({e}). Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from CSV: {csv_path}")
    data_dict = process_data(csv_path, mode=mode)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_path}")
    np.savez_compressed(
        cache_path,
        inputs=data_dict["inputs"],
        partner_indices=data_dict["partner_indices"],
        targets=data_dict["targets"],
        ids=data_dict["ids"],
    )

    return data_dict


# =============================================================================
# DATASET & DATALOADER
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict["inputs"]
        self.partner_indices = data_dict["partner_indices"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (Seq_Len, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # Targets: (Seq_Len, Targets)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Sample ID
        sample_id = self.ids[idx]

        return x, p_idx, y, sample_id


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_data = get_dataset_data("train", load_cached_data)
    val_data = get_dataset_data("val", load_cached_data)
    test_data = get_dataset_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create DataLoaders
    # Train: Shuffle=True, Drop_Last=True (for stable batch norm/stats)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=True,
    )

    # Val: Shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    # Test: Shuffle=False
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
