import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_list_column

# ------------------------------------------------------------------------
# Constants & Mappings
# ------------------------------------------------------------------------
SEQ_MAP = {c: i for i, c in enumerate("AGCU")}
STRUCT_MAP = {c: i for i, c in enumerate("().")}
LOOP_MAP = {c: i for i, c in enumerate("SMIBHEX")}


# ------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------
def get_one_hot(seq, mapping):
    """
    Converts a sequence string to a one-hot numpy array.

    Args:
        seq (str): Input string.
        mapping (dict): Character to index mapping.

    Returns:
        np.ndarray: One-hot encoded array of shape (len(seq), len(mapping)).
    """
    seq_len = len(seq)
    num_classes = len(mapping)
    one_hot = np.zeros((seq_len, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            one_hot[i, mapping[char]] = 1.0
    return one_hot


def process_structure_info(sequence, structure):
    """
    Parses structure to get partner indices and partner identity features.

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket notation structure.

    Returns:
        tuple:
            - partner_indices (np.ndarray): (L,) array, value is index of partner or -1.
            - partner_identity (np.ndarray): (L, 4) one-hot array of the partner base.
    """
    L = len(sequence)
    partner_indices = np.full(L, -1, dtype=np.int64)
    stack = []

    # 1. Parse Parentheses to find pairs
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i

    # 2. Generate Partner Identity Feature
    # If i is paired with j, partner_identity[i] is the one-hot encoding of sequence[j]
    # If unpaired, it remains all zeros.
    partner_identity = np.zeros((L, 4), dtype=np.float32)
    seq_one_hot = get_one_hot(sequence, SEQ_MAP)

    for i in range(L):
        j = partner_indices[i]
        if j != -1:
            partner_identity[i] = seq_one_hot[j]

    return partner_indices, partner_identity


# ------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------
class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets, ids):
        """
        Args:
            inputs (np.ndarray): Shape (N, L, 18).
            partner_indices (np.ndarray): Shape (N, L).
            targets (np.ndarray): Shape (N, L, 5).
            ids (np.ndarray): Shape (N,).
        """
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.partner_indices[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            self.ids[idx],
        )


# ------------------------------------------------------------------------
# Data Processing & Loading
# ------------------------------------------------------------------------
def process_data(csv_path, cache_path, mode="train", load_cached_data=True):
    """
    Processes data from CSV to numpy arrays, with caching.

    Args:
        csv_path (str): Path to the metadata CSV file.
        cache_path (str): Path to the .npz cache file.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (inputs, partner_indices, targets, ids)
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data["inputs"], data["partner_indices"], data["targets"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load CSV
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 3. Initialize containers
    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    # 4. Iterate and Process
    for idx, row in df.iterrows():
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # --- Feature Engineering ---
        oh_seq = get_one_hot(seq, SEQ_MAP)  # Shape: (L, 4)
        oh_struct = get_one_hot(struct, STRUCT_MAP)  # Shape: (L, 3)
        oh_loop = get_one_hot(loop, LOOP_MAP)  # Shape: (L, 7)

        # Compute Partner Indices and Partner Identity
        p_indices, p_identity = process_structure_info(
            seq, struct
        )  # p_ind: (L,), p_id: (L, 4)

        # Concatenate Input Features: Seq + Struct + Loop + PartnerID
        # Total Channels: 4 + 3 + 7 + 4 = 18
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, p_identity], axis=1)

        # --- Target Processing ---
        # Initialize (107, 5) with zeros
        sample_targets = np.zeros(
            (Config.SEQ_LEN, Config.NUM_TARGETS), dtype=np.float32
        )

        if mode in ["train", "val"]:
            # Parse targets from stringified lists
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_arr = parse_list_column(row[col])
                # Fill the valid positions (usually first 68)
                if len(val_arr) > 0:
                    length = min(len(val_arr), Config.SEQ_LEN)
                    sample_targets[:length, t_i] = val_arr[:length]

        all_inputs.append(sample_input)
        all_partner_indices.append(p_indices)
        all_targets.append(sample_targets)

    # 5. Convert to Numpy
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_partner_indices = np.array(all_partner_indices, dtype=np.int32)
    all_targets = np.array(all_targets, dtype=np.float32)

    # 6. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=all_inputs,
        partner_indices=all_partner_indices,
        targets=all_targets,
        ids=all_ids,
    )
    print(f"Saved processed data to {cache_path}")

    return all_inputs, all_partner_indices, all_targets, all_ids


def get_loaders(debug=False):
    """
    Generates DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define paths from Config
    train_csv = Config.TRAIN_CSV
    val_csv = Config.VAL_CSV
    test_csv = Config.TEST_CSV

    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # Process Data
    train_inputs, train_pi, train_targets, train_ids = process_data(
        train_csv, train_cache, mode="train"
    )
    val_inputs, val_pi, val_targets, val_ids = process_data(
        val_csv, val_cache, mode="val"
    )
    test_inputs, test_pi, test_targets, test_ids = process_data(
        test_csv, test_cache, mode="test"
    )

    # Debug Mode: Slice data to reduce size
    if debug:
        print("Debug mode: using subset of data")
        subset_size = 32
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
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
