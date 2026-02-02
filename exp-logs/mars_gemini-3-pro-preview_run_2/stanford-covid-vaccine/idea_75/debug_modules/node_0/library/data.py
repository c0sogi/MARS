import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library import config

# =========================================================================================
# Constants & Mappings
# =========================================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# =========================================================================================
# Helper Functions
# =========================================================================================
def get_structure_couples(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a numpy array where arr[i] = j if i is paired with j, else -1.
    """
    pairing = np.full(len(structure), -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairing[i] = j
                pairing[j] = i
    return pairing


def one_hot(indices, depth):
    """
    Creates a one-hot encoding numpy array from a list of indices.
    """
    arr = np.zeros((len(indices), depth), dtype=np.float32)
    for i, val in enumerate(indices):
        if 0 <= val < depth:
            arr[i, val] = 1.0
    return arr


# =========================================================================================
# Data Processing
# =========================================================================================
def process_data(df, is_test=False):
    """
    Generates features and targets from the dataframe.

    Features generated:
    1. Sequence One-Hot (4 channels)
    2. Structure One-Hot (3 channels)
    3. Loop Type One-Hot (7 channels)
    4. Partner Identity One-Hot (4 channels)
    Total Input Channels: 18

    Returns:
        inputs: (N, L, 18)
        partner_indices: (N, L)
        targets: (N, L, 5) or None
    """
    num_samples = len(df)
    seq_len = config.SEQ_LENGTH

    # Feature dimensions: Seq(4) + Struct(3) + Loop(7) + PartnerID(4)
    input_dim = 4 + 3 + 7 + 4

    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    partner_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    targets = (
        np.zeros((num_samples, seq_len, 5), dtype=np.float32) if not is_test else None
    )

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # --- 1. Base Encodings ---
        seq_int = [SEQ_MAP.get(c, -1) for c in sequence]
        struct_int = [STRUCT_MAP.get(c, -1) for c in structure]
        loop_int = [LOOP_MAP.get(c, -1) for c in loop_type]

        oh_seq = one_hot(seq_int, 4)
        oh_struct = one_hot(struct_int, 3)
        oh_loop = one_hot(loop_int, 7)

        # --- 2. Partner Index Map ---
        p_idx = get_structure_couples(structure)
        partner_indices[idx] = p_idx

        # --- 3. Partner Identity Feature ---
        # If i is paired with j, feature is one-hot of sequence[j]. Else zeros.
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i, j in enumerate(p_idx):
            if j != -1:
                # Use the already computed one-hot sequence for the partner
                oh_partner[i] = oh_seq[j]

        # Concatenate all features
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        inputs[idx] = sample_input

        # --- 4. Targets (Train/Val only) ---
        if not is_test:
            # Parse stringified lists for each target column
            for t_i, col in enumerate(config.TARGET_COLS):
                val_str = row[col]
                try:
                    val_list = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val_list = []

                # Fill the target array up to the available length (usually 68)
                # The rest remains 0 (padded)
                length = len(val_list)
                if length > 0:
                    targets[idx, :length, t_i] = val_list

    return inputs, partner_indices, targets


def get_data(mode="train", load_cached_data=True):
    """
    Loads data from cache or processes from metadata CSVs.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        inputs, partner_indices, targets
    """
    # Specific cache key for this idea version
    cache_filename = f"{mode}_data_hc_hsgfn_v1.npz"
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            # Load targets if they exist (train/val), else None
            targets = data["targets"] if "targets" in data.files else None
            return inputs, partner_indices, targets
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from metadata...")
    csv_path = os.path.join(config.METADATA_DIR, f"{mode}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file {csv_path} not found.")

    df = pd.read_csv(csv_path)
    is_test = mode == "test"

    inputs, partner_indices, targets = process_data(df, is_test=is_test)

    # 3. Save to Cache
    print(f"Saving {mode} data to {cache_path}...")
    save_dict = {"inputs": inputs, "partner_indices": partner_indices}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return inputs, partner_indices, targets


# =========================================================================================
# Dataset Class
# =========================================================================================
class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to tensors
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # Return dummy targets for test set to maintain consistent signature
            y = torch.zeros((config.SEQ_LENGTH, 5), dtype=torch.float32)

        return x, p_idx, y


# =========================================================================================
# DataLoader Factory
# =========================================================================================
def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # Train Loader
    train_inputs, train_pidx, train_targets = get_data("train", load_cached_data)
    train_dataset = RNADataset(train_inputs, train_pidx, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Val Loader
    val_inputs, val_pidx, val_targets = get_data("val", load_cached_data)
    val_dataset = RNADataset(val_inputs, val_pidx, val_targets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Test Loader
    test_inputs, test_pidx, _ = get_data("test", load_cached_data)
    test_dataset = RNADataset(test_inputs, test_pidx, targets=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
