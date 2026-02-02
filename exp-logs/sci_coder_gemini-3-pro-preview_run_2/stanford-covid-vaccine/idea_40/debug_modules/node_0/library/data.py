import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_list_column

# =============================================================================
# Vocabularies & Mappings
# =============================================================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# Inverse mappings for debugging if needed, but not used in core logic
IDX_TO_SEQ = {v: k for k, v in SEQ_MAP.items()}

# =============================================================================
# Helper Functions
# =============================================================================


def get_partner_indices(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] is the index of the base paired with i,
    or -1 if i is unpaired.
    """
    n = len(structure)
    partner = np.full(n, -1, dtype=np.int32)
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


def one_hot_encode(seq, mapping, vocab_size):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    Shape: (Length, VocabSize)
    """
    arr = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_data(df, config, mode="train"):
    """
    Processes a dataframe into numpy arrays suitable for the model.

    Args:
        df (pd.DataFrame): Input dataframe containing sequences and structures.
        config (Config): Configuration object.
        mode (str): 'train' (expects targets) or 'test' (no targets).

    Returns:
        dict: Dictionary containing 'inputs', 'partner_indices', 'targets', 'ids'.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize arrays
    # Channels: Seq(4) + Struct(3) + Loop(7) + PartnerSeq(4) = 18
    num_channels = 4 + 3 + 7 + 4

    inputs = np.zeros((num_samples, seq_len, num_channels), dtype=np.float32)
    partner_indices_all = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: (N, SeqLen, 5)
    # We pad targets to seq_len even if they are shorter (68)
    targets = np.zeros((num_samples, seq_len, config.num_targets), dtype=np.float32)

    ids = df["id"].values

    # Pre-compute column availability
    has_targets = mode == "train" and all(
        col in df.columns for col in config.target_cols
    )

    for idx, row in df.iterrows():
        # 0. Base Index (for array filling)
        # Reset index if df index is not 0-based contiguous
        # Using enumerate on df.itertuples() or similar is safer, but here we use a counter
        # Actually, let's use a separate counter or convert df to list of dicts
        pass

    # Re-implement loop for safety with indices
    for i, row in enumerate(df.to_dict("records")):
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 1. Basic One-Hot Encodings
        oh_seq = one_hot_encode(sequence, SEQ_MAP, 4)  # (107, 4)
        oh_struct = one_hot_encode(structure, STRUCT_MAP, 3)  # (107, 3)
        oh_loop = one_hot_encode(loop_type, LOOP_MAP, 7)  # (107, 7)

        # 2. Partner Indices
        p_indices = get_partner_indices(structure)
        partner_indices_all[i] = p_indices

        # 3. Partner Identity
        # Create a (107, 4) array. For each pos, if paired, get partner's OH seq.
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)

        # Vectorized gather for this sample
        # Mask of paired bases
        paired_mask = p_indices != -1
        # Get indices of partners (safe to use 0 for -1 temporarily, then mask)
        safe_indices = p_indices.copy()
        safe_indices[~paired_mask] = 0

        # Gather
        oh_partner[paired_mask] = oh_seq[safe_indices[paired_mask]]

        # 4. Concatenate Inputs
        # Order: Seq, Struct, Loop, PartnerSeq
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)
        inputs[i] = sample_input

        # 5. Targets (if available)
        if has_targets:
            for t_idx, col in enumerate(config.target_cols):
                # Parse stringified list
                val = parse_list_column(row[col])
                # Copy to targets array (handle length mismatch by slicing/padding)
                length = min(len(val), seq_len)
                targets[i, :length, t_idx] = val[:length]

    return {
        "inputs": inputs,
        "partner_indices": partner_indices_all,
        "targets": targets,
        "ids": ids,
    }


# =============================================================================
# Dataset Class
# =============================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays from process_data.
        """
        self.inputs = torch.tensor(data_dict["inputs"], dtype=torch.float32)
        self.partner_indices = torch.tensor(
            data_dict["partner_indices"], dtype=torch.long
        )
        self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.partner_indices[idx], self.targets[idx]


# =============================================================================
# Main Data Loading Function
# =============================================================================


def get_dataloaders(config):
    """
    Prepares DataLoaders for train, validation, and test sets.
    Handles caching of processed data to .npz files.

    Args:
        config (Config): Configuration object.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    def load_or_process(csv_path, cache_path, mode="train"):
        # Check cache
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                return {
                    "inputs": loaded["inputs"],
                    "partner_indices": loaded["partner_indices"],
                    "targets": loaded["targets"],
                    "ids": loaded["ids"],
                }
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        # Process from scratch
        print(f"Processing data from {csv_path}...")
        df = pd.read_csv(csv_path)
        data_dict = process_data(df, config, mode=mode)

        # Save cache
        print(f"Saving cache to {cache_path}...")
        np.savez_compressed(
            cache_path,
            inputs=data_dict["inputs"],
            partner_indices=data_dict["partner_indices"],
            targets=data_dict["targets"],
            ids=data_dict["ids"],
        )
        return data_dict

    # 1. Load Data (Train/Val/Test)
    train_data = load_or_process(config.train_csv, config.train_cache, mode="train")
    val_data = load_or_process(config.val_csv, config.val_cache, mode="train")
    test_data = load_or_process(config.test_csv, config.test_cache, mode="test")

    # 2. Handle Debug/Subset
    if config.debug and config.subset_size:
        print(f"Debug mode: Slicing datasets to {config.subset_size} samples.")
        for d in [train_data, val_data, test_data]:
            limit = min(len(d["inputs"]), config.subset_size)
            d["inputs"] = d["inputs"][:limit]
            d["partner_indices"] = d["partner_indices"][:limit]
            d["targets"] = d["targets"][:limit]
            d["ids"] = d["ids"][:limit]

    # 3. Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # 4. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    return train_loader, val_loader, test_loader
