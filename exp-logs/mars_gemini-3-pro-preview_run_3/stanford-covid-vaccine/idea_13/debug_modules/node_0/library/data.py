import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# ==========================================
# Helper Functions
# ==========================================
def parse_structure_pairs(structure_str):
    """
    Parses a dot-bracket structure string to find paired indices.
    Returns a numpy array of shape (L,) where arr[i] is the index of the base
    paired with i, or -1 if i is unpaired.
    """
    L = len(structure_str)
    pair_indices = np.full(L, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i
            # If stack is empty, we have an unbalanced ')', which implies unpaired in this context
            # or malformed input, but we default to -1 (unpaired).

    return pair_indices


def one_hot_encode_sequence(seq, mapping, length, num_channels):
    """
    One-hot encodes a sequence string into a (Length, Channels) matrix.
    """
    encoding = np.zeros((length, num_channels), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def preprocess_data(df, is_test=False):
    """
    Converts a pandas DataFrame into numpy arrays for inputs, pair indices, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # Input channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    ids = df["id"].values

    # Targets
    targets = None
    if not is_test:
        # 5 targets
        targets = np.zeros(
            (num_samples, Config.SEQ_SCORED, Config.NUM_TARGETS), dtype=np.float32
        )

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Inputs
        # Sequence (4)
        seq_enc = one_hot_encode_sequence(row["sequence"], SEQ_MAP, seq_len, 4)
        # Structure (3)
        struct_enc = one_hot_encode_sequence(row["structure"], STRUCT_MAP, seq_len, 3)
        # Loop Type (7)
        loop_enc = one_hot_encode_sequence(
            row["predicted_loop_type"], LOOP_MAP, seq_len, 7
        )

        # Concatenate: (L, 4) + (L, 3) + (L, 7) -> (L, 14)
        inputs[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 2. Pair Indices
        pair_indices[idx] = parse_structure_pairs(row["structure"])

        # 3. Targets (if not test)
        if not is_test:
            # Extract lists and stack
            t_list = []
            for col in Config.TARGET_COLS:
                val = row[col]
                # Ensure it's a list/array of length SEQ_SCORED
                if len(val) > Config.SEQ_SCORED:
                    val = val[: Config.SEQ_SCORED]
                t_list.append(val)

            # Stack to (5, 68) then transpose to (68, 5)
            targets[idx] = np.array(t_list, dtype=np.float32).T

    return {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "targets": targets,
        "ids": ids,
    }


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing 'inputs', 'pair_indices', 'targets', 'ids'.
        """
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.targets = data_dict.get("targets")
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Pair Indices: (107,)
        # Note: We keep -1 for unpaired. The model handles this (e.g., via embedding or masking).
        p = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        sample = {"inputs": x, "pair_indices": p, "id": self.ids[idx]}

        if self.targets is not None:
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        return sample


# ==========================================
# Main Data Loading Function
# ==========================================
def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Prepares and returns DataLoaders for train, validation, and test sets.
    Handles caching to disk to avoid re-processing.
    """
    Config.create_dirs()

    # Define cache paths
    cache_files = {
        "train": Config.TRAIN_CACHE,
        "val": Config.VAL_CACHE,
        "test": Config.TEST_CACHE,
    }

    metadata_files = {
        "train": Config.TRAIN_METADATA,
        "val": Config.VAL_METADATA,
        "test": Config.TEST_METADATA,
    }

    datasets = {}

    for split in ["train", "val", "test"]:
        cache_path = cache_files[split]
        meta_path = metadata_files[split]
        is_test = split == "test"

        data_dict = None

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # Allow pickle=True because we are saving a dict
                data_dict = np.load(cache_path, allow_pickle=True).item()
                print(f"Loaded {split} data from cache: {cache_path}")
            except Exception as e:
                print(f"Failed to load cache for {split}: {e}. Re-processing.")
                data_dict = None

        # 2. Process if not loaded
        if data_dict is None:
            print(f"Processing {split} data from {meta_path}...")
            df = pd.read_parquet(meta_path)

            # Debugging subset
            if debug and split == "train":
                df = df.iloc[: Config.DEBUG_SUBSET_SIZE].reset_index(drop=True)
                print(f"DEBUG MODE: Using subset of {len(df)} samples for training.")

            data_dict = preprocess_data(df, is_test=is_test)

            # Save to cache
            np.save(cache_path, data_dict)
            print(f"Saved {split} data to cache: {cache_path}")

        # 3. Create Dataset
        datasets[split] = RNADataset(data_dict)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
