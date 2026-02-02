import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# --------------------------------------------------------------------------
# Mappings
# --------------------------------------------------------------------------
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ".": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def get_partner_indices(structure):
    """
    Parses dot-bracket structure to find paired indices.
    Returns an array of shape (L,) where arr[i] is the index of the base paired with i,
    or -1 if i is unpaired.
    """
    length = len(structure)
    partner_indices = np.full(length, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partner_indices[i] = j
                partner_indices[j] = i

    return partner_indices


def one_hot_encode(indices, num_classes):
    """
    One-hot encodes a 1D array of indices.
    Shape: (L,) -> (L, num_classes)
    """
    return np.eye(num_classes)[indices]


def process_single_sequence(seq, struct, loop):
    """
    Generates the static features for a single RNA sequence.
    Returns:
        static_features: (Seq_Len, 18)
        partner_indices: (Seq_Len,)
    """
    length = len(seq)

    # 1. Map characters to indices
    seq_idx = np.array([SEQ_MAP.get(c, 0) for c in seq])
    struct_idx = np.array([STRUCT_MAP.get(c, 1) for c in struct])
    loop_idx = np.array([LOOP_MAP.get(c, 5) for c in loop])

    # 2. Get Partner Indices
    partner_idx = get_partner_indices(struct)

    # 3. One-Hot Encoding
    # Sequence (4)
    oh_seq = one_hot_encode(seq_idx, 4)
    # Structure (3)
    oh_struct = one_hot_encode(struct_idx, 3)
    # Loop Type (7)
    oh_loop = one_hot_encode(loop_idx, 7)

    # 4. Partner Identity (4)
    # Create a zero-filled array for partner identity
    oh_partner = np.zeros((length, 4), dtype=np.float32)

    # Find positions that are paired
    paired_mask = partner_idx != -1

    # For paired positions, get the sequence index of the partner
    # partner_idx[paired_mask] gives the indices of the partners
    # seq_idx[partner_idx[paired_mask]] gives the base identity (0-3) of the partners
    if np.any(paired_mask):
        partner_base_indices = seq_idx[partner_idx[paired_mask]]
        oh_partner[paired_mask] = one_hot_encode(partner_base_indices, 4)

    # 5. Concatenate Static Features
    # 4 + 3 + 7 + 4 = 18 channels
    static_features = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)

    return static_features.astype(np.float32), partner_idx.astype(np.int64)


def parse_target_columns(df, target_cols):
    """
    Parses stringified lists in target columns into a 3D numpy array.
    Returns: (N, 68, Num_Targets)
    """
    parsed_targets = []
    for col in target_cols:
        # Convert string "[0.1, ...]" to list [0.1, ...]
        # Handle potential NaNs or errors by treating as empty lists or zeros if necessary,
        # but competition data is usually clean enough for literal_eval.
        # We assume the list length is 68.
        values = (
            df[col]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            .tolist()
        )
        parsed_targets.append(values)

    # Stack to get (Num_Targets, N, 68) -> Transpose to (N, 68, Num_Targets)
    # Note: values is list of lists. np.array(values) -> (N, 68)
    target_array = np.stack(
        [np.array(v, dtype=np.float32) for v in parsed_targets], axis=-1
    )
    return target_array


# --------------------------------------------------------------------------
# Main Processing Function
# --------------------------------------------------------------------------
def preprocess_data(split="train", load_cached_data=True):
    """
    Loads data from CSV, processes it into tensors, and caches the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npz cache.

    Returns:
        dict: Contains 'inputs', 'partner_indices', and optionally 'targets'/'ids'.
    """
    # Determine paths based on split
    if split == "train":
        csv_path = Config.TRAIN_CSV
        cache_path = Config.TRAIN_CACHE
    elif split == "val":
        csv_path = Config.VAL_CSV
        cache_path = Config.VAL_CACHE
    elif split == "test":
        csv_path = Config.TEST_CSV
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            return dict(data)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {split} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Initialize containers
    all_inputs = []
    all_partner_indices = []

    # Process sequences
    # Using a loop is acceptable for dataset size ~2000
    for _, row in df.iterrows():
        feats, p_idx = process_single_sequence(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        all_inputs.append(feats)
        all_partner_indices.append(p_idx)

    inputs_arr = np.array(all_inputs, dtype=np.float32)  # (N, 107, 18)
    partner_indices_arr = np.array(all_partner_indices, dtype=np.int64)  # (N, 107)
    ids_arr = df["id"].values

    save_dict = {
        "inputs": inputs_arr,
        "partner_indices": partner_indices_arr,
        "ids": ids_arr,
    }

    # Process targets for train/val
    if split != "test":
        targets_arr = parse_target_columns(df, Config.TARGET_COLS)  # (N, 68, 5)
        save_dict["targets"] = targets_arr

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, **save_dict)
    print(f"Saved {split} data to cache: {cache_path}")

    return save_dict


# --------------------------------------------------------------------------
# Dataset Class
# --------------------------------------------------------------------------
class RNADataset(Dataset):
    def __init__(self, data, phase="train"):
        """
        Args:
            data (dict): Dictionary containing numpy arrays from preprocess_data.
            phase (str): 'train', 'val', or 'test'.
        """
        self.inputs = data["inputs"]
        self.partner_indices = data["partner_indices"]
        self.ids = data["ids"]
        self.phase = phase

        if phase != "test":
            self.targets = data["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (107, 18)
        # partner_indices: (107,)

        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
            "id": self.ids[idx],
        }

        if self.phase != "test":
            # targets: (68, 5)
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


# --------------------------------------------------------------------------
# DataLoader Factory
# --------------------------------------------------------------------------
def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load Data
    train_data = preprocess_data("train", load_cached_data)
    val_data = preprocess_data("val", load_cached_data)
    test_data = preprocess_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data, phase="train")
    val_dataset = RNADataset(val_data, phase="val")
    test_dataset = RNADataset(test_data, phase="test")

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
