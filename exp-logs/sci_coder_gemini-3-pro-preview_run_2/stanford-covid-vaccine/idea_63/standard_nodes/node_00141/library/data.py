import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SEQ_LENGTH,
    SCORED_LENGTH,
    TARGET_COLS,
    BATCH_SIZE,
    DEBUG,
    DEBUG_SUBSET_SIZE,
)

# =============================================================================
# MAPPINGS
# =============================================================================
TOKEN_TO_INT_SEQ = {x: i for i, x in enumerate("AGCU")}
TOKEN_TO_INT_STRUCT = {x: i for i, x in enumerate("().")}
TOKEN_TO_INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns a list of tuples (i, j).
    """
    pairs = []
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs


def to_one_hot(seq, mapping, length, channels):
    """
    Converts a sequence string to a one-hot numpy array.
    """
    arr = np.zeros((length, channels), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


# =============================================================================
# DATA PROCESSING
# =============================================================================
def preprocess_data(df, mode="train"):
    """
    Generates features and targets from the dataframe.
    """
    n_samples = len(df)

    # Feature Channels:
    # Sequence (4) + Structure (3) + LoopType (7) + PartnerIdentity (4) = 18
    input_channels = 4 + 3 + 7 + 4

    inputs = np.zeros((n_samples, SEQ_LENGTH, input_channels), dtype=np.float32)
    partner_indices = np.full((n_samples, SEQ_LENGTH), -1, dtype=np.int32)
    targets = np.zeros((n_samples, SEQ_LENGTH, 5), dtype=np.float32)
    ids = df["id"].values

    for idx, row in df.iterrows():
        # --- 1. Basic Features ---
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # One-hot encodings
        oh_seq = to_one_hot(seq, TOKEN_TO_INT_SEQ, SEQ_LENGTH, 4)
        oh_struct = to_one_hot(struct, TOKEN_TO_INT_STRUCT, SEQ_LENGTH, 3)
        oh_loop = to_one_hot(loop, TOKEN_TO_INT_LOOP, SEQ_LENGTH, 7)

        # --- 2. Partner Features ---
        pairs = get_pairs(struct)
        oh_partner = np.zeros((SEQ_LENGTH, 4), dtype=np.float32)
        p_indices = np.full(SEQ_LENGTH, -1, dtype=np.int32)

        for i, j in pairs:
            if i < SEQ_LENGTH and j < SEQ_LENGTH:
                # Partner Identity: Base i gets one-hot of Base j
                oh_partner[i] = oh_seq[j]
                oh_partner[j] = oh_seq[i]
                # Partner Indices
                p_indices[i] = j
                p_indices[j] = i

        # Concatenate all features
        # Shape: [Length, 18]
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)

        inputs[idx] = sample_input
        partner_indices[idx] = p_indices

        # --- 3. Targets (Train/Val only) ---
        if mode in ["train", "val"]:
            for t_i, col in enumerate(TARGET_COLS):
                val_str = row[col]
                try:
                    # Parse stringified list: "[0.1, 0.2, ...]"
                    val_list = ast.literal_eval(val_str)
                    length = len(val_list)
                    # Fill available ground truth (usually first 68 positions)
                    # Remaining positions stay 0.0
                    if length > 0:
                        targets[idx, :length, t_i] = val_list
                except Exception:
                    pass  # Keep as zeros if parsing fails

    if mode == "test":
        return inputs, partner_indices, None, ids
    else:
        return inputs, partner_indices, targets, ids


def get_data(mode="train", load_cached_data=True):
    """
    Orchestrates loading, processing, and caching of data.
    """
    # Determine file paths based on mode
    if mode == "train":
        csv_path = TRAIN_PATH
        cache_file = "train_data_hs_gfdn_v1.npz"
    elif mode == "val":
        csv_path = VAL_PATH
        cache_file = "val_data_hs_gfdn_v1.npz"
    else:
        csv_path = TEST_PATH
        cache_file = "test_data_hs_gfdn_v1.npz"

    cache_path = os.path.join(CACHE_DIR, cache_file)

    # --- 1. Try Loading from Cache ---
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            partner_indices = data["partner_indices"]
            ids = data["ids"]

            if mode in ["train", "val"]:
                targets = data["targets"]
                return inputs, partner_indices, targets, ids
            else:
                return inputs, partner_indices, None, ids
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing...")

    # --- 2. Process from Scratch ---
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if DEBUG:
        print(f"DEBUG Mode: Using first {DEBUG_SUBSET_SIZE} samples.")
        df = df.head(DEBUG_SUBSET_SIZE)

    inputs, partner_indices, targets, ids = preprocess_data(df, mode=mode)

    # --- 3. Save to Cache ---
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Saving {mode} data to {cache_path}...")
    save_dict = {"inputs": inputs, "partner_indices": partner_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return inputs, partner_indices, targets, ids


# =============================================================================
# DATASET CLASS
# =============================================================================
class RNADataset(Dataset):
    def __init__(self, inputs, partner_indices, targets=None, ids=None):
        self.inputs = inputs
        self.partner_indices = partner_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (L, Channels)
        # partner_indices: (L,)
        # targets: (L, 5)

        sample = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "partner_indices": torch.tensor(
                self.partner_indices[idx], dtype=torch.long
            ),
        }

        if self.targets is not None:
            sample["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["ids"] = str(self.ids[idx])

        return sample


# =============================================================================
# DATALOADER FACTORY
# =============================================================================
def get_dataloaders(load_cached_data=True):
    """
    Returns train, val, and test dataloaders.
    """
    # Load data arrays
    train_inputs, train_pi, train_targets, train_ids = get_data(
        "train", load_cached_data
    )
    val_inputs, val_pi, val_targets, val_ids = get_data("val", load_cached_data)
    test_inputs, test_pi, _, test_ids = get_data("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_inputs, train_pi, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_pi, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_pi, None, test_ids)

    # Create DataLoaders
    # Pin memory helps with transfer to GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
