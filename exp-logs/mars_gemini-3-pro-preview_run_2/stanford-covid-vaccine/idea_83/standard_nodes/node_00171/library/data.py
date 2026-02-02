import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Dictionaries for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# Inverse map for partner identity lookup
IDX_TO_BASE = {0: "A", 1: "G", 2: "C", 3: "U"}


def get_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array of indices where arr[i] = j if i pairs with j, else -1.
    """
    L = len(structure)
    pairs = np.full(L, -1, dtype=np.int32)
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


def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on the provided mapping.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_data(csv_path, mode="train", load_cached_data=True):
    """
    Loads data from CSV, performs feature engineering (One-Hot, Partner Identity),
    and handles caching to disk.

    Args:
        csv_path (str): Path to the metadata CSV file.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for inputs, partner_indices, targets, and ids.
    """
    # Construct cache filename based on mode and version
    cache_filename = f"{mode}_data_{Config.CACHE_VERSION}.npz"
    cache_path = os.path.join(Config.IDEA_DIR, cache_filename)

    # 1. Try to load from cache
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
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Initialize lists
    all_inputs = []
    all_partner_indices = []
    all_targets = []
    all_ids = df["id"].values

    seq_len = Config.TOTAL_SEQ_LEN
    scored_len = Config.SCORED_SEQ_LEN

    # Pre-compute target column parsing if not test
    target_cols = Config.ALL_TARGET_COLS

    for idx, row in df.iterrows():
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # A. Basic One-Hot Encoding
        # Sequence (4 channels)
        oh_seq = one_hot_encode(sequence, SEQ_MAP, seq_len)
        # Structure (3 channels)
        oh_struct = one_hot_encode(structure, STRUCT_MAP, seq_len)
        # Loop Type (7 channels)
        oh_loop = one_hot_encode(loop_type, LOOP_MAP, seq_len)

        # B. Partner Index & Identity
        # Get pairing map
        partner_idx = get_structure_pairs(structure)

        # Generate Partner Identity Features (4 channels)
        # For each position i, if paired with j, this is one-hot of sequence[j]
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i in range(seq_len):
            pidx = partner_idx[i]
            if pidx != -1:
                # Find the base at the partner index
                partner_base = sequence[pidx]
                if partner_base in SEQ_MAP:
                    oh_partner[i, SEQ_MAP[partner_base]] = 1.0

        # Concatenate all inputs: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18 channels
        sample_input = np.concatenate([oh_seq, oh_struct, oh_loop, oh_partner], axis=1)

        # C. Targets
        # Initialize targets with zeros (shape: 107, 5)
        sample_targets = np.zeros((seq_len, len(target_cols)), dtype=np.float32)

        if mode != "test":
            for c_i, col in enumerate(target_cols):
                # Parse stringified list
                try:
                    val_list = ast.literal_eval(row[col])
                    # Copy available values (usually first 68)
                    # Note: val_list might be shorter than scored_len in rare cases, or match it
                    length_to_copy = min(len(val_list), seq_len)
                    sample_targets[:length_to_copy, c_i] = val_list[:length_to_copy]
                except (ValueError, SyntaxError):
                    # Handle cases where parsing fails (should not happen in clean data)
                    pass

        all_inputs.append(sample_input)
        all_partner_indices.append(partner_idx)
        all_targets.append(sample_targets)

    # Convert to numpy arrays
    inputs_arr = np.array(all_inputs, dtype=np.float32)
    partner_indices_arr = np.array(all_partner_indices, dtype=np.int32)
    targets_arr = np.array(all_targets, dtype=np.float32)
    ids_arr = np.array(all_ids)

    # 3. Save to cache
    print(f"Saving processed {mode} data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        inputs=inputs_arr,
        partner_indices=partner_indices_arr,
        targets=targets_arr,
        ids=ids_arr,
    )

    return {
        "inputs": inputs_arr,
        "partner_indices": partner_indices_arr,
        "targets": targets_arr,
        "ids": ids_arr,
    }


class RNA_Dataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict["inputs"]
        self.partner_indices = data_dict["partner_indices"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Inputs: (Seq_Len, Channels)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Partner Indices: (Seq_Len,)
        # Note: Contains -1 for unpaired. Model handles masking.
        p_idx = torch.tensor(self.partner_indices[idx], dtype=torch.long)

        # Targets: (Seq_Len, 5)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        return x, p_idx, y


def get_loader(
    mode,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates a DataLoader for the specified mode (train/val/test).

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached preprocessed data.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    # Determine CSV path
    if mode == "train":
        csv_path = Config.TRAIN_METADATA
    elif mode == "val":
        csv_path = Config.VAL_METADATA
    elif mode == "test":
        csv_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Process Data
    data_dict = process_data(csv_path, mode=mode, load_cached_data=load_cached_data)

    # Create Dataset
    dataset = RNA_Dataset(data_dict)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
