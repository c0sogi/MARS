import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    SEQ_LEN,
    SEQ_SCORED,
    BATCH_SIZE,
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_STRUCT,
    VOCAB_SIZE_LOOP,
    TARGET_COLS,
    ID_COL,
    SEQUENCE_COL,
    STRUCTURE_COL,
    LOOP_TYPE_COL,
)
from library.utils import parse_structure_to_indices

# Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.ids = data_dict["ids"]
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_mask = data_dict["pair_mask"]
        self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensors
        inputs = torch.from_numpy(self.inputs[idx]).float()
        pair_indices = torch.from_numpy(self.pair_indices[idx]).long()
        pair_mask = torch.from_numpy(self.pair_mask[idx]).float()
        targets = torch.from_numpy(self.targets[idx]).float()
        sample_id = self.ids[idx]

        return {
            "inputs": inputs,
            "pair_indices": pair_indices,
            "pair_mask": pair_mask,
            "targets": targets,
            "ids": sample_id,
        }


def one_hot_encode(seq, mapping, vocab_size):
    """Encodes a sequence string into a one-hot numpy array."""
    arr = np.zeros((len(seq), vocab_size), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_dataframe(df, is_test=False):
    """
    Process dataframe into numpy arrays for inputs, adjacency, and targets.
    """
    num_samples = len(df)

    # Initialize arrays
    # Input channels: Seq(4) + Struct(3) + Loop(7) = 14
    total_channels = VOCAB_SIZE_SEQ + VOCAB_SIZE_STRUCT + VOCAB_SIZE_LOOP
    inputs = np.zeros((num_samples, SEQ_LEN, total_channels), dtype=np.float32)
    pair_indices = np.zeros((num_samples, SEQ_LEN), dtype=np.int32)
    pair_mask = np.zeros((num_samples, SEQ_LEN), dtype=np.float32)
    targets = np.zeros((num_samples, SEQ_LEN, len(TARGET_COLS)), dtype=np.float32)
    ids = df[ID_COL].values

    print(f"Processing {num_samples} samples...")

    for i, row in df.iterrows():
        # 1. Features
        seq_oh = one_hot_encode(row[SEQUENCE_COL], SEQ_MAP, VOCAB_SIZE_SEQ)
        struct_oh = one_hot_encode(row[STRUCTURE_COL], STRUCT_MAP, VOCAB_SIZE_STRUCT)
        loop_oh = one_hot_encode(row[LOOP_TYPE_COL], LOOP_MAP, VOCAB_SIZE_LOOP)

        # Concatenate features
        inputs[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Adjacency / Structure
        # parse_structure_to_indices returns -1 for unpaired
        p_idx = parse_structure_to_indices(row[STRUCTURE_COL])

        # Handle -1 for gathering: set to 0, but use mask to zero out result
        mask = (p_idx != -1).astype(np.float32)
        p_idx_safe = np.where(p_idx == -1, 0, p_idx)

        pair_indices[i] = p_idx_safe
        pair_mask[i] = mask

        # 3. Targets
        if not is_test:
            for t_idx, col in enumerate(TARGET_COLS):
                # Target is a list of length SEQ_SCORED (68)
                val_list = row[col]
                # Pad to SEQ_LEN (107)
                padded_val = np.zeros(SEQ_LEN, dtype=np.float32)
                padded_val[: len(val_list)] = val_list
                targets[i, :, t_idx] = padded_val
        # For test, targets remain zero

    return {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "pair_mask": pair_mask,
        "targets": targets,
        "ids": ids,
    }


def get_cache_path(source_path):
    """Generate a cache filename based on the source file path hash."""
    file_hash = hashlib.md5(source_path.encode()).hexdigest()
    filename = f"data_cache_{file_hash}.npz"
    return os.path.join(WORKING_DIR, filename)


def load_or_process_data(source_path, load_cached_data=True, is_test=False):
    """
    Load data from cache or process from Parquet.
    """
    cache_path = get_cache_path(source_path)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": loaded["inputs"],
                "pair_indices": loaded["pair_indices"],
                "pair_mask": loaded["pair_mask"],
                "targets": loaded["targets"],
                "ids": loaded["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Loading raw data from {source_path}")
    df = pd.read_parquet(source_path)
    data_dict = preprocess_dataframe(df, is_test=is_test)

    # Save to cache
    print(f"Saving processed data to {cache_path}")
    np.savez(
        cache_path,
        inputs=data_dict["inputs"],
        pair_indices=data_dict["pair_indices"],
        pair_mask=data_dict["pair_mask"],
        targets=data_dict["targets"],
        ids=data_dict["ids"],
    )

    return data_dict


def get_dataloaders(load_cached_data=True, debug=False, debug_subset_size=100):
    """
    Main function to get DataLoaders for Train, Val, and Test.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Load Data
    train_data = load_or_process_data(TRAIN_PATH, load_cached_data, is_test=False)
    val_data = load_or_process_data(VAL_PATH, load_cached_data, is_test=False)
    test_data = load_or_process_data(TEST_PATH, load_cached_data, is_test=True)

    # Handle Debug Mode (Slice the data)
    if debug:
        print(f"DEBUG MODE: Slicing datasets to {debug_subset_size} samples.")
        for d in [train_data, val_data, test_data]:
            limit = min(len(d["ids"]), debug_subset_size)
            d["inputs"] = d["inputs"][:limit]
            d["pair_indices"] = d["pair_indices"][:limit]
            d["pair_mask"] = d["pair_mask"][:limit]
            d["targets"] = d["targets"][:limit]
            d["ids"] = d["ids"][:limit]

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create DataLoaders
    # Note: Shuffle Train, but not Val/Test
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
