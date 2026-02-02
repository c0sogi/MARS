import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SEQ_LENGTH,
    SCORED_SEQ_LENGTH,
    BATCH_SIZE,
    NUM_WORKERS,
    ALL_TARGETS,
)
from library.utils import parse_dot_bracket, parse_list_column

# =============================================================================
# MAPPINGS
# =============================================================================
# Sequence: A, G, U, C
SEQ_MAP = {"A": [1, 0, 0, 0], "G": [0, 1, 0, 0], "U": [0, 0, 1, 0], "C": [0, 0, 0, 1]}

# Structure: (, ), .
STRUCT_MAP = {"(": [1, 0, 0], ")": [0, 1, 0], ".": [0, 0, 1]}

# Loop Type: S, M, I, B, H, E, X
LOOP_MAP = {
    "S": [1, 0, 0, 0, 0, 0, 0],
    "M": [0, 1, 0, 0, 0, 0, 0],
    "I": [0, 0, 1, 0, 0, 0, 0],
    "B": [0, 0, 0, 1, 0, 0, 0],
    "H": [0, 0, 0, 0, 1, 0, 0],
    "E": [0, 0, 0, 0, 0, 1, 0],
    "X": [0, 0, 0, 0, 0, 0, 1],
}


def get_one_hot(seq, mapping, length):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    # Default to zero vector if char not found
    dim = len(next(iter(mapping.values())))
    one_hot = np.zeros((length, dim), dtype=np.float32)

    for i, char in enumerate(seq[:length]):
        if char in mapping:
            one_hot[i] = mapping[char]
    return one_hot


def preprocess_data(csv_path, cache_path, is_test=False, load_cached_data=True):
    """
    Loads raw data from CSV, generates features (including Partner Identity),
    and caches the result to an NPZ file.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": data["inputs"],
                "partner_map": data["partner_map"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    # 2. Load CSV
    print(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 3. Initialize Arrays
    num_samples = len(df)
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner) = 18
    input_dim = 4 + 3 + 7 + 4

    inputs = np.zeros((num_samples, SEQ_LENGTH, input_dim), dtype=np.float32)
    partner_maps = np.zeros((num_samples, SEQ_LENGTH), dtype=np.int32)
    targets = np.zeros((num_samples, SEQ_LENGTH, 5), dtype=np.float32)
    ids = df["id"].values

    # 4. Process Rows
    # We use tqdm for progress tracking manually since we can't print progress bars per instructions,
    # but for data processing usually a simple print is allowed. I'll stick to simple prints.

    for idx, row in df.iterrows():
        # --- Feature Generation ---
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # Basic One-Hot
        oh_seq = get_one_hot(sequence, SEQ_MAP, SEQ_LENGTH)  # (L, 4)
        oh_struct = get_one_hot(structure, STRUCT_MAP, SEQ_LENGTH)  # (L, 3)
        oh_loop = get_one_hot(loop_type, LOOP_MAP, SEQ_LENGTH)  # (L, 7)

        # Partner Map
        p_map = parse_dot_bracket(structure)
        # Pad or truncate p_map to SEQ_LENGTH
        if len(p_map) < SEQ_LENGTH:
            p_map = np.pad(p_map, (0, SEQ_LENGTH - len(p_map)), constant_values=-1)
        else:
            p_map = p_map[:SEQ_LENGTH]

        partner_maps[idx] = p_map

        # Partner Identity Injection
        # We want to gather the sequence identity of the partner.
        # If p_map[i] == -1, partner identity is 0.
        # If p_map[i] == j, partner identity is oh_seq[j].

        # Create an index array for gathering. Replace -1 with 0 temporarily for valid indexing,
        # then mask the result.
        gather_indices = p_map.copy()
        mask_unpaired = gather_indices == -1
        gather_indices[mask_unpaired] = 0  # Point to index 0 temporarily

        # Gather: oh_seq is (L, 4), gather_indices is (L,) -> result (L, 4)
        partner_identity = oh_seq[gather_indices]

        # Apply mask: Zero out vectors where base is unpaired
        partner_identity[mask_unpaired] = 0.0

        # Concatenate all features
        # (L, 4+3+7+4) = (L, 18)
        sample_input = np.concatenate(
            [oh_seq, oh_struct, oh_loop, partner_identity], axis=1
        )
        inputs[idx] = sample_input

        # --- Target Parsing ---
        if not is_test:
            # Targets are stored as stringified lists in columns
            # We need to parse them and pad to SEQ_LENGTH (107)
            # The raw data usually has length 68 for targets.
            for t_i, col_name in enumerate(ALL_TARGETS):
                val_arr = parse_list_column(row[col_name])

                # Copy into the targets array
                # Note: val_arr might be length 68. We place it at the beginning.
                length = min(len(val_arr), SEQ_LENGTH)
                targets[idx, :length, t_i] = val_arr[:length]
                # Remaining positions stay 0.0 as initialized

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path, inputs=inputs, partner_map=partner_maps, targets=targets, ids=ids
    )

    return {
        "inputs": inputs,
        "partner_map": partner_maps,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.inputs = torch.from_numpy(data_dict["inputs"]).float()
        self.partner_map = torch.from_numpy(data_dict["partner_map"]).long()
        self.targets = torch.from_numpy(data_dict["targets"]).float()
        self.ids = data_dict["ids"]
        self.is_test = is_test

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (SeqLen, 18)
        # partner_map: (SeqLen,)
        # targets: (SeqLen, 5)

        sample = {
            "inputs": self.inputs[idx],
            "partner_map": self.partner_map[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }

        return sample


def get_dataloaders(load_cached_data=True):
    """
    Preprocesses data and returns DataLoaders for train, val, and test sets.
    """
    # 1. Process Data
    train_data = preprocess_data(
        TRAIN_CSV, TRAIN_CACHE_PATH, is_test=False, load_cached_data=load_cached_data
    )
    val_data = preprocess_data(
        VAL_CSV, VAL_CACHE_PATH, is_test=False, load_cached_data=load_cached_data
    )
    test_data = preprocess_data(
        TEST_CSV, TEST_CACHE_PATH, is_test=True, load_cached_data=load_cached_data
    )

    # 2. Create Datasets
    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
