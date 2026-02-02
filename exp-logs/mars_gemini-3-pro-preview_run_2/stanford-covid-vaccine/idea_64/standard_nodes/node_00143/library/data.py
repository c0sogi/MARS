import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_one_hot(sequence, mapping, length):
    """
    Converts a sequence string into a one-hot encoded numpy array.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def parse_structure_pairs(structure):
    """
    Parses a dot-bracket structure string to find pairing partners.
    Returns:
        pair_indices: np.array of shape (L,), where arr[i] = j if i pairs with j,
                      and -1 if i is unpaired.
    """
    L = len(structure)
    pair_indices = np.full(L, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i

    return pair_indices


def preprocess_dataframe(df, mode="train"):
    """
    Extracts features and targets from the dataframe.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Feature containers
    # Channels: Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
    input_features = np.zeros((num_samples, seq_len, 18), dtype=np.float32)
    pair_indices_all = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Target containers
    # 5 target columns
    targets_all = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    masks_all = np.zeros((num_samples, seq_len), dtype=np.bool_)

    ids_all = []

    target_cols = Config.TARGET_COLS

    for idx, row in df.iterrows():
        # 1. Basic Sequence Information
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # 2. Generate One-Hot Features
        oh_seq = get_one_hot(sequence, SEQ_MAP, seq_len)  # (L, 4)
        oh_struct = get_one_hot(structure, STRUCT_MAP, seq_len)  # (L, 3)
        oh_loop = get_one_hot(loop_type, LOOP_MAP, seq_len)  # (L, 7)

        # 3. Partner Parsing & Partner Identity
        pair_map = parse_structure_pairs(structure)  # (L,) with indices or -1
        pair_indices_all[idx] = pair_map

        # Generate Partner Identity (One-Hot of the paired base)
        oh_partner = np.zeros((seq_len, 4), dtype=np.float32)
        for i in range(len(sequence)):
            partner_idx = pair_map[i]
            if partner_idx != -1:
                # If paired, copy the one-hot vector of the partner
                oh_partner[i] = oh_seq[partner_idx]
            # Else remains zero

        # Concatenate all features
        # Shape: (L, 4+3+7+4) = (L, 18)
        sample_features = np.concatenate(
            [oh_seq, oh_struct, oh_loop, oh_partner], axis=1
        )
        input_features[idx] = sample_features

        # 4. Targets & Mask
        # Only process targets if they exist (Train/Val)
        # Test set might not have targets or they might be dummy

        # Determine valid scored length (usually 68)
        seq_scored = row.get("seq_scored", Config.SEQ_SCORED)

        # Create mask (1 for scored positions, 0 for padding/unscored)
        # We strictly mask 0-67 based on competition rules
        masks_all[idx, :seq_scored] = True

        if mode in ["train", "val"]:
            for t_i, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    # Parse string "[0.1, ...]" -> list
                    val_list = ast.literal_eval(val_str)
                    # Assign to the first 68 positions
                    length_to_assign = min(len(val_list), seq_len)
                    targets_all[idx, :length_to_assign, t_i] = val_list[
                        :length_to_assign
                    ]
                except Exception:
                    # Fallback for parsing errors or missing data
                    pass

        ids_all.append(row["id"])

    return {
        "inputs": input_features,
        "targets": targets_all,
        "masks": masks_all,
        "pair_indices": pair_indices_all,
        "ids": np.array(ids_all),
    }


def load_or_process_data(csv_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes CSV and saves to cache.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "inputs": data["inputs"],
                "targets": data["targets"],
                "masks": data["masks"],
                "pair_indices": data["pair_indices"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from CSV
    print(f"Processing {mode} data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    processed_data = preprocess_dataframe(df, mode=mode)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        inputs=processed_data["inputs"],
        targets=processed_data["targets"],
        masks=processed_data["masks"],
        pair_indices=processed_data["pair_indices"],
        ids=processed_data["ids"],
    )
    print(f"Saved processed data to {cache_path}")

    return processed_data


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.tensor(data_dict["inputs"], dtype=torch.float32)
        self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        self.masks = torch.tensor(data_dict["masks"], dtype=torch.bool)
        self.pair_indices = torch.tensor(data_dict["pair_indices"], dtype=torch.long)
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": self.inputs[idx],  # (L, 18)
            "targets": self.targets[idx],  # (L, 5)
            "mask": self.masks[idx],  # (L,)
            "pair_indices": self.pair_indices[idx],  # (L,)
            "id": self.ids[idx],  # str
        }


def get_loaders(load_cached_data=True, batch_size=None):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Paths from Config
    train_csv = Config.TRAIN_CSV
    val_csv = Config.VAL_CSV
    test_csv = Config.TEST_CSV

    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # Load Data
    train_data = load_or_process_data(
        train_csv, train_cache, mode="train", load_cached_data=load_cached_data
    )
    val_data = load_or_process_data(
        val_csv, val_cache, mode="val", load_cached_data=load_cached_data
    )
    test_data = load_or_process_data(
        test_csv, test_cache, mode="test", load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create Loaders
    # Note: Shuffle Train, but not Val/Test
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop incomplete batch to maintain statistics stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
