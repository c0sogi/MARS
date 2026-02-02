import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Mappings
TOKEN_MAP = {
    "sequence": {"A": 0, "G": 1, "C": 2, "U": 3},
    "structure": {"(": 0, ")": 1, ".": 2},
    "predicted_loop_type": {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6},
}


def get_structure_adj(structure_str, seq_len):
    """
    Parses dot-bracket structure to get pair indices and mask.
    Returns:
        pair_indices: (seq_len,) array where arr[i] = j if paired with j, else 0
        pair_mask: (seq_len,) array where arr[i] = 1 if paired, else 0
    """
    pair_indices = np.zeros(seq_len, dtype=np.int32)  # Default point to 0
    pair_mask = np.zeros(seq_len, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0

    # Note: Unpaired positions in pair_indices remain 0.
    # The pair_mask ensures these 0-pointers are ignored during message passing.
    return pair_indices, pair_mask


def one_hot_encode(seq, token_map, length):
    """
    One-hot encodes a sequence string based on the provided map.
    """
    encoding = np.zeros((length, len(token_map)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in token_map:
            encoding[i, token_map[char]] = 1.0
    return encoding


def process_dataframe(df, is_test=False):
    """
    Processes a dataframe into numpy arrays for features, adjacency, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM

    # Initialize arrays
    # Features: (N, L, 14)
    features = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)

    # Adjacency: (N, L)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: (N, L, 5) - padded with zeros
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # IDs
    ids = df["id"].values

    for idx, row in df.iterrows():
        # 1. Features
        seq_enc = one_hot_encode(row["sequence"], TOKEN_MAP["sequence"], seq_len)
        struct_enc = one_hot_encode(row["structure"], TOKEN_MAP["structure"], seq_len)
        loop_enc = one_hot_encode(
            row["predicted_loop_type"], TOKEN_MAP["predicted_loop_type"], seq_len
        )

        # Concatenate features: Sequence (4) + Structure (3) + Loop (7)
        features[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 2. Adjacency Map
        p_idx, p_mask = get_structure_adj(row["structure"], seq_len)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets (if not test)
        if not is_test:
            # Targets are provided as lists/arrays of length seq_scored (68)
            # We map them to the first 68 positions of the (107, 5) tensor
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure we handle cases where val_list might be shorter/longer or numpy array
                length = len(val_list)
                targets[idx, :length, t_i] = val_list

    return {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


def load_or_process_data(
    parquet_path, cache_path, load_cached_data=True, is_test=False
):
    """
    Loads data from cache if available, otherwise processes from Parquet and caches it.
    """
    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            data = np.load(cache_path, allow_pickle=True)
            return {
                "features": data["features"],
                "pair_indices": data["pair_indices"],
                "pair_masks": data["pair_masks"],
                "targets": data["targets"],
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    processed_data = process_dataframe(df, is_test=is_test)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        features=processed_data["features"],
        pair_indices=processed_data["pair_indices"],
        pair_masks=processed_data["pair_masks"],
        targets=processed_data["targets"],
        ids=processed_data["ids"],
    )
    print(f"Data cached to {cache_path}.")

    return processed_data


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.features = torch.tensor(data_dict["features"], dtype=torch.float32)
        self.pair_indices = torch.tensor(data_dict["pair_indices"], dtype=torch.long)
        self.pair_masks = torch.tensor(data_dict["pair_masks"], dtype=torch.float32)
        self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        self.ids = data_dict["ids"]
        self.is_test = is_test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "features": self.features[idx],  # (107, 14)
            "pair_indices": self.pair_indices[idx],  # (107,)
            "pair_masks": self.pair_masks[idx],  # (107,)
            "id": self.ids[idx],
        }

        if not self.is_test:
            sample["targets"] = self.targets[idx]  # (107, 5)

        return sample


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    """
    # 1. Train
    train_data = load_or_process_data(
        Config.TRAIN_PATH,
        Config.TRAIN_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )
    train_dataset = RNADataset(train_data, is_test=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Val
    val_data = load_or_process_data(
        Config.VAL_PATH,
        Config.VAL_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )
    val_dataset = RNADataset(val_data, is_test=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test
    test_data = load_or_process_data(
        Config.TEST_PATH,
        Config.TEST_CACHE,
        load_cached_data=load_cached_data,
        is_test=True,
    )
    test_dataset = RNADataset(test_data, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
