import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Token mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_to_distance(structure_str):
    """
    Parses a dot-bracket structure string and returns a vector of signed pairing distances.
    If base i is paired with base j, the value at index i is (j - i).
    Unpaired bases have a value of 0.
    """
    n = len(structure_str)
    # Initialize with 0
    dist_vector = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = i
                open_idx = stack.pop()
                # Signed distance
                dist_vector[open_idx] = j - open_idx
                dist_vector[j] = open_idx - j
            else:
                # Unbalanced closing bracket (should not happen in valid data, but safe to ignore)
                pass

    return dist_vector


def tokenize_sequence(seq_str, token_map):
    """Tokenizes a string sequence based on a mapping dictionary."""
    return np.array([token_map.get(c, 0) for c in seq_str], dtype=np.int64)


def process_dataframe(df, mode="train"):
    """
    Extracts features and targets from a dataframe.
    Returns a dictionary of numpy arrays.
    """
    # 1. Sequences
    sequences = df[Config.SEQUENCE_COL].values
    seq_tokens = np.array([tokenize_sequence(s, SEQ_MAP) for s in sequences])

    # 2. Loop Types
    loops = df[Config.LOOP_TYPE_COL].values
    loop_tokens = np.array([tokenize_sequence(s, LOOP_MAP) for s in loops])

    # 3. Structure / Pairing Distance
    structures = df[Config.STRUCTURE_COL].values
    pair_dists = np.array([parse_structure_to_distance(s) for s in structures])

    # 4. Targets (only for train/val)
    # We need to handle the fact that targets are lists of length 68, but we want
    # to pad them to 107 to match sequence length for easier batching.
    targets_padded = None

    if mode in ["train", "val"]:
        # Initialize (N, 107, 3)
        n_samples = len(df)
        targets_padded = np.zeros(
            (n_samples, Config.SEQ_LENGTH, len(Config.TARGET_COLS)), dtype=np.float32
        )

        # Extract target columns
        # Each column in the parquet dataframe is a list/array of length 68
        for i, col_name in enumerate(Config.TARGET_COLS):
            # Convert column of lists to 2D numpy array
            # Note: df[col_name] is a Series of lists.
            # np.vstack works if lengths are consistent.
            col_data = np.vstack(df[col_name].values)  # Shape (N, 68)

            # Assign to the first 68 positions
            targets_padded[:, : Config.PRED_LENGTH, i] = col_data

    # ids for reference
    ids = df["id"].values

    return {
        "ids": ids,
        "seq_tokens": seq_tokens,
        "loop_tokens": loop_tokens,
        "pair_dists": pair_dists,
        "targets": targets_padded,
    }


def get_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode ('train', 'val', 'test').
    Uses caching to speed up subsequent loads.
    """
    # Determine paths
    if mode == "train":
        parquet_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.CACHE_TRAIN_PATH
    elif mode == "val":
        parquet_path = Config.VAL_METADATA_PATH
        cache_path = Config.CACHE_VAL_PATH
    elif mode == "test":
        parquet_path = Config.TEST_METADATA_PATH
        cache_path = Config.CACHE_TEST_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Reconstruct dictionary
            result = {
                "ids": data["ids"],
                "seq_tokens": data["seq_tokens"],
                "loop_tokens": data["loop_tokens"],
                "pair_dists": data["pair_dists"],
            }
            if "targets" in data and mode != "test":
                result["targets"] = data["targets"]
            return result
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # Process from scratch
    print(f"Processing {mode} data from {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    processed_data = process_dataframe(df, mode=mode)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_path}")
    save_dict = {
        "ids": processed_data["ids"],
        "seq_tokens": processed_data["seq_tokens"],
        "loop_tokens": processed_data["loop_tokens"],
        "pair_dists": processed_data["pair_dists"],
    }
    if processed_data["targets"] is not None:
        save_dict["targets"] = processed_data["targets"]

    np.savez_compressed(cache_path, **save_dict)

    return processed_data


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.ids = data_dict["ids"]
        self.seq_tokens = data_dict["seq_tokens"]
        self.loop_tokens = data_dict["loop_tokens"]
        self.pair_dists = data_dict["pair_dists"]
        self.targets = data_dict.get("targets")
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.seq_tokens[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_tokens[idx], dtype=torch.long)
        dist = torch.tensor(self.pair_dists[idx], dtype=torch.float32)

        sample = {"seq": seq, "loop": loop, "dist": dist, "id": self.ids[idx]}

        # Targets
        if self.mode in ["train", "val"] and self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["target"] = target

        return sample


def collate_fn(batch):
    """
    Custom collate function to stack tensors.
    """
    seqs = torch.stack([item["seq"] for item in batch])
    loops = torch.stack([item["loop"] for item in batch])
    dists = torch.stack([item["dist"] for item in batch])
    ids = [item["id"] for item in batch]

    result = {"seq": seqs, "loop": loops, "dist": dists, "id": ids}

    if "target" in batch[0]:
        targets = torch.stack([item["target"] for item in batch])
        result["target"] = targets

    return result
