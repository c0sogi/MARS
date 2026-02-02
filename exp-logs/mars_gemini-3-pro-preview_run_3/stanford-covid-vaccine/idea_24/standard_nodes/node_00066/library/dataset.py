import os
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Dictionaries for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Serves One-Hot encoded features and structural adjacency maps.
    """

    def __init__(self, inputs, adjacency, pairing_mask, targets=None, ids=None):
        self.inputs = torch.from_numpy(inputs).float()
        self.adjacency = torch.from_numpy(adjacency).long()
        self.pairing_mask = torch.from_numpy(pairing_mask).float()

        # Targets are only present for Train/Val
        if targets is not None:
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = None

        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "sequence": self.inputs[idx],
            "adjacency": self.adjacency[idx],
            "mask": self.pairing_mask[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate adjacency indices and a pairing mask.

    Args:
        structure (str): Dot-bracket string (e.g., ".(..).").

    Returns:
        adj (np.array): Array of length L. adj[i] = j if paired with j.
                        If unpaired, adj[i] = 0 (dummy index).
        mask (np.array): Array of length L. 1.0 if paired, 0.0 if unpaired.
    """
    length = len(structure)
    adj = np.zeros(length, dtype=np.int32)
    mask = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Register pair
                adj[i] = j
                adj[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
            else:
                # Unbalanced closing bracket, treat as unpaired
                pass

    # Unpaired positions ('.') remain 0 in adj and 0.0 in mask.
    # Note: adj[i]=0 for unpaired is safe because mask[i]=0.0 will zero out the gathered vector.
    # This satisfies the "Zero-Masking" requirement.
    return adj, mask


def one_hot_encode(seq_str, struct_str, loop_str):
    """
    Generates a concatenated one-hot encoding for sequence, structure, and loop type.
    Output shape: (L, 14)
    """
    length = len(seq_str)
    encoding = np.zeros((length, Config.INPUT_CHANNELS), dtype=np.float32)

    for i in range(length):
        # Sequence (0-3)
        s_char = seq_str[i]
        if s_char in SEQ_MAP:
            encoding[i, SEQ_MAP[s_char]] = 1.0

        # Structure (4-6)
        st_char = struct_str[i]
        if st_char in STRUCT_MAP:
            encoding[i, Config.DIM_SEQ + STRUCT_MAP[st_char]] = 1.0

        # Loop Type (7-13)
        l_char = loop_str[i]
        if l_char in LOOP_MAP:
            encoding[i, Config.DIM_SEQ + Config.DIM_STRUCT + LOOP_MAP[l_char]] = 1.0

    return encoding


def preprocess_data(df, mode="train"):
    """
    Converts a DataFrame into numpy arrays suitable for the model.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize containers
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    adjacency = np.zeros((num_samples, seq_len), dtype=np.int32)
    pairing_mask = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Targets container (only for train/val)
    targets = None
    if mode in ["train", "val"]:
        # Targets are lists of length SEQ_SCORED (68)
        # We store them as (N, 68, 5)
        targets = np.zeros(
            (num_samples, Config.SEQ_SCORED, Config.NUM_TARGETS), dtype=np.float32
        )

    print(f"Processing {mode} data: {num_samples} samples...")

    for idx, row in df.iterrows():
        # 1. Input Features
        inputs[idx] = one_hot_encode(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )

        # 2. Structural Adjacency
        adj, mask = get_structure_adj(row["structure"])
        adjacency[idx] = adj
        pairing_mask[idx] = mask

        # 3. Targets
        if mode in ["train", "val"]:
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_react = row["reactivity"]
            t_mg_ph10 = row["deg_Mg_pH10"]
            t_ph10 = row["deg_pH10"]
            t_mg_50c = row["deg_Mg_50C"]
            t_50c = row["deg_50C"]

            # Stack into (68, 5)
            # Ensure we take exactly SEQ_SCORED elements
            targets[idx, :, 0] = t_react[: Config.SEQ_SCORED]
            targets[idx, :, 1] = t_mg_ph10[: Config.SEQ_SCORED]
            targets[idx, :, 2] = t_ph10[: Config.SEQ_SCORED]
            targets[idx, :, 3] = t_mg_50c[: Config.SEQ_SCORED]
            targets[idx, :, 4] = t_50c[: Config.SEQ_SCORED]

    return inputs, adjacency, pairing_mask, targets, ids


def get_loader(
    mode,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    num_workers=Config.NUM_WORKERS,
    shuffle=None,
):
    """
    Factory function to create DataLoaders. Handles caching and data loading.

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to try loading from cache.
        num_workers (int): Number of workers for DataLoader.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False otherwise.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Determine paths
    if mode == "train":
        file_path = Config.TRAIN_PATH
    elif mode == "val":
        file_path = Config.VAL_PATH
    elif mode == "test":
        file_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Generate Cache Key
    # We use a simple hash of the Config class string representation + mode to ensure consistency
    # If Config changes (e.g. dimensions), the hash should ideally change, but here we rely on mode.
    # To be robust, we just use the mode and assume Config doesn't change mid-run.
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    data_loaded = False
    inputs, adjacency, pairing_mask, targets, ids = None, None, None, None, None

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached {mode} data from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            inputs = data["inputs"]
            adjacency = data["adjacency"]
            pairing_mask = data["pairing_mask"]
            ids = data["ids"]

            if mode in ["train", "val"]:
                targets = data["targets"]

            data_loaded = True
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute if not loaded
    if not data_loaded:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        print(f"Loading raw data from {file_path}...")
        df = pd.read_parquet(file_path)

        inputs, adjacency, pairing_mask, targets, ids = preprocess_data(df, mode)

        # Save to cache
        print(f"Saving {mode} data to cache {cache_file}...")
        save_dict = {
            "inputs": inputs,
            "adjacency": adjacency,
            "pairing_mask": pairing_mask,
            "ids": ids,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_file, **save_dict)

    # Create Dataset
    dataset = RNADataset(inputs, adjacency, pairing_mask, targets, ids)

    # Determine shuffle
    if shuffle is None:
        shuffle = mode == "train"

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=(
            mode == "train"
        ),  # Drop last incomplete batch during training for stability
    )

    return loader
