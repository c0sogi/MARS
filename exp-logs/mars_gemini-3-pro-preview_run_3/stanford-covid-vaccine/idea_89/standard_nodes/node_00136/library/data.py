import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, data, mode="train"):
        self.mode = mode
        self.features = data["features"]
        self.adjacency_indices = data["adjacency_indices"]
        self.adjacency_mask = data["adjacency_mask"]
        self.ids = data["ids"]

        if mode != "test":
            self.targets = data["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (107, 14)
        features = torch.tensor(self.features[idx], dtype=torch.float32)

        # Adjacency Indices: (107,) LongTensor
        # Points to the paired base index. If unpaired, points to 0 (but masked out).
        adj_indices = torch.tensor(self.adjacency_indices[idx], dtype=torch.long)

        # Adjacency Mask: (107,) FloatTensor
        # 1.0 if paired, 0.0 if unpaired. Used to zero-out messages for unpaired bases.
        adj_mask = torch.tensor(self.adjacency_mask[idx], dtype=torch.float32)

        sample = {
            "inputs": features,
            "adjacency_indices": adj_indices,
            "adjacency_mask": adj_mask,
            "id": self.ids[idx],
        }

        if self.targets is not None:
            # Targets: (107, 5)
            # Padded with zeros from 68 to 107.
            targets = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = targets

        return sample


def get_structure_adj(structure, length):
    """
    Parses dot-bracket structure to get adjacency indices and mask.

    Args:
        structure (str): Dot-bracket structure string.
        length (int): Sequence length.

    Returns:
        indices: (length,) np.int32. indices[i] = j if paired with j, else 0.
        mask: (length,) np.float32. 1.0 if paired, 0.0 if unpaired.
    """
    indices = np.zeros(length, dtype=np.int32)
    mask = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
    return indices, mask


def one_hot_encode(sequences, structures, loop_types, length):
    """
    One-hot encodes sequence, structure, and loop type into a single tensor.

    Channels (Total 14):
    0-3:   A, G, C, U
    4-6:   (, ), .
    7-13:  S, M, I, B, H, E, X
    """
    N = len(sequences)
    encoding = np.zeros((N, length, 14), dtype=np.float32)

    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    for i in range(N):
        seq = sequences[i]
        struct = structures[i]
        loop = loop_types[i]

        for j in range(length):
            # Sequence
            if j < len(seq) and seq[j] in seq_map:
                encoding[i, j, seq_map[seq[j]]] = 1.0

            # Structure
            if j < len(struct) and struct[j] in struct_map:
                encoding[i, j, 4 + struct_map[struct[j]]] = 1.0

            # Loop Type
            if j < len(loop) and loop[j] in loop_map:
                encoding[i, j, 7 + loop_map[loop[j]]] = 1.0

    return encoding


def process_data(df, mode="train"):
    """
    Process dataframe into numpy arrays for the dataset.
    """
    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loop_types = df["predicted_loop_type"].values

    N = len(df)
    L = Config.SEQ_LENGTH

    # 1. Features (One-Hot Encoding)
    features = one_hot_encode(sequences, structures, loop_types, L)

    # 2. Adjacency Maps
    adj_indices = np.zeros((N, L), dtype=np.int32)
    adj_mask = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        idx, msk = get_structure_adj(structures[i], L)
        adj_indices[i] = idx
        adj_mask[i] = msk

    data = {
        "features": features,
        "adjacency_indices": adj_indices,
        "adjacency_mask": adj_mask,
        "ids": ids,
    }

    # 3. Targets (Train/Val only)
    if mode != "test":
        targets = np.zeros((N, L, 5), dtype=np.float32)
        target_cols = Config.TARGET_COLS

        for i, col in enumerate(target_cols):
            # df[col] contains lists of length seq_scored (68)
            values = df[col].values
            for j, val_list in enumerate(values):
                # Ensure we handle list/array properly
                if hasattr(val_list, "__len__"):
                    length_scored = len(val_list)
                    targets[j, :length_scored, i] = val_list

        data["targets"] = targets

    return data


def get_data(mode="train", load_cached_data=True):
    """
    Loads data from cache or processes from raw parquet files.
    """
    # Determine paths
    if mode == "train":
        raw_path = Config.TRAIN_DATA_PATH
        cache_path = Config.TRAIN_CACHE_PATH
    elif mode == "val":
        raw_path = Config.VAL_DATA_PATH
        cache_path = Config.VAL_CACHE_PATH
    elif mode == "test":
        raw_path = Config.TEST_DATA_PATH
        cache_path = Config.TEST_CACHE_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process raw data
    print(f"Processing {mode} data from {raw_path}...")
    df = pd.read_parquet(raw_path)

    data = process_data(df, mode=mode)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, data)
    print(f"Saved {mode} data cache to {cache_path}.")

    return data


def get_dataloaders():
    """
    Creates DataLoaders for train, val, and test sets.
    """
    set_seed(Config.SEED)

    # Load data
    train_data = get_data("train", Config.LOAD_CACHED_DATA)
    val_data = get_data("val", Config.LOAD_CACHED_DATA)
    test_data = get_data("test", Config.LOAD_CACHED_DATA)

    # Debug mode subsetting
    if Config.DEBUG:
        print(f"DEBUG mode: Subsetting to {Config.DEBUG_SUBSET_SIZE} samples.")
        for d in [train_data, val_data, test_data]:
            for k in d.keys():
                d[k] = d[k][: Config.DEBUG_SUBSET_SIZE]

    # Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
