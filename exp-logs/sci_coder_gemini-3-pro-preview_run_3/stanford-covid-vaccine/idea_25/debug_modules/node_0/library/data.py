import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Encoding Dictionaries
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_bpp_indices(structure):
    """
    Parses a dot-bracket structure string to generate a mapping of paired indices.
    Returns an array where arr[i] is the index of the base paired with i, or -1 if unpaired.
    """
    length = len(structure)
    bpp = np.full(length, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                bpp[i] = j
                bpp[j] = i
    return bpp


def one_hot_encode(seq, mapping, num_classes):
    """
    One-hot encodes a sequence string based on the provided mapping.
    """
    arr = np.array([mapping.get(c, 0) for c in seq])
    return np.eye(num_classes)[arr]


class RNADataset(Dataset):
    def __init__(self, inputs, bpp_indices, targets=None, ids=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.bpp_indices = torch.tensor(bpp_indices, dtype=torch.long)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "bpp_indices": self.bpp_indices[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        return sample


def process_dataframe(df, config, is_test=False):
    """
    Processes a dataframe into numpy arrays for inputs, bpp, and targets.
    """
    num_samples = len(df)
    seq_len = config.seq_len
    input_dim = config.input_dim  # 14

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    ids = df["id"].values

    # Process features
    for idx, row in df.iterrows():
        # 1. Sequence (4 channels)
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, 4)

        # 2. Structure (3 channels)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, 3)

        # 3. Loop Type (7 channels)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate: (L, 4) + (L, 3) + (L, 7) -> (L, 14)
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. BPP Indices
        bpp_indices[idx] = get_bpp_indices(row["structure"])

    # Process targets if not test set
    targets = None
    if not is_test:
        # Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
        # Original data has lists of length seq_scored (68). We pad to seq_len (107).
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

        for idx, row in df.iterrows():
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                # Fill the scored positions
                length = len(val_list)
                targets[idx, :length, t_i] = val_list
                # Remaining positions are already 0.0 initialization

    return inputs, bpp_indices, targets, ids


def get_dataloaders(config, load_cached_data=True):
    """
    Loads data, processes it (with caching), and returns PyTorch DataLoaders.

    Args:
        config: Config object containing paths and settings.
        load_cached_data: If True, attempts to load preprocessed .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(config.seed)

    # Cache file paths
    cache_dir = config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_cache.npz")
    val_cache = os.path.join(cache_dir, "val_cache.npz")
    test_cache = os.path.join(cache_dir, "test_cache.npz")

    # --- Helper to load or process ---
    def get_data_split(name, parquet_path, cache_path, is_test=False):
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {name} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            inputs = data["inputs"]
            bpp = data["bpp"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None
            return inputs, bpp, targets, ids

        print(f"Processing {name} data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # Debug mode: subset data
        if config.debug:
            df = df.iloc[: config.debug_subset_size].copy().reset_index(drop=True)

        inputs, bpp, targets, ids = process_dataframe(df, config, is_test=is_test)

        # Save to cache
        save_dict = {"inputs": inputs, "bpp": bpp, "ids": ids}
        if targets is not None:
            save_dict["targets"] = targets
        np.savez_compressed(cache_path, **save_dict)

        return inputs, bpp, targets, ids

    # --- Load Data ---
    train_inputs, train_bpp, train_targets, train_ids = get_data_split(
        "train", config.train_data_path, train_cache, is_test=False
    )

    val_inputs, val_bpp, val_targets, val_ids = get_data_split(
        "val", config.val_data_path, val_cache, is_test=False
    )

    test_inputs, test_bpp, test_targets, test_ids = get_data_split(
        "test", config.test_data_path, test_cache, is_test=True
    )

    # --- Create Datasets ---
    train_dataset = RNADataset(train_inputs, train_bpp, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_bpp, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_bpp, None, test_ids)

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
