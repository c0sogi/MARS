import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, sequences, structures, loops, targets=None, ids=None):
        self.sequences = sequences
        self.structures = structures
        self.loops = loops
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        struct = torch.tensor(self.structures[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)

        item = {
            "sequence": seq,
            "structure": struct,
            "loop_type": loop,
        }

        if self.ids is not None:
            item["id"] = self.ids[idx]

        if self.targets is not None:
            # Targets are float
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = target

        return item


def tokenization(texts, mapping, length):
    """
    Converts a list of strings to a numpy array of integers based on mapping.
    """
    # Vectorized approach or list comprehension.
    # Given the dataset size (~2400), list comprehension is fast enough.
    # We assume all strings are length 107 based on dataset info.

    # Create a translation table or simple lookup
    # Using list comprehension for clarity and speed on small data
    encoded = []
    for text in texts:
        # Map chars to ints, default to 0 if unknown (though data should be clean)
        vec = [mapping.get(c, 0) for c in text]
        encoded.append(vec)

    return np.array(encoded, dtype=np.int32)


def process_dataframe(df, mode="train"):
    """
    Extracts features and targets from the dataframe.
    """
    # 1. Extract Inputs
    sequences = tokenization(df["sequence"].values, Config.SEQ_MAP, Config.SEQ_LEN)
    structures = tokenization(df["structure"].values, Config.STRUCT_MAP, Config.SEQ_LEN)
    loops = tokenization(
        df["predicted_loop_type"].values, Config.LOOP_MAP, Config.SEQ_LEN
    )
    ids = df["id"].values

    # 2. Extract Targets (only for train/val)
    targets = None
    if mode in ["train", "val"]:
        # The target columns are lists in the parquet file.
        # We need to stack them: (N, 5, 68) -> Transpose to (N, 68, 5) usually preferred for Linear layers
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        # Extract each column which is a Series of lists
        target_arrays = []
        for col in Config.TARGET_COLS:
            # Convert Series of lists to 2D numpy array
            # np.vstack works if all lists are same length (they are 68)
            arr = np.vstack(df[col].values)
            target_arrays.append(arr)

        # Stack along the last dimension: Result shape (N, 68, 5)
        # target_arrays is a list of 5 arrays of shape (N, 68)
        # dstack stacks along the 3rd axis (depth)
        targets = np.dstack(target_arrays)

    return sequences, structures, loops, targets, ids


def get_data(split, load_cached_data=True, debug=False):
    """
    Loads data from cache or raw parquet files.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data.npz")

    # Determine source file
    if split == "train":
        source_file = Config.TRAIN_FILE
    elif split == "val":
        source_file = Config.VAL_FILE
    elif split == "test":
        source_file = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            sequences = data["sequences"]
            structures = data["structures"]
            loops = data["loops"]
            ids = data["ids"]

            # Targets might be None for test set, np.savez handles None as object or omits
            if "targets" in data:
                targets = data["targets"]
            else:
                targets = None

            # If debug, slice the data
            if debug:
                limit = Config.DEBUG_SAMPLES
                sequences = sequences[:limit]
                structures = structures[:limit]
                loops = loops[:limit]
                ids = ids[:limit]
                if targets is not None:
                    targets = targets[:limit]

            return RNADataset(sequences, structures, loops, targets, ids)
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Re-processing...")

    # Process from scratch
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")

    df = pd.read_parquet(source_file)

    # Pre-slice for debug to save processing time if debug is ON and we are re-processing
    # However, for caching purposes, we usually want to cache the FULL dataset.
    # So we process full, save full, then slice if debug.

    sequences, structures, loops, targets, ids = process_dataframe(df, mode=split)

    # Save to cache
    save_dict = {
        "sequences": sequences,
        "structures": structures,
        "loops": loops,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    # Slice if debug
    if debug:
        limit = Config.DEBUG_SAMPLES
        sequences = sequences[:limit]
        structures = structures[:limit]
        loops = loops[:limit]
        ids = ids[:limit]
        if targets is not None:
            targets = targets[:limit]

    return RNADataset(sequences, structures, loops, targets, ids)


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): Whether to use a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """

    # Load Datasets
    train_dataset = get_data("train", load_cached_data, debug)
    val_dataset = get_data("val", load_cached_data, debug)
    test_dataset = get_data("test", load_cached_data, debug)

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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
