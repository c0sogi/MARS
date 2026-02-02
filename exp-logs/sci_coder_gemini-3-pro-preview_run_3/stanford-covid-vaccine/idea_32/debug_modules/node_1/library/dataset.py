import os
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_couples


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Returns:
        features (torch.Tensor): (Seq_Len, 14) - One-hot encoded features.
        indices (torch.Tensor): (Seq_Len,) - Neighbor indices for gathering.
        mask (torch.Tensor): (Seq_Len,) - 1.0 if paired, 0.0 if unpaired.
        targets (torch.Tensor): (Seq_Len, 5) - Ground truth values (padded).
    """

    def __init__(self, features, indices, masks, targets=None, ids=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        # Targets might be None for test set
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "features": self.features[idx],
            "indices": self.indices[idx],
            "mask": self.masks[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def _get_config_hash():
    """Generates a hash based on relevant Config parameters to ensure cache validity."""
    relevant_params = [
        Config.SEQ_LEN,
        Config.SEQ_SCORED,
        Config.INPUT_DIM,
        str(Config.TARGET_COLS),
        Config.IDEA_NAME,
    ]
    param_str = "_".join(map(str, relevant_params))
    return hashlib.md5(param_str.encode()).hexdigest()


def _one_hot_encode(seq, mapping):
    """Helper to one-hot encode a sequence string based on a mapping dictionary."""
    # Create array of shape (len(seq), len(mapping))
    arr = np.zeros((len(seq), len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_dataframe(df, mode="train"):
    """
    Process a dataframe into numpy arrays for features, adjacency, and targets.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Features: (N, L, 14)
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    # Neighbor Indices: (N, L)
    neighbor_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    # Pair Masks: (N, L)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    # Targets: (N, L, 5) - Only for train/val
    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].values

    # Iterate and process
    for i, row in df.iterrows():
        # 1. Features
        s_seq = row["sequence"]
        s_struct = row["structure"]
        s_loop = row["predicted_loop_type"]

        # Ensure lengths match Config.SEQ_LEN (107)
        # (Assuming data is clean as per metadata description, but good to be safe)
        current_len = len(s_seq)

        # One-hot encoding
        oh_seq = _one_hot_encode(s_seq, seq_map)
        oh_struct = _one_hot_encode(s_struct, struct_map)
        oh_loop = _one_hot_encode(s_loop, loop_map)

        # Concatenate: (L, 4) + (L, 3) + (L, 7) -> (L, 14)
        features[i, :current_len, :] = np.concatenate(
            [oh_seq, oh_struct, oh_loop], axis=1
        )

        # 2. Adjacency / Couples
        # get_couples returns -1 for unpaired
        couples = get_couples(s_struct)

        # For the model, we need valid indices for gather.
        # We set unpaired (-1) to 0, but set the mask to 0 so the value is ignored.
        # We set paired to their actual index, and mask to 1.
        is_paired = couples != -1

        # Safe indices: replace -1 with 0
        safe_indices = couples.copy()
        safe_indices[~is_paired] = 0

        neighbor_indices[i, :current_len] = safe_indices
        pair_masks[i, :current_len] = is_paired.astype(np.float32)

        # 3. Targets (Train/Val only)
        if targets is not None:
            # Targets are lists of length 68 (Config.SEQ_SCORED)
            # We pad them to 107 with zeros.
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Parquet loads lists/arrays directly
                if isinstance(val_list, (list, np.ndarray)):
                    val_len = len(val_list)
                    targets[i, :val_len, t_idx] = val_list
                else:
                    # Fallback for unexpected format, though metadata guarantees arrays
                    pass

    return features, neighbor_indices, pair_masks, targets, ids


def get_loaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders. Handles caching and processing.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        train_loader, val_loader, test_loader
    """
    config_hash = _get_config_hash()
    cache_dir = Config.CACHE_DIR

    # Define cache paths
    cache_paths = {
        "train": os.path.join(cache_dir, f"train_data_{config_hash}.npz"),
        "val": os.path.join(cache_dir, f"val_data_{config_hash}.npz"),
        "test": os.path.join(cache_dir, f"test_data_{config_hash}.npz"),
    }

    datasets = {}

    modes = ["train", "val", "test"]
    source_paths = {
        "train": Config.TRAIN_DATA_PATH,
        "val": Config.VAL_DATA_PATH,
        "test": Config.TEST_DATA_PATH,
    }

    for mode in modes:
        cache_path = cache_paths[mode]

        # Try loading from cache
        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                if Config.VERBOSE:
                    print(f"Loading {mode} data from cache: {cache_path}")
                data = np.load(cache_path, allow_pickle=True)
                features = data["features"]
                indices = data["indices"]
                masks = data["masks"]
                ids = data["ids"]
                # Targets only exist for train/val
                targets = data["targets"] if "targets" in data else None
                if mode == "test":
                    targets = None  # Explicitly None for test

                loaded = True
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                loaded = False

        # Process from scratch if not loaded
        if not loaded:
            if Config.VERBOSE:
                print(f"Processing {mode} data from source: {source_paths[mode]}")

            df = pd.read_parquet(source_paths[mode])
            features, indices, masks, targets, ids = process_dataframe(df, mode=mode)

            # Save to cache
            save_dict = {
                "features": features,
                "indices": indices,
                "masks": masks,
                "ids": ids,
            }
            if targets is not None:
                save_dict["targets"] = targets

            np.savez_compressed(cache_path, **save_dict)
            if Config.VERBOSE:
                print(f"Saved {mode} data to cache: {cache_path}")

        # Create Dataset
        datasets[mode] = RNADataset(features, indices, masks, targets, ids)

    # Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last to maintain batch statistics
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
