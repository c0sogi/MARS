import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_structure_pairs

# One-hot encoding mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Holds features, structure adjacency indices, targets, and masks.
    """

    def __init__(self, features, pair_indices, targets=None, masks=None, ids=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        if masks is not None:
            self.masks = torch.tensor(masks, dtype=torch.float32)
        else:
            self.masks = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "features": self.features[idx],
            "pair_indices": self.pair_indices[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.masks is not None:
            sample["mask"] = self.masks[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def get_one_hot(sequence, mapping, length):
    """
    Converts a string sequence into a one-hot encoded numpy array.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_data(df, config: Config, is_test=False):
    """
    Converts DataFrame columns into numpy arrays for the dataset.
    """
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values
    ids = df["id"].values

    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize Feature Arrays
    # Shape: (N, L, 14) -> 4 (Seq) + 3 (Struct) + 7 (Loop)
    features = np.zeros((num_samples, seq_len, config.input_channels), dtype=np.float32)

    # Initialize Pair Indices
    # Shape: (N, L)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Initialize Targets and Masks
    targets = None
    masks = None

    if not is_test:
        targets = np.zeros((num_samples, seq_len, config.num_targets), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    for i in range(num_samples):
        # 1. Construct Features
        seq_oh = get_one_hot(sequences[i], SEQ_MAP, seq_len)
        struct_oh = get_one_hot(structures[i], STRUCT_MAP, seq_len)
        loop_oh = get_one_hot(loops[i], LOOP_MAP, seq_len)

        features[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Construct Pair Indices
        pairs = parse_structure_pairs(structures[i])
        # Handle length mismatch if any (truncation or padding)
        current_len = len(pairs)
        if current_len < seq_len:
            pad = np.full(seq_len - current_len, -1, dtype=np.int32)
            pairs = np.concatenate([pairs, pad])
        elif current_len > seq_len:
            pairs = pairs[:seq_len]
        pair_indices[i] = pairs

        # 3. Construct Targets & Masks
        if not is_test:
            # Get the number of scored positions for this sample
            # (Usually 68, but reading from column ensures robustness)
            seq_scored = int(df.iloc[i]["seq_scored"])

            for t_idx, col in enumerate(config.target_cols):
                val_list = df.iloc[i][col]
                # val_list is a list or array
                length = len(val_list)
                # Fill the valid portion
                targets[i, :length, t_idx] = val_list

            # Mask is 1.0 for scored positions, 0.0 for unscored/padding
            masks[i, :seq_scored] = 1.0

    return {
        "features": features,
        "pair_indices": pair_indices,
        "targets": targets,
        "masks": masks,
        "ids": ids,
    }


def load_and_cache_data(config: Config, split="train", load_cached_data=True):
    """
    Loads data from Parquet metadata, preprocesses it, and caches the result.
    Uses .npz format to avoid pickling objects.
    """
    # Determine paths based on split
    if split == "train":
        meta_path = config.train_metadata_path
        # Replace .npy with .npz for np.savez compatibility
        cache_path = config.train_cache_path.replace(".npy", ".npz")
        is_test = False
    elif split == "val":
        meta_path = config.val_metadata_path
        cache_path = config.val_cache_path.replace(".npy", ".npz")
        is_test = False
    elif split == "test":
        meta_path = config.test_metadata_path
        cache_path = config.test_cache_path.replace(".npy", ".npz")
        is_test = True
    else:
        raise ValueError(f"Unknown split: {split}")

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached {split} data from {cache_path}...")
            data = np.load(cache_path)
            # Reconstruct dictionary from NpzFile
            data_dict = {
                "features": data["features"],
                "pair_indices": data["pair_indices"],
                "ids": data["ids"],
            }
            if "targets" in data and "masks" in data:
                data_dict["targets"] = data["targets"]
                data_dict["masks"] = data["masks"]
            else:
                data_dict["targets"] = None
                data_dict["masks"] = None

            return RNADataset(**data_dict)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Preprocessing {split} data from {meta_path}...")
    df = pd.read_parquet(meta_path)

    # Debug Subsetting
    if config.debug:
        subset_size = int(len(df) * config.subset_fraction)
        if subset_size < 1:
            subset_size = 1
        print(f"Debug mode: Subsetting {split} to {subset_size} samples.")
        df = df.iloc[:subset_size].reset_index(drop=True)

    data_dict = preprocess_data(df, config, is_test=is_test)

    # Save cache using np.savez (no pickle of arbitrary objects)
    print(f"Saving {split} cache to {cache_path}...")
    save_dict = {k: v for k, v in data_dict.items() if v is not None}
    np.savez(cache_path, **save_dict)

    return RNADataset(**data_dict)


def get_dataloaders(config: Config):
    """
    Generates DataLoaders for Train, Validation, and Test sets.
    """
    train_dataset = load_and_cache_data(config, split="train")
    val_dataset = load_and_cache_data(config, split="val")
    test_dataset = load_and_cache_data(config, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
