import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Returns:
        features: (Seq_Len, 14) - One-hot encoded sequence, structure, and loop type.
        pair_indices: (Seq_Len,) - Indices of paired bases (or self if unpaired).
        pair_masks: (Seq_Len, 1) - Mask indicating if a base is paired (1.0) or not (0.0).
        targets: (Seq_Len, 5) - Ground truth values (padded to Seq_Len), only for train/val.
        ids: (str) - Sample identifier.
    """

    def __init__(self, data_dict, mode="train"):
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        sample = {
            "features": torch.from_numpy(self.features[idx]).float(),
            "pair_indices": torch.from_numpy(self.pair_indices[idx]).long(),
            "pair_masks": torch.from_numpy(self.pair_masks[idx]).float().unsqueeze(-1),
            "ids": self.ids[idx],
        }

        if self.mode != "test":
            sample["targets"] = torch.from_numpy(self.targets[idx]).float()

        return sample


def get_structure_adj(structure):
    """
    Parses dot-bracket structure string to find base pairs.

    Returns:
        pair_index: Array where pair_index[i] is the index of the base paired with i.
                    If i is unpaired, pair_index[i] = i (to allow valid gathering).
        mask: Array where mask[i] = 1.0 if paired, 0.0 if unpaired.
    """
    L = len(structure)
    pair_index = np.arange(L)  # Default to self-connection for unpaired
    mask = np.zeros(L, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_index[i] = j
                pair_index[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
    return pair_index, mask


def process_dataframe(df, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays for the dataset.
    """
    # Mappings for One-Hot Encoding
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    N = len(df)
    L = Config.SEQ_LEN

    # Pre-allocate feature arrays
    features = np.zeros((N, L, 14), dtype=np.float32)
    pair_indices = np.zeros((N, L), dtype=np.int64)
    pair_masks = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        seq = sequences[i]
        struc = structures[i]
        loop = loops[i]

        # 1. Encode Sequence (Channels 0-3)
        for j, char in enumerate(seq):
            if char in seq_map:
                features[i, j, seq_map[char]] = 1.0

        # 2. Encode Structure (Channels 4-6)
        for j, char in enumerate(struc):
            if char in struct_map:
                features[i, j, 4 + struct_map[char]] = 1.0

        # 3. Encode Loop Type (Channels 7-13)
        for j, char in enumerate(loop):
            if char in loop_map:
                features[i, j, 7 + loop_map[char]] = 1.0

        # 4. Generate Adjacency Map
        p_idx, p_mask = get_structure_adj(struc)
        pair_indices[i] = p_idx
        pair_masks[i] = p_mask

    data_dict = {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "ids": ids,
    }

    # Process Targets for Train/Val
    if mode != "test":
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        # Extract lists from dataframe columns and stack them
        # df[col] contains lists of length 68
        raw_targets_list = []
        for col in target_cols:
            # Convert Series of lists to (N, 68) array
            col_data = np.array(df[col].tolist())
            raw_targets_list.append(col_data)

        # Stack to (N, 68, 5)
        raw_targets = np.stack(raw_targets_list, axis=2)

        # Pad from 68 to 107 with zeros
        pad_len = L - raw_targets.shape[1]
        if pad_len > 0:
            padding = np.zeros((N, pad_len, 5), dtype=np.float32)
            targets = np.concatenate([raw_targets, padding], axis=1)
        else:
            targets = raw_targets

        data_dict["targets"] = targets

    return data_dict


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Main entry point to get DataLoaders. Handles caching and processing.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_files = {
        "train": Config.TRAIN_CACHE,
        "val": Config.VAL_CACHE,
        "test": Config.TEST_CACHE,
    }

    metadata_files = {
        "train": Config.TRAIN_METADATA,
        "val": Config.VAL_METADATA,
        "test": Config.TEST_METADATA,
    }

    datasets = {}

    for mode in ["train", "val", "test"]:
        cache_path = cache_files[mode]
        loaded = False

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True).item()
                loaded = True
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                loaded = False

        # Process from scratch if not loaded
        if not loaded:
            df = pd.read_parquet(metadata_files[mode])
            data = process_dataframe(df, mode=mode)
            np.save(cache_path, data)

        # Apply Debug Slicing if enabled
        if Config.DEBUG:
            subset_size = min(Config.DEBUG_SUBSET_SIZE, len(data["features"]))
            data["features"] = data["features"][:subset_size]
            data["pair_indices"] = data["pair_indices"][:subset_size]
            data["pair_masks"] = data["pair_masks"][:subset_size]
            data["ids"] = data["ids"][:subset_size]
            if mode != "test":
                data["targets"] = data["targets"][:subset_size]

        datasets[mode] = RNADataset(data, mode=mode)

    # Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
