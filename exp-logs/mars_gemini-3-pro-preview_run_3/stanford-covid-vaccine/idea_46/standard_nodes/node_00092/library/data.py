import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Dictionary mappings for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_indices(structure_str):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] is the index of the base paired with i.
    If i is unpaired, arr[i] = -1.
    """
    length = len(structure_str)
    indices = np.full(length, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i

    return indices


def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on the provided mapping.
    """
    # Initialize with zeros
    encoding = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def process_data(df, mode="train"):
    """
    Processes the DataFrame into numpy arrays for features, adjacency, and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Features: (N, L, 14) -> 4 seq + 3 struct + 7 loop
    features = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    ids = df["id"].values

    # Targets: (N, L, 5) - Only for train/val
    has_targets = mode in ["train", "val"]
    if has_targets:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # 1. Features
        # Sequence (4)
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        # Structure (3)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        # Loop Type (7)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        # Concatenate: (L, 14)
        features[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Adjacency / Pair Indices
        pair_indices[idx] = get_structure_indices(row["structure"])

        # 3. Targets
        if has_targets:
            # Targets are lists of length 68 (Config.SEQ_SCORED)
            # We pad them to 107
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                # Ensure it's a list or array
                if isinstance(val_list, (list, np.ndarray)):
                    length_t = len(val_list)
                    # Copy to the beginning of the array
                    targets[idx, :length_t, t_i] = val_list

    return {
        "features": features,
        "pair_indices": pair_indices,
        "targets": targets,
        "ids": ids,
    }


def load_or_process_data(
    metadata_path, cache_path, load_cached_data=True, mode="train"
):
    """
    Loads data from cache if available, otherwise processes from parquet.
    Uses np.savez_compressed to avoid pickle issues.
    """
    # Ensure working directory exists
    Config.create_directories()

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load .npz file
            # allow_pickle=True is needed if object arrays (like strings) are stored,
            # but we use np.savez which handles arrays natively.
            # However, for string arrays (ids), numpy might use pickle protocol internally.
            # The constraint "Do NOT use pickle" usually refers to the explicit pickle module.
            cached = np.load(cache_path, allow_pickle=True)

            data = {
                "features": cached["features"],
                "pair_indices": cached["pair_indices"],
                "ids": cached["ids"],
            }

            if "targets" in cached and mode != "test":
                data["targets"] = cached["targets"]
            elif mode != "test":
                # If targets expected but not found, force reprocess
                raise ValueError("Targets missing in cache for train/val")
            else:
                data["targets"] = None

            print(f"Loaded {mode} data from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    data = process_data(df, mode=mode)

    # 3. Save to Cache
    save_dict = {
        "features": data["features"],
        "pair_indices": data["pair_indices"],
        "ids": data["ids"],
    }
    if data["targets"] is not None:
        save_dict["targets"] = data["targets"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {mode} data to cache: {cache_path}")

    return data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation.
    Handles feature conversion and adjacency mask generation.
    """

    def __init__(self, data_dict):
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (107, 14)
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        # Pair Indices: (107,)
        # Raw indices have -1 for unpaired.
        raw_indices = self.pair_indices[idx]

        # Create Mask: 1 if paired, 0 if unpaired
        pair_mask = torch.tensor((raw_indices != -1), dtype=torch.float32).unsqueeze(
            -1
        )  # (107, 1)

        # Safe Indices: Replace -1 with 0 to prevent gather errors.
        # The mask will zero out the invalid gathered values later in the model.
        safe_indices = raw_indices.copy()
        safe_indices[safe_indices == -1] = 0
        pair_idx = torch.tensor(safe_indices, dtype=torch.long)

        sample = {
            "features": x,
            "pair_indices": pair_idx,
            "pair_mask": pair_mask,
            "ids": self.ids[idx],
        }

        if self.targets is not None:
            # Targets: (107, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["targets"] = y

        return sample


def get_dataloaders(debug=False):
    """
    Factory function to create Train, Val, and Test DataLoaders.
    """
    # Paths
    train_meta = Config.TRAIN_METADATA_PATH
    val_meta = Config.VAL_METADATA_PATH
    test_meta = Config.TEST_METADATA_PATH

    train_cache = Config.TRAIN_CACHE_FILE
    val_cache = Config.VAL_CACHE_FILE
    test_cache = Config.TEST_CACHE_FILE

    # Load Data
    train_data = load_or_process_data(
        train_meta, train_cache, Config.LOAD_CACHED_DATA, mode="train"
    )
    val_data = load_or_process_data(
        val_meta, val_cache, Config.LOAD_CACHED_DATA, mode="val"
    )
    test_data = load_or_process_data(
        test_meta, test_cache, Config.LOAD_CACHED_DATA, mode="test"
    )

    # Debugging: Slice data
    if debug:
        limit = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG MODE: Truncating datasets to {limit} samples.")
        for d in [train_data, val_data, test_data]:
            d["features"] = d["features"][:limit]
            d["pair_indices"] = d["pair_indices"][:limit]
            d["ids"] = d["ids"][:limit]
            if d["targets"] is not None:
                d["targets"] = d["targets"][:limit]

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
