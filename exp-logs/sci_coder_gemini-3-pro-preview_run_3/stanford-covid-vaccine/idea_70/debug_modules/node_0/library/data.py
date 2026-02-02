import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Dictionaries for One-Hot Encoding
# Sequence: 4 channels
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
# Structure: 3 channels
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
# Loop Type: 7 channels
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string to determine paired indices.
    Returns:
        pair_indices: Array where arr[i] is the index of the base paired with i.
                      If unpaired, value is -1.
    """
    pair_indices = np.full(len(structure), -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i

    return pair_indices


def process_data(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for features, adjacency, and targets.

    Args:
        df (pd.DataFrame): Input dataframe containing sequences and structures.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing numpy arrays for 'features', 'pair_indices', 'pair_mask', 'targets', 'ids'.
    """
    num_samples = len(df)
    seq_len = Config.seq_length
    input_dim = Config.input_dim  # 14

    # Initialize arrays
    features = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_mask = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets are only present in train/val
    # Shape: (N, 68, 5)
    if mode in ["train", "val"]:
        targets = np.zeros(
            (num_samples, Config.seq_scored, Config.num_targets), dtype=np.float32
        )
    else:
        targets = None

    ids = df["id"].values

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Sequence Encoding (Channels 0-3)
        seq = row["sequence"]
        for i, char in enumerate(seq):
            if char in SEQ_MAP:
                features[idx, i, SEQ_MAP[char]] = 1.0

        # 2. Structure Encoding (Channels 4-6)
        struct = row["structure"]
        for i, char in enumerate(struct):
            if char in STRUCT_MAP:
                features[idx, i, 4 + STRUCT_MAP[char]] = 1.0

        # 3. Loop Type Encoding (Channels 7-13)
        loop = row["predicted_loop_type"]
        for i, char in enumerate(loop):
            if char in LOOP_MAP:
                features[idx, i, 7 + LOOP_MAP[char]] = 1.0

        # 4. Adjacency / Pair Indices
        # Get indices where -1 indicates unpaired
        p_indices = get_structure_indices(struct)

        # For the model, we need valid indices for gather.
        # We set unpaired indices to 0 (dummy) and use pair_mask to zero out the result.
        # Mask is 1 if paired, 0 if unpaired.
        mask = (p_indices != -1).astype(np.float32)
        p_indices_safe = p_indices.copy()
        p_indices_safe[p_indices == -1] = 0

        pair_indices[idx] = p_indices_safe
        pair_mask[idx] = mask

        # 5. Targets
        if mode in ["train", "val"]:
            # Columns are lists of floats
            for t_i, col in enumerate(Config.target_cols):
                val_list = row[col]
                # Ensure we only take up to seq_scored length
                length = min(len(val_list), Config.seq_scored)
                targets[idx, :length, t_i] = val_list[:length]

    return {
        "features": features,
        "pair_indices": pair_indices,
        "pair_mask": pair_mask,
        "targets": targets,
        "ids": ids,
    }


def load_data_cached(mode, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from metadata and caches it.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Processed data dictionary.
    """
    # Determine paths based on mode
    if mode == "train":
        metadata_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        metadata_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        metadata_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Allow pickle is sometimes needed for object arrays (like string IDs),
            # though we try to keep things numeric.
            data = np.load(cache_path, allow_pickle=True)

            # Reconstruct dictionary
            output = {
                "features": data["features"],
                "pair_indices": data["pair_indices"],
                "pair_mask": data["pair_mask"],
                "ids": data["ids"],
            }
            if "targets" in data and mode != "test":
                output["targets"] = data["targets"]
            else:
                output["targets"] = None

            print(f"Loaded {mode} data from cache: {cache_path}")
            return output
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    # Debugging subset
    if Config.debug:
        df = df.iloc[: Config.debug_subset_size].copy()
        print(f"Debug mode: Reduced {mode} dataset to {len(df)} samples.")

    processed_data = process_data(df, mode=mode)

    # Save to cache
    save_dict = {
        "features": processed_data["features"],
        "pair_indices": processed_data["pair_indices"],
        "pair_mask": processed_data["pair_mask"],
        "ids": processed_data["ids"],
    }
    if processed_data["targets"] is not None:
        save_dict["targets"] = processed_data["targets"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {mode} data to cache: {cache_path}")

    return processed_data


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Serves One-Hot encoded features, structural adjacency maps, and targets.
    """

    def __init__(self, data_dict, mode="train"):
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_mask = data_dict["pair_mask"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to torch tensors
        # Features: (107, 14)
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        # Pair Indices: (107,) Long
        p_idx = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        # Pair Mask: (107,) Float (used for multiplicative masking)
        p_mask = torch.tensor(self.pair_mask[idx], dtype=torch.float32)

        if self.mode in ["train", "val"]:
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, p_idx, p_mask, y
        else:
            # For test, return ID as well for submission file creation
            sample_id = self.ids[idx]
            return x, p_idx, p_mask, sample_id


def get_loaders(batch_size=32, num_workers=4, load_cached_data=True):
    """
    Utility to get DataLoaders for train, val, and test sets.
    """
    # Load data
    train_data = load_data_cached("train", load_cached_data)
    val_data = load_data_cached("val", load_cached_data)
    test_data = load_data_cached("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
