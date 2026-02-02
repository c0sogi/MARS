import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def compute_pair_dist(structure_str, seq_len):
    """
    Parses a dot-bracket structure string and calculates the signed pairing distance.
    For a pair (i, j), index i has value (j - i) and index j has value (i - j).
    Unpaired bases have value 0.
    """
    dists = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is upstream (smaller index), i is downstream (larger index)
                # Distance at j: i - j (positive)
                # Distance at i: j - i (negative)
                dists[j] = float(i - j)
                dists[i] = float(j - i)
    return dists


def load_and_process_data(split, debug=False, load_cached_data=True):
    """
    Loads data from Parquet, processes features, and handles caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        debug (bool): If True, uses a subset of data.
        load_cached_data (bool): If True, attempts to load from .npz cache.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    # Determine cache filename
    suffix = "_debug" if debug else ""
    cache_filename = f"{split}_data{suffix}.npz"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split} data from cache: {cache_path}")
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                return {
                    "sequences": data["sequences"],
                    "loop_types": data["loop_types"],
                    "pair_dists": data["pair_dists"],
                    "targets": data["targets"],
                    "ids": data["ids"],
                }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load raw data from Parquet
    if split == "train":
        file_path = Config.TRAIN_FILE
    elif split == "val":
        file_path = Config.VAL_FILE
    elif split == "test":
        file_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"Processing {split} data from {file_path}...")
    df = pd.read_parquet(file_path)

    # Handle Debug Mode
    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE).copy()
        print(f"Debug mode: reduced {split} size to {len(df)}")

    # 3. Process Features
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    sequences = np.zeros((n_samples, seq_len), dtype=np.int64)
    loop_types = np.zeros((n_samples, seq_len), dtype=np.int64)
    pair_dists = np.zeros((n_samples, seq_len), dtype=np.float32)
    ids = df["id"].values.astype(str)

    # Tokenization Maps
    token_map = Config.TOKEN_MAP
    loop_map = Config.LOOP_TYPE_MAP

    # Iterate and process
    # Note: Vectorization is possible but iteration is clear and robust for structure parsing
    for idx, row in df.iterrows():
        # A. Sequence Tokenization
        seq_str = row["sequence"]
        sequences[idx] = np.array([token_map.get(c, 0) for c in seq_str])

        # B. Loop Type Tokenization
        loop_str = row["predicted_loop_type"]
        loop_types[idx] = np.array([loop_map.get(c, 0) for c in loop_str])

        # C. Structure / Pair Distance
        struct_str = row["structure"]
        pair_dists[idx] = compute_pair_dist(struct_str, seq_len)

    # 4. Process Targets
    # Targets are (N, 68, 3)
    target_len = Config.SEQ_SCORED
    num_targets = Config.NUM_TARGETS

    if split == "test":
        # Test set has no targets
        targets = np.zeros((n_samples, target_len, num_targets), dtype=np.float32)
    else:
        # Extract specific target columns
        # Each cell in the dataframe column is a list/array of floats
        targets = np.zeros((n_samples, target_len, num_targets), dtype=np.float32)

        for t_idx, col_name in enumerate(Config.TARGET_COLS):
            # Convert column of lists to 2D numpy array
            # We assume the metadata generation ensured valid lists
            col_data = np.vstack(df[col_name].values)
            targets[:, :, t_idx] = col_data

    # 5. Save to Cache
    # Ensure directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    print(f"Saving processed {split} data to {cache_path}...")
    np.savez(
        cache_path,
        sequences=sequences,
        loop_types=loop_types,
        pair_dists=pair_dists,
        targets=targets,
        ids=ids,
    )

    return {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary output from load_and_process_data
        """
        self.sequences = data_dict["sequences"]
        self.loop_types = data_dict["loop_types"]
        self.pair_dists = data_dict["pair_dists"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

        self.seq_len = Config.SEQ_LENGTH

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to tensors
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop_type = torch.tensor(self.loop_types[idx], dtype=torch.long)
        pair_dist = torch.tensor(self.pair_dists[idx], dtype=torch.float)

        # Absolute Position: 0 to 106
        position = torch.arange(self.seq_len, dtype=torch.long)

        # Targets: (68, 3)
        targets = torch.tensor(self.targets[idx], dtype=torch.float)

        sample_id = self.ids[idx]

        return {
            "sequence": sequence,
            "loop_type": loop_type,
            "pair_dist": pair_dist,
            "position": position,
            "targets": targets,
            "id": sample_id,
        }


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses reduced datasets.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data
    train_data = load_and_process_data(
        "train", debug=debug, load_cached_data=load_cached_data
    )
    val_data = load_and_process_data(
        "val", debug=debug, load_cached_data=load_cached_data
    )
    test_data = load_and_process_data(
        "test", debug=debug, load_cached_data=load_cached_data
    )

    # 2. Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # 3. Create DataLoaders
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
