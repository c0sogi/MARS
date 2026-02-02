import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Constants & Mappings
# ==========================================
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCTURE_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_indices(structure_str):
    """
    Parses dot-bracket structure string to find paired indices.
    Returns an array of shape (L,) where arr[i] is the index paired with i,
    or -1 if unpaired.
    """
    length = len(structure_str)
    indices = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                indices[start_idx] = i
                indices[i] = start_idx

    return indices


def process_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into numpy arrays for features, adjacency, and targets.

    Args:
        df: Pandas DataFrame containing the data.
        is_test: Boolean, if True, targets are generated as zeros.

    Returns:
        features: (N, 107, 14) float32
        adjacency: (N, 107) int32
        targets: (N, 107, 5) float32
        ids: List of IDs
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    adjacency = np.zeros((num_samples, seq_len), dtype=np.int32)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].tolist()

    # Extract columns
    sequences = df["sequence"].tolist()
    structures = df["structure"].tolist()
    loops = df["predicted_loop_type"].tolist()

    # Target columns
    target_cols = Config.get_target_columns()

    for i in range(num_samples):
        seq = sequences[i]
        struct = structures[i]
        loop = loops[i]

        # 1. Features
        for j, char in enumerate(seq):
            if char in NUCLEOTIDE_MAP:
                features[i, j, NUCLEOTIDE_MAP[char]] = 1.0

        for j, char in enumerate(struct):
            if char in STRUCTURE_MAP:
                features[i, j, 4 + STRUCTURE_MAP[char]] = 1.0

        for j, char in enumerate(loop):
            if char in LOOP_TYPE_MAP:
                features[i, j, 7 + LOOP_TYPE_MAP[char]] = 1.0

        # 2. Adjacency
        adjacency[i] = get_structure_indices(struct)

        # 3. Targets
        if not is_test:
            for k, col in enumerate(target_cols):
                # The target column contains a list/array of floats
                val_list = df.iloc[i][col]

                # Handle potential float/string issues if not loaded correctly, though parquet handles this.
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    targets[i, :length, k] = val_list
                # Remaining positions (68-106) stay 0.0

    return features, adjacency, targets, ids


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, features, adjacency, targets, ids):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.adjacency = torch.tensor(adjacency, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Returns: (107, 14), (107,), (107, 5)
        return self.features[idx], self.adjacency[idx], self.targets[idx]


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Loads data, processes it (or loads from cache), and returns DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int, optional): Override Config.BATCH_SIZE.
        num_workers (int, optional): Override Config.NUM_WORKERS.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    datasets = {}
    test_ids = []

    for split in splits:
        cache_path = os.path.join(cache_dir, f"{split}_cache.npz")

        # Try loading from cache
        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                features = data["features"]
                adjacency = data["adjacency"]
                targets = data["targets"]
                ids = data["ids"].tolist()
                loaded = True
            except Exception as e:
                print(f"Failed to load {split} cache: {e}. Recomputing.")

        if not loaded:
            # Load metadata
            if split == "train":
                meta_path = Config.TRAIN_METADATA
                is_test = False
            elif split == "val":
                meta_path = Config.VAL_METADATA
                is_test = False
            else:
                meta_path = Config.TEST_METADATA
                is_test = True

            df = pd.read_parquet(meta_path)

            # Debug mode subsampling
            if Config.DEBUG:
                df = df.iloc[: Config.DEBUG_SIZE]

            features, adjacency, targets, ids = process_dataframe(df, is_test=is_test)

            # Save to cache
            np.savez(
                cache_path,
                features=features,
                adjacency=adjacency,
                targets=targets,
                ids=ids,
            )

        datasets[split] = RNADataset(features, adjacency, targets, ids)
        if split == "test":
            test_ids = ids

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, test_ids
