import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Helper Functions & Maps
# =========================================================================

SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_to_distances(structure):
    """
    Parses a dot-bracket structure string into an array of signed pairing distances.
    If base i is paired with base j, dist[i] = j - i.
    If base i is unpaired, dist[i] = 0.
    """
    n = len(structure)
    dists = np.zeros(n, dtype=np.float32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair (j, i) created
                # Distance for j (upstream) is i - j (positive)
                # Distance for i (downstream) is j - i (negative)
                dists[j] = i - j
                dists[i] = j - i
    return dists


def get_sinusoidal_features(distances, dim):
    """
    Encodes signed distances into sinusoidal features.
    distances: (N, L) array of signed distances.
    dim: Output dimension (must be even).
    Returns: (N, L, dim) array.
    """
    batch_size, seq_len = distances.shape
    half_dim = dim // 2

    # Compute frequencies
    # exp(arange(0, half_dim) * -log(10000) / half_dim)
    emb = np.exp(
        np.arange(0, half_dim, dtype=np.float32) * -(np.log(10000.0) / half_dim)
    )

    # Broadcast multiplication: (N, L, 1) * (1, 1, half_dim) -> (N, L, half_dim)
    scaled_dists = distances[:, :, None] * emb[None, None, :]

    # Compute Sin and Cos
    pe_sin = np.sin(scaled_dists)
    pe_cos = np.cos(scaled_dists)

    # Concatenate -> (N, L, dim)
    pe = np.concatenate([pe_sin, pe_cos], axis=-1)
    return pe


# =========================================================================
# Processing & Caching Logic
# =========================================================================


def process_data(parquet_path, cache_path, load_cached_data=True, is_test=False):
    """
    Loads data from Parquet, processes features, and caches them as .npz.
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=True is required if the npz contains object arrays (like string IDs)
            data = np.load(cache_path, allow_pickle=True)

            # Verify keys
            required_keys = ["ids", "seq_indices", "loop_indices", "dist_features"]
            if not is_test:
                required_keys.append("targets")

            if all(k in data for k in required_keys):
                # Convert NpzFile to dict to keep in memory
                return {k: data[k] for k in required_keys}
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path} ({e}). Recomputing...")

    # 2. Process from scratch
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # -- IDs --
    ids = df[Config.ID_COL].values

    # -- Sequence Indices --
    # Map chars to integers
    seq_indices = np.array(
        [[SEQ_MAP.get(c, 0) for c in seq] for seq in df[Config.SEQUENCE_COL]],
        dtype=np.int64,
    )

    # -- Loop Type Indices --
    loop_indices = np.array(
        [[LOOP_MAP.get(c, 0) for c in loop] for loop in df[Config.LOOP_TYPE_COL]],
        dtype=np.int64,
    )

    # -- Structure / Distance Features --
    structures = df[Config.STRUCTURE_COL].values
    raw_dists = np.array([parse_structure_to_distances(s) for s in structures])

    # Apply Sinusoidal Encoding
    dist_features = get_sinusoidal_features(raw_dists, Config.EMBED_DIM_DIST)
    dist_features = dist_features.astype(np.float32)

    result = {
        "ids": ids,
        "seq_indices": seq_indices,
        "loop_indices": loop_indices,
        "dist_features": dist_features,
    }

    # -- Targets (Train/Val only) --
    if not is_test:
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Data is stored as lists in the dataframe columns.
        target_arrays = []
        for col in Config.TARGET_COLS:
            # Stack lists into (N, 68) array
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along last axis -> (N, 68, 3)
        targets = np.stack(target_arrays, axis=-1).astype(np.float32)
        result["targets"] = targets

    # 3. Save to Cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **result)

    return result


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        self.ids = data_dict["ids"]
        self.seq_indices = data_dict["seq_indices"]
        self.loop_indices = data_dict["loop_indices"]
        self.dist_features = data_dict["dist_features"]
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Inputs
        item = {
            "seq": torch.tensor(self.seq_indices[idx], dtype=torch.long),
            "loop": torch.tensor(self.loop_indices[idx], dtype=torch.long),
            "dist": torch.tensor(self.dist_features[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }

        # Targets
        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


# =========================================================================
# DataLoader Factory
# =========================================================================


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates data loading, processing, caching, and DataLoader creation.
    Handles DEBUG subsetting.
    """
    Config.create_dirs()

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_data.npz")
    val_cache = os.path.join(Config.WORKING_DIR, "val_data.npz")
    test_cache = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # Load Data (Process or Load Cache)
    train_data = process_data(
        Config.TRAIN_PATH, train_cache, load_cached_data, is_test=False
    )
    val_data = process_data(Config.VAL_PATH, val_cache, load_cached_data, is_test=False)
    test_data = process_data(
        Config.TEST_PATH, test_cache, load_cached_data, is_test=True
    )

    # Handle DEBUG mode: Subset data
    if Config.DEBUG:
        limit = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG mode: Subsetting data to {limit} samples.")

        for k in train_data:
            train_data[k] = train_data[k][:limit]
        for k in val_data:
            val_data[k] = val_data[k][:limit]
        for k in test_data:
            test_data[k] = test_data[k][:limit]

    # Create Datasets
    train_ds = RNADataset(train_data, is_test=False)
    val_ds = RNADataset(val_data, is_test=False)
    test_ds = RNADataset(test_data, is_test=True)

    # Create DataLoaders
    # Pin memory enables faster transfer to CUDA
    pin_mem = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_mem,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_mem,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader, test_loader
