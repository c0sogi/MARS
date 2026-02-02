import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Serves inputs, targets, adjacency indices, and pair masks.
    """

    def __init__(self, X, y, adj, mask, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.adj = torch.tensor(adj, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "inputs": self.X[idx],  # (107, 14)
            "targets": self.y[idx],  # (107, 5)
            "adjacency": self.adj[idx],  # (107,)
            "mask": self.mask[idx],  # (107,)
            "ids": self.ids[idx],  # str
        }


# ==========================================
# Processing Functions
# ==========================================
def parse_structure(structure_str, length):
    """
    Parses dot-bracket structure to generate adjacency indices and pair mask.
    """
    adj = np.arange(length)  # Default: point to self
    mask = np.zeros(length, dtype=np.float32)  # Default: unpaired (0)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Record pair
                adj[i] = j
                adj[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return adj, mask


def encode_sequence(seq_str, map_dict, length, dim):
    """
    One-hot encodes a sequence string.
    """
    arr = np.zeros((length, dim), dtype=np.float32)
    for i, char in enumerate(seq_str):
        if i >= length:
            break
        if char in map_dict:
            arr[i, map_dict[char]] = 1.0
    return arr


def process_dataframe(df):
    """
    Converts DataFrame into numpy arrays for the dataset.
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 14

    # Pre-allocate arrays
    X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    y = np.zeros((n_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    adj = np.zeros((n_samples, seq_len), dtype=np.int64)
    mask = np.zeros((n_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Check if targets exist
    has_targets = all(col in df.columns for col in Config.TARGET_COLS)

    for idx, (_, row) in enumerate(df.iterrows()):
        # 1. Input Features
        # Sequence (4)
        seq_feat = encode_sequence(row["sequence"], SEQ_MAP, seq_len, 4)
        # Structure (3)
        struct_feat = encode_sequence(row["structure"], STRUCT_MAP, seq_len, 3)
        # Loop Type (7)
        loop_feat = encode_sequence(row["predicted_loop_type"], LOOP_MAP, seq_len, 7)

        # Concatenate
        X[idx] = np.concatenate([seq_feat, struct_feat, loop_feat], axis=1)

        # 2. Structural Adjacency & Mask
        adj[idx], mask[idx] = parse_structure(row["structure"], seq_len)

        # 3. Targets
        if has_targets:
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Targets are length 68, pad to 107
                valid_len = min(len(val_list), seq_len)
                y[idx, :valid_len, t_i] = val_list[:valid_len]

    return X, y, adj, mask, ids


# ==========================================
# Caching & Loading Logic
# ==========================================
def load_and_cache_data(parquet_path, split_name, load_cached_data=True, debug=False):
    """
    Loads data from cache or processes from parquet.
    """
    cache_filename = f"{split_name}_data.npz"
    if debug:
        cache_filename = f"{split_name}_data_debug.npz"

    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return data["X"], data["y"], data["adj"], data["mask"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Handle Debug Slicing
    if debug:
        print(
            f"DEBUG: Slicing {split_name} dataset to {Config.DEBUG_SUBSET_SIZE} samples."
        )
        df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

    X, y, adj, mask, ids = process_dataframe(df)

    # 3. Save Cache
    print(f"Saving {split_name} data to cache: {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y, adj=adj, mask=mask, ids=ids)

    return X, y, adj, mask, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Train ---
    X_train, y_train, adj_train, mask_train, ids_train = load_and_cache_data(
        Config.TRAIN_PATH, "train", load_cached_data, debug
    )
    train_ds = RNADataset(X_train, y_train, adj_train, mask_train, ids_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Val ---
    X_val, y_val, adj_val, mask_val, ids_val = load_and_cache_data(
        Config.VAL_PATH, "val", load_cached_data, debug
    )
    val_ds = RNADataset(X_val, y_val, adj_val, mask_val, ids_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Test ---
    # Note: Test set usually doesn't need debug slicing unless specifically testing inference pipeline
    # We apply debug slicing if requested to keep runtime low during development
    X_test, y_test, adj_test, mask_test, ids_test = load_and_cache_data(
        Config.TEST_PATH, "test", load_cached_data, debug
    )
    test_ds = RNADataset(X_test, y_test, adj_test, mask_test, ids_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
