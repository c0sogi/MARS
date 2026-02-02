import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_config_hash

# --------------------------------------------------------------------------
# Encoding Dictionaries
# --------------------------------------------------------------------------
# Nucleotides: A, G, U, C
TOKEN2INT_SEQ = {"A": 0, "G": 1, "U": 2, "C": 3}
# Structure: (, ), .
TOKEN2INT_STRUCT = {"(": 0, ")": 1, ".": 2}
# Loop Type: S, M, I, B, H, E, X
TOKEN2INT_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, inputs, targets=None, ids=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.inputs[idx], self.targets[idx]
        else:
            # For inference, return input and ID to map predictions back to sample
            return self.inputs[idx], self.ids[idx]


def one_hot_encode(seq, mapping, dim):
    """
    Helper to one-hot encode a string sequence.
    """
    seq_len = len(seq)
    encoding = np.zeros((seq_len, dim), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def preprocess_data(df, is_test=False):
    """
    Converts DataFrame columns into numpy arrays for inputs and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        inputs (np.ndarray): Shape (N, 107, 14)
        targets (np.ndarray): Shape (N, 68, 5) or None
        ids (np.ndarray): Shape (N,)
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 14

    # Initialize input array
    inputs = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    ids = df["id"].values

    # Process inputs
    for i, row in df.iterrows():
        # 1. Sequence (4 dims)
        seq_enc = one_hot_encode(row["sequence"], TOKEN2INT_SEQ, 4)

        # 2. Structure (3 dims)
        struct_enc = one_hot_encode(row["structure"], TOKEN2INT_STRUCT, 3)

        # 3. Loop Type (7 dims)
        loop_enc = one_hot_encode(row["predicted_loop_type"], TOKEN2INT_LOOP, 7)

        # Concatenate features: (107, 4+3+7) -> (107, 14)
        inputs[i] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

    targets = None
    if not is_test:
        # Target columns in specific order
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        pred_len = Config.PRED_LEN
        output_dim = Config.OUTPUT_DIM

        targets = np.zeros((n_samples, pred_len, output_dim), dtype=np.float32)

        for i, row in df.iterrows():
            for j, col in enumerate(target_cols):
                # Columns are lists of floats
                val_list = row[col]
                # Ensure length matches pred_len (68)
                length = min(len(val_list), pred_len)
                targets[i, :length, j] = val_list[:length]

    return inputs, targets, ids


def get_dataloaders(load_cached_data=Config.LOAD_CACHED_DATA, debug=Config.DEBUG):
    """
    Main function to get DataLoaders. Handles caching and preprocessing.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed .npz files.
        debug (bool): If True, uses a small subset of data.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Setup paths and hash
    config_hash = get_config_hash()

    cache_train_path = os.path.join(Config.CACHE_DIR, f"train_data_{config_hash}.npz")
    cache_val_path = os.path.join(Config.CACHE_DIR, f"val_data_{config_hash}.npz")
    cache_test_path = os.path.join(Config.CACHE_DIR, f"test_data_{config_hash}.npz")

    # 2. Check Cache
    data_exists = (
        os.path.exists(cache_train_path)
        and os.path.exists(cache_val_path)
        and os.path.exists(cache_test_path)
    )

    if load_cached_data and data_exists:
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        train_data = np.load(cache_train_path, allow_pickle=True)
        val_data = np.load(cache_val_path, allow_pickle=True)
        test_data = np.load(cache_test_path, allow_pickle=True)

        X_train, y_train = train_data["inputs"], train_data["targets"]
        X_val, y_val = val_data["inputs"], val_data["targets"]
        X_test, ids_test = test_data["inputs"], test_data["ids"]

    else:
        print("Preprocessing data from scratch...")
        # Load Parquet files
        df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
        df_val = pd.read_parquet(Config.VAL_DATA_PATH)
        df_test = pd.read_parquet(Config.TEST_DATA_PATH)

        # Process
        X_train, y_train, _ = preprocess_data(df_train, is_test=False)
        X_val, y_val, _ = preprocess_data(df_val, is_test=False)
        X_test, _, ids_test = preprocess_data(df_test, is_test=True)

        # Save to cache
        np.savez_compressed(cache_train_path, inputs=X_train, targets=y_train)
        np.savez_compressed(cache_val_path, inputs=X_val, targets=y_val)
        np.savez_compressed(cache_test_path, inputs=X_test, ids=ids_test)
        print(f"Data cached to {Config.CACHE_DIR}")

    # 3. Debug Slicing
    if debug:
        print("DEBUG MODE: Slicing datasets to 100 samples.")
        X_train, y_train = X_train[:100], y_train[:100]
        X_val, y_val = X_val[:100], y_val[:100]
        X_test, ids_test = X_test[:100], ids_test[:100]

    # 4. Create Datasets
    train_dataset = RNADataset(X_train, y_train)
    val_dataset = RNADataset(X_val, y_val)
    test_dataset = RNADataset(X_test, ids=ids_test)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
