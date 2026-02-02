import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# =========================================================================
# Constants & Mappings
# =========================================================================
TOKEN_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_STRUCT = {"(": 0, ")": 1, ".": 2}
TOKEN_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


# =========================================================================
# Helper Functions
# =========================================================================
def get_couples(structure):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array where arr[i] is the index of the base paired with i.
    If i is unpaired, arr[i] = 0 (to be masked by the model).
    """
    length = len(structure)
    mapping = np.zeros(length, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                mapping[i] = j
                mapping[j] = i
    return mapping


def one_hot_encode(seq, mapping, length):
    """One-hot encodes a sequence string based on a mapping."""
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


# =========================================================================
# Dataset Class
# =========================================================================
class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, targets=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            # Create dummy targets for test set inference
            self.targets = torch.zeros(
                (len(inputs), Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.pair_indices[idx], self.targets[idx]


# =========================================================================
# Data Preprocessing
# =========================================================================
def preprocess_data(df, cache_path, load_cached_data=True, is_test=False):
    """
    Converts DataFrame into numpy arrays for inputs, structure indices, and targets.
    Uses caching to speed up subsequent runs.
    """
    # Adjust cache path extension to .npz for np.savez
    base, _ = os.path.splitext(cache_path)
    npz_path = base + ".npz"

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(npz_path):
        print(f"Loading cached data from {npz_path}...")
        try:
            data = np.load(npz_path)
            inputs = data["inputs"]
            pair_indices = data["pair_indices"]
            targets = data["targets"] if "targets" in data else None
            return inputs, pair_indices, targets
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print(f"Processing data (Is Test: {is_test})...")

    # 2. Initialize Arrays
    n_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 14

    inputs = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((n_samples, seq_len), dtype=np.int32)

    if not is_test:
        targets = np.zeros((n_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    # 3. Fill Arrays
    for idx, row in df.iterrows():
        # --- Inputs ---
        # Sequence (4)
        seq_oh = one_hot_encode(row["sequence"], TOKEN_SEQ, seq_len)
        # Structure (3)
        struct_oh = one_hot_encode(row["structure"], TOKEN_STRUCT, seq_len)
        # Loop Type (7)
        loop_oh = one_hot_encode(row["predicted_loop_type"], TOKEN_LOOP, seq_len)

        # Concatenate: (107, 14)
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # --- Pair Indices ---
        pair_indices[idx] = get_couples(row["structure"])

        # --- Targets ---
        if not is_test:
            # Targets are lists of length 68 (seq_scored). We pad to 107.
            for t_i, col in enumerate(TARGET_COLS):
                val_list = row[col]
                # Parquet loads lists directly. If JSON, might need literal_eval (but metadata is parquet)
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    targets[idx, :length, t_i] = val_list
                else:
                    # Fallback for unexpected format
                    pass

    # 4. Save Cache
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    if targets is not None:
        np.savez(npz_path, inputs=inputs, pair_indices=pair_indices, targets=targets)
    else:
        np.savez(npz_path, inputs=inputs, pair_indices=pair_indices)

    print(f"Data processed and saved to {npz_path}")
    return inputs, pair_indices, targets


# =========================================================================
# Main Data Loader Function
# =========================================================================
def get_dataloaders(load_cached_data=True, batch_size=None):
    """
    Orchestrates loading metadata, preprocessing, and creating DataLoaders.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --- Load Train ---
    # We load the dataframe only if we need to process data (cache miss or forced reload)
    # However, to check cache effectively inside preprocess, we usually pass the DF.
    # To optimize, we could check cache existence here, but preprocess_data handles the logic cleanly.

    print("Preparing Training Data...")
    df_train = pd.read_parquet(Config.TRAIN_METADATA_PATH)
    train_inputs, train_pairs, train_targets = preprocess_data(
        df_train,
        Config.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    train_dataset = RNADataset(train_inputs, train_pairs, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- Load Val ---
    print("Preparing Validation Data...")
    df_val = pd.read_parquet(Config.VAL_METADATA_PATH)
    val_inputs, val_pairs, val_targets = preprocess_data(
        df_val, Config.VAL_CACHE_PATH, load_cached_data=load_cached_data, is_test=False
    )

    val_dataset = RNADataset(val_inputs, val_pairs, val_targets)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # --- Load Test ---
    print("Preparing Test Data...")
    df_test = pd.read_parquet(Config.TEST_METADATA_PATH)
    test_inputs, test_pairs, _ = preprocess_data(
        df_test, Config.TEST_CACHE_PATH, load_cached_data=load_cached_data, is_test=True
    )

    test_dataset = RNADataset(test_inputs, test_pairs, targets=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
