import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_structure_adj

# =========================================================================
# Constants & Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# =========================================================================
# Dataset Class
# =========================================================================
class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, inputs, adj, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Input features (N, 107, 14).
            adj (np.ndarray): Adjacency indices (N, 107).
            targets (np.ndarray, optional): Target values (N, 107, 5).
            ids (np.ndarray, optional): Sample IDs (N,).
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.adj = torch.tensor(adj, dtype=torch.long)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "adj_indices": self.adj[idx],
            "id": self.ids[idx] if self.ids is not None else "",
        }
        if self.targets is not None:
            sample["targets"] = self.targets[idx]
        return sample


# =========================================================================
# Helper Functions
# =========================================================================
def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into numpy arrays for the model.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize containers
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    adj = np.zeros((num_samples, seq_len), dtype=np.int32)
    ids = df["id"].values

    # Initialize targets
    if not is_test:
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
    else:
        targets = None

    # Pre-fetch columns for speed
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    # Process Targets if training data
    if not is_test:
        # Extract list columns and stack them
        # df[col] is a Series of lists (length 68)
        raw_target_list = []
        for col in Config.TARGET_COLS:
            # Convert series of lists to 2D array
            col_data = np.array(df[col].tolist(), dtype=np.float32)
            raw_target_list.append(col_data)

        # Stack to shape (5, N, 68) then transpose to (N, 68, 5)
        raw_targets = np.stack(raw_target_list, axis=0).transpose(1, 2, 0)

    # Iterate over samples to build inputs and adjacency
    for i in range(num_samples):
        # 1. One-Hot Encoding
        ohe_seq = one_hot_encode(sequences[i], SEQ_MAP, seq_len)
        ohe_struct = one_hot_encode(structures[i], STRUCT_MAP, seq_len)
        ohe_loop = one_hot_encode(loops[i], LOOP_MAP, seq_len)

        # Concatenate channels
        inputs[i] = np.concatenate([ohe_seq, ohe_struct, ohe_loop], axis=1)

        # 2. Structural Adjacency
        adj[i] = get_structure_adj(structures[i])

        # 3. Targets (Padding)
        if not is_test:
            # raw_targets[i] has shape (68, 5)
            # We copy it into the (107, 5) container
            # The remaining positions stay 0.0
            scored_len = raw_targets.shape[1]
            targets[i, :scored_len, :] = raw_targets[i]

    return inputs, adj, targets, ids


def load_and_cache_data(path, cache_path, is_test=False, load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches it to disk.
    Handles DEBUG mode by modifying the cache path.
    """
    # Modify cache path for debug mode to avoid overwriting full data
    if Config.DEBUG:
        base, ext = os.path.splitext(cache_path)
        cache_path = f"{base}_debug{ext}"
        # In debug mode, we force reprocessing or check debug cache
        # We'll respect load_cached_data but for the debug file

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data_dict = np.load(cache_path, allow_pickle=True).item()
            return (
                data_dict["inputs"],
                data_dict["adj"],
                data_dict["targets"],
                data_dict["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from {path}...")
    df = pd.read_parquet(path)

    # Handle Debug Subset
    if Config.DEBUG:
        print(f"DEBUG MODE: Reducing dataset to {Config.DEBUG_SUBSET_SIZE} samples.")
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    inputs, adj, targets, ids = preprocess_dataframe(df, is_test=is_test)

    # 3. Save Cache
    print(f"Saving processed data to {cache_path}...")
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache_dict = {"inputs": inputs, "adj": adj, "targets": targets, "ids": ids}
    np.save(cache_path, cache_dict)

    return inputs, adj, targets, ids


# =========================================================================
# Main Interface
# =========================================================================
def get_dataloaders(load_cached_data=True):
    """
    Constructs and returns DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Train Data
    train_inputs, train_adj, train_targets, train_ids = load_and_cache_data(
        Config.TRAIN_DATA_PATH,
        Config.TRAIN_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # 2. Load Validation Data
    val_inputs, val_adj, val_targets, val_ids = load_and_cache_data(
        Config.VAL_DATA_PATH,
        Config.VAL_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # 3. Load Test Data
    test_inputs, test_adj, test_targets, test_ids = load_and_cache_data(
        Config.TEST_DATA_PATH,
        Config.TEST_CACHE,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    # 4. Create Datasets
    train_dataset = RNADataset(train_inputs, train_adj, train_targets, train_ids)
    val_dataset = RNADataset(val_inputs, val_adj, val_targets, val_ids)
    test_dataset = RNADataset(test_inputs, test_adj, targets=None, ids=test_ids)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
