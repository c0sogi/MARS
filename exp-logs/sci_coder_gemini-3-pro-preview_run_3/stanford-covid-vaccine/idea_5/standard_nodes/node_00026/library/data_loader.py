import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, inputs, targets=None, ids=None):
        self.inputs = inputs
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        sample_id = self.ids[idx]

        if self.targets is not None:
            # targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y, sample_id
        else:
            return x, sample_id


def get_one_hot(seq_list, mapping, vocab_size):
    """
    Helper to convert a list of strings into a one-hot encoded numpy array.
    """
    batch_size = len(seq_list)
    seq_len = len(seq_list[0])

    # Initialize output array
    one_hot = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)

    for i, seq in enumerate(seq_list):
        for j, char in enumerate(seq):
            if char in mapping:
                idx = mapping[char]
                one_hot[i, j, idx] = 1.0
            # If unknown char, it remains all zeros

    return one_hot


def preprocess_data(df, is_test=False):
    """
    Converts DataFrame columns into tensor-ready numpy arrays.

    Args:
        df: Pandas DataFrame containing sequence, structure, etc.
        is_test: Boolean, if True, does not process targets.

    Returns:
        inputs: (N, 107, 14) numpy array
        targets: (N, 68, 5) numpy array (or None if is_test)
        ids: List of IDs
    """
    # 1. Extract IDs
    ids = df["id"].tolist()

    # 2. Process Inputs
    # Mappings based on Config comments
    # Sequence: A, G, C, U -> 4 channels
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}

    # Structure: (, ), . -> 3 channels
    struct_map = {"(": 0, ")": 1, ".": 2}

    # Loop Type: S, M, I, B, H, E, X -> 7 channels
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    sequences = df["sequence"].tolist()
    structures = df["structure"].tolist()
    loops = df["predicted_loop_type"].tolist()

    # Generate One-Hot Encodings
    # Shape: (N, 107, C)
    oh_seq = get_one_hot(sequences, seq_map, 4)
    oh_struct = get_one_hot(structures, struct_map, 3)
    oh_loop = get_one_hot(loops, loop_map, 7)

    # Concatenate along channel dimension (axis 2)
    # Order: Sequence (4) + Structure (3) + Loop (7) = 14
    inputs = np.concatenate([oh_seq, oh_struct, oh_loop], axis=2)

    # 3. Process Targets
    targets = None
    if not is_test:
        target_cols = Config.TARGET_COLS

        # Each column in df contains lists of floats.
        # We need to stack them into (N, 68, 5)

        arrays = []
        for col in target_cols:
            # Convert column of lists to numpy array
            col_data = np.array(df[col].tolist(), dtype=np.float32)
            arrays.append(col_data)

        # Stack along the last dimension
        # Result shape: (N, 68, 5)
        targets = np.stack(arrays, axis=2)

    return inputs, targets, ids


def load_or_process(metadata_path, cache_path, is_test=False, load_cached_data=True):
    """
    Handles caching logic: Load from .npy if exists, else process from .parquet and save.
    """
    # Check if cache exists and loading is requested
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data_dict = np.load(cache_path, allow_pickle=True)
            data = data_dict.item()
            inputs = data["inputs"]
            ids = data["ids"]
            targets = data.get("targets", None)

            return inputs, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    inputs, targets, ids = preprocess_data(df, is_test=is_test)

    # Save to cache
    print(f"Saving processed data to {cache_path}...")
    save_dict = {"inputs": inputs, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.save(cache_path, save_dict)

    return inputs, targets, ids


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): If True, tries to load preprocessed data from disk.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================
    # 1. Training Data
    # ==========================
    train_inputs, train_targets, train_ids = load_or_process(
        Config.TRAIN_METADATA,
        Config.TRAIN_DATA_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    train_dataset = RNADataset(train_inputs, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # ==========================
    # 2. Validation Data
    # ==========================
    val_inputs, val_targets, val_ids = load_or_process(
        Config.VAL_METADATA,
        Config.VAL_DATA_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    val_dataset = RNADataset(val_inputs, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # ==========================
    # 3. Test Data
    # ==========================
    test_inputs, test_targets, test_ids = load_or_process(
        Config.TEST_METADATA,
        Config.TEST_DATA_CACHE,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    test_dataset = RNADataset(test_inputs, targets=None, ids=test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
