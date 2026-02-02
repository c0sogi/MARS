import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.

    Returns:
        inputs (torch.Tensor): Shape (Seq_Len, Input_Channels) -> (107, 14)
        targets (torch.Tensor): Shape (Seq_Len, Num_Targets) -> (107, 5)
    """

    def __init__(self, inputs, targets=None, ids=None):
        self.inputs = inputs
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy rows to tensors
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            # Return zero-filled tensor for test set (or inference)
            y = torch.zeros((Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32)

        return x, y


def get_one_hot(sequence, mapping, num_classes):
    """
    Converts a sequence string into a one-hot numpy array.

    Args:
        sequence (str): The input sequence (e.g., "AGUC...").
        mapping (dict): Dictionary mapping characters to integers.
        num_classes (int): Total number of classes for the feature.

    Returns:
        np.ndarray: Shape (Seq_Len, Num_Classes)
    """
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, num_classes), dtype=np.float32)

    for i, char in enumerate(sequence):
        if char in mapping:
            idx = mapping[char]
            one_hot[i, idx] = 1.0
    return one_hot


def process_data(csv_path, mode="train", load_cached_data=True, max_samples=None):
    """
    Processes data from CSV to Numpy arrays, with caching.

    Args:
        csv_path (str): Path to the source CSV file.
        mode (str): Dataset mode ('train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from .npy cache.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        tuple: (inputs, targets, ids)
    """
    # Ensure directories exist
    Config.create_dirs()
    cache_dir = Config.CACHE_DIR

    # Define cache file paths
    inputs_cache = os.path.join(cache_dir, f"{mode}_inputs.npy")
    targets_cache = os.path.join(cache_dir, f"{mode}_targets.npy")
    ids_cache = os.path.join(cache_dir, f"{mode}_ids.npy")

    # 1. Attempt to Load from Cache
    if load_cached_data:
        # Check if essential files exist
        cache_exists = os.path.exists(inputs_cache) and os.path.exists(ids_cache)
        # For train/val, targets must also exist
        if mode != "test":
            cache_exists = cache_exists and os.path.exists(targets_cache)

        if cache_exists:
            print(f"Loading {mode} data from cache: {cache_dir}")
            try:
                inputs = np.load(inputs_cache)
                ids = np.load(ids_cache, allow_pickle=True)

                if mode != "test":
                    targets = np.load(targets_cache)
                else:
                    targets = None

                # Apply max_samples slicing if requested
                if max_samples is not None:
                    inputs = inputs[:max_samples]
                    ids = ids[:max_samples]
                    if targets is not None:
                        targets = targets[:max_samples]

                return inputs, targets, ids
            except Exception as e:
                print(f"Failed to load cache ({e}). Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    # Load metadata CSV
    df = pd.read_csv(csv_path)

    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    # Inputs: (N, 107, 14)
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)

    # Targets: (N, 107, 5) - Only for train/val
    if mode != "test":
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].values

    target_cols = Config.TARGET_COLS

    # Iterate over dataframe rows
    for idx, row in df.iterrows():
        # --- Feature Engineering ---
        # 1. Sequence One-Hot (4 channels)
        seq_oh = get_one_hot(
            row["sequence"], Config.TOKEN2INT_SEQ, Config.NUM_SEQ_TOKENS
        )

        # 2. Structure One-Hot (3 channels)
        struct_oh = get_one_hot(
            row["structure"], Config.TOKEN2INT_STRUCT, Config.NUM_STRUCT_TOKENS
        )

        # 3. Loop Type One-Hot (7 channels)
        loop_oh = get_one_hot(
            row["predicted_loop_type"], Config.TOKEN2INT_LOOP, Config.NUM_LOOP_TOKENS
        )

        # Concatenate all features: Shape (107, 14)
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # --- Target Parsing ---
        if mode != "test":
            for t_idx, col in enumerate(target_cols):
                val_str = row[col]
                try:
                    # Parse string list e.g. "[0.1, 0.2]" -> list
                    val_list = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val_list = []

                # Fill the target array
                # Note: Ground truth is typically provided for the first 68 bases (Config.SCORED_LEN)
                # We pad the rest with zeros (initialized)
                length = len(val_list)
                if length > 0:
                    # Truncate if somehow longer than seq_len, though unlikely
                    length = min(length, seq_len)
                    targets[idx, :length, t_idx] = val_list[:length]

    # 3. Save to Cache
    print(f"Saving {mode} data to cache...")
    np.save(inputs_cache, inputs)
    np.save(ids_cache, ids)
    if mode != "test":
        np.save(targets_cache, targets)

    return inputs, targets, ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_samples=None,
):
    """
    Creates and returns DataLoaders for Train, Validation, and Test sets.

    Args:
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached numpy arrays.
        max_samples (int): Limit number of samples (for debugging).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Train Loader
    train_inputs, train_targets, train_ids = process_data(
        Config.TRAIN_CSV, "train", load_cached_data, max_samples
    )
    train_dataset = RNADataset(train_inputs, train_targets, train_ids)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Validation Loader
    val_inputs, val_targets, val_ids = process_data(
        Config.VAL_CSV, "val", load_cached_data, max_samples
    )
    val_dataset = RNADataset(val_inputs, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Loader
    test_inputs, test_targets, test_ids = process_data(
        Config.TEST_CSV, "test", load_cached_data, max_samples
    )
    test_dataset = RNADataset(test_inputs, test_targets, test_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
