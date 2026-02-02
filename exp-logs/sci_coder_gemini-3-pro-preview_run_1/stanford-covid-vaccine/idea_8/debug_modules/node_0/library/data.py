import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_pairing_distances(structure):
    """
    Parses a dot-bracket structure string and calculates relative pairing distances.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len(structure),) containing signed distances
                    (j - i) for paired bases, and 0 for unpaired bases.
    """
    length = len(structure)
    distances = np.zeros(length, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Distance for the opening base at j
                distances[j] = i - j
                # Distance for the closing base at i
                distances[i] = j - i

    return distances


def process_data(split_key, config, load_cached_data=True):
    """
    Loads raw data, processes features (tokenization, distance calculation),
    and caches the result as .npz files.

    Args:
        split_key (str): 'train', 'val', or 'test'.
        config (class): Configuration class.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for inputs and targets.
    """
    cache_file = os.path.join(config.CACHE_DIR, f"{split_key}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split_key} data from cache: {cache_file}")
        try:
            loaded = np.load(cache_file)
            data = {
                "ids": loaded["ids"],
                "sequences": loaded["sequences"],
                "loops": loaded["loops"],
                "distances": loaded["distances"],
            }
            if "targets" in loaded:
                data["targets"] = loaded["targets"]
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Process from scratch
    print(f"Processing {split_key} data from metadata...")

    # Determine source file
    if split_key == "train":
        meta_path = config.TRAIN_METADATA
    elif split_key == "val":
        meta_path = config.VAL_METADATA
    elif split_key == "test":
        meta_path = config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split_key: {split_key}")

    df = pd.read_parquet(meta_path)

    # Debugging subset
    if config.DEBUG:
        df = df.head(config.DEBUG_SAMPLES)
        print(f"DEBUG MODE: Reduced {split_key} size to {len(df)}")

    # Initialize arrays
    num_samples = len(df)
    seq_len = config.SEQ_LEN

    # Feature: Sequence Tokenization
    # Map chars to ints using VOCAB_MAP_SEQ
    seq_map = config.VOCAB_MAP_SEQ
    sequences = np.array(
        [[seq_map.get(c, 0) for c in seq] for seq in df["sequence"]], dtype=np.int64
    )

    # Feature: Loop Type Tokenization
    # Map chars to ints using VOCAB_MAP_LOOP
    loop_map = config.VOCAB_MAP_LOOP
    loops = np.array(
        [[loop_map.get(c, 0) for c in loop] for loop in df["predicted_loop_type"]],
        dtype=np.int64,
    )

    # Feature: Structural Distances
    distances = np.array(
        [get_pairing_distances(s) for s in df["structure"]], dtype=np.float32
    )

    # IDs
    ids = df["id"].values

    # Result dictionary
    data = {"ids": ids, "sequences": sequences, "loops": loops, "distances": distances}

    # Targets (Only for train/val)
    if split_key in ["train", "val"]:
        target_arrays = []
        for col in config.TARGET_COLS:
            # Each column in df is a list/array of floats. Stack them.
            # Shape of col_data: (N, seq_scored)
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along the last dimension -> (N, seq_scored, 5)
        targets = np.stack(target_arrays, axis=2).astype(np.float32)
        data["targets"] = targets

    # 3. Save to cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    print(f"Saving {split_key} data to cache: {cache_file}")
    np.savez_compressed(cache_file, **data)

    return data


class RNADataset(Dataset):
    def __init__(self, data, config, is_test=False):
        """
        Args:
            data (dict): Dictionary containing numpy arrays from process_data.
            config (class): Configuration class.
            is_test (bool): Whether this is the test set (no targets).
        """
        self.ids = data["ids"]
        self.sequences = data["sequences"]
        self.loops = data["loops"]
        self.distances = data["distances"]
        self.is_test = is_test
        self.config = config

        if not self.is_test:
            self.targets = data["targets"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        dist = torch.tensor(self.distances[idx], dtype=torch.float32)

        # Targets
        if self.is_test:
            # Return dummy targets for test set to maintain consistent signature if needed,
            # or just return inputs. The training loop/inference loop should handle this.
            # We'll return a placeholder of correct shape (seq_scored, 5)
            target = torch.zeros(
                (self.config.PRED_LEN, self.config.NUM_TARGETS), dtype=torch.float32
            )
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "id": self.ids[idx],
            "sequence": seq,
            "loop": loop,
            "distance": dist,
            "target": target,
        }


def get_dataloaders(config, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (class): Configuration class.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Process Data
    train_data = process_data("train", config, load_cached_data)
    val_data = process_data("val", config, load_cached_data)
    test_data = process_data("test", config, load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data, config, is_test=False)
    val_dataset = RNADataset(val_data, config, is_test=False)
    test_dataset = RNADataset(test_data, config, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
