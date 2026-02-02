import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Token Maps
# =========================================================================
TOKEN_MAP_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_pairing_distance(structure):
    """
    Parses a dot-bracket structure string and calculates the signed pairing distance.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len(structure),) containing signed distances.
                    If i pairs with j (i < j), arr[i] = j - i, arr[j] = i - j.
                    Unpaired bases are 0.
    """
    n = len(structure)
    distances = np.zeros(n, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i is current (closing), j is opening
                # dist for j (opening) is i - j (positive)
                # dist for i (closing) is j - i (negative)
                dist = i - j
                distances[j] = dist
                distances[i] = -dist

    return distances


def process_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into numpy arrays for features and targets.
    """
    # 1. Sequence Encoding
    # Convert string column to list of lists of indices
    sequences = (
        df["sequence"].apply(lambda s: [TOKEN_MAP_SEQ.get(c, 0) for c in s]).tolist()
    )
    sequences = np.array(sequences, dtype=np.int32)  # (N, 107)

    # 2. Loop Type Encoding
    loops = (
        df["predicted_loop_type"]
        .apply(lambda s: [TOKEN_MAP_LOOP.get(c, 0) for c in s])
        .tolist()
    )
    loops = np.array(loops, dtype=np.int32)  # (N, 107)

    # 3. Structure / Pairing Distance
    structures = df["structure"].values
    distances = np.array(
        [get_pairing_distance(s) for s in structures], dtype=np.int32
    )  # (N, 107)

    # 4. Targets
    if not is_test:
        # Extract specific target columns defined in Config
        # Each cell in the dataframe for these columns is a list/array of length 68
        # We stack them to get (N, 68, 3)
        target_arrays = []
        for col in Config.TARGET_COLS:
            # Stack the lists for this column -> (N, 68)
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along the last dimension -> (N, 68, 3)
        targets = np.stack(target_arrays, axis=2).astype(np.float32)

        # Also extract ids for tracking
        ids = df["id"].values
    else:
        # Test set has no targets, create dummy (N, 68, 3)
        n_samples = len(df)
        targets = np.zeros(
            (n_samples, Config.PRED_LEN, Config.N_OUTPUTS), dtype=np.float32
        )
        ids = df["id"].values

    return {
        "ids": ids,
        "sequences": sequences,
        "loops": loops,
        "distances": distances,
        "targets": targets,
    }


def load_or_process_data(split, parquet_path, load_cached_data=True):
    """
    Loads data from cache or processes from Parquet.

    Args:
        split (str): 'train', 'val', or 'test'.
        parquet_path (str): Path to input parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{split}_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {split} data from cache: {cache_file}")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            return {k: loaded[k] for k in loaded.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {split} data from {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    is_test = split == "test"
    data_dict = process_dataframe(df, is_test=is_test)

    # Save to cache
    print(f"Saving {split} data to cache: {cache_file}")
    np.savez_compressed(cache_file, **data_dict)

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.ids = data_dict["ids"]
        self.sequences = data_dict["sequences"]
        self.loops = data_dict["loops"]
        self.distances = data_dict["distances"]
        self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        dist = torch.tensor(self.distances[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Return a dictionary for flexibility
        return {
            "sequence": seq,  # (107,)
            "loop": loop,  # (107,)
            "distance": dist,  # (107,)
            "target": target,  # (68, 3)
            "id": self.ids[idx],  # str
        }


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load/Process Data
    train_data = load_or_process_data("train", Config.TRAIN_PATH, load_cached_data)
    val_data = load_or_process_data("val", Config.VAL_PATH, load_cached_data)
    test_data = load_or_process_data("test", Config.TEST_PATH, load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
