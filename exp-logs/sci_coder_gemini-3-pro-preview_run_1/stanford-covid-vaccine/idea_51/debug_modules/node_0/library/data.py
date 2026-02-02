import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Constants & Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def get_structure_distance(structure_str, seq_len):
    """
    Parses a dot-bracket structure string and calculates the signed distance
    to the paired base for each position.

    Args:
        structure_str (str): Dot-bracket notation (e.g., ".(..).").
        seq_len (int): Length of the sequence.

    Returns:
        np.ndarray: Array of shape (seq_len,) containing (j - i) if i is paired with j,
                    and 0 if unpaired.
    """
    stack = []
    mapping = {}

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                mapping[start_idx] = i
                mapping[i] = start_idx

    # Create distance array
    # If paired: value is (pair_idx - current_idx)
    # If unpaired: value is 0
    dists = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len):
        if i in mapping:
            dists[i] = mapping[i] - i

    return dists


def encode_sequence(seq_str, map_dict):
    """Encodes a string sequence into integer indices."""
    return np.array([map_dict.get(c, 0) for c in seq_str], dtype=np.int64)


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, sequences, loop_types, pair_dists, targets=None, ids=None):
        """
        PyTorch Dataset for RNA data.

        Args:
            sequences (np.ndarray): (N, 107) Integer encoded sequences.
            loop_types (np.ndarray): (N, 107) Integer encoded loop types.
            pair_dists (np.ndarray): (N, 107) Signed pairing distances.
            targets (np.ndarray, optional): (N, 107, 3) Target values. Defaults to None.
            ids (list/np.ndarray, optional): Sample IDs. Defaults to None.
        """
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_dists = pair_dists
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        item = {
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.long),
            "loop_type": torch.tensor(self.loop_types[idx], dtype=torch.long),
            "pair_dist": torch.tensor(self.pair_dists[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


# =========================================================================
# Data Processing & Loading
# =========================================================================


def process_dataframe(df, is_test=False):
    """
    Extracts features and targets from a pandas DataFrame.
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    sequences = np.zeros((n_samples, seq_len), dtype=np.int64)
    loop_types = np.zeros((n_samples, seq_len), dtype=np.int64)
    pair_dists = np.zeros((n_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Process inputs
    for i, row in df.iterrows():
        # Using row index i might be unsafe if df index is not reset, use integer location
        # However, enumerate on df.iterrows() is safe for sequential filling
        # But let's use the enumerate index for array assignment
        pass

    # Re-loop with explicit enumeration for safety
    for idx, (_, row) in enumerate(df.iterrows()):
        sequences[idx] = encode_sequence(row["sequence"], SEQ_MAP)
        loop_types[idx] = encode_sequence(row["predicted_loop_type"], LOOP_MAP)
        pair_dists[idx] = get_structure_distance(row["structure"], seq_len)

    targets = None
    if not is_test:
        # Initialize targets with zeros (padding for indices 68-106)
        # Shape: (N, 107, 3)
        targets = np.zeros((n_samples, seq_len, 3), dtype=np.float32)

        # Target columns defined in Config: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # In the dataframe, these are lists of length 68
        t_cols = Config.TARGET_COLS

        for idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col_name in enumerate(t_cols):
                # Extract the list/array
                val_array = np.array(row[col_name])
                # Assign to the first 68 positions
                # Note: seq_scored is typically 68
                scored_len = len(val_array)
                targets[idx, :scored_len, col_idx] = val_array

    return sequences, loop_types, pair_dists, targets, ids


def load_or_process_data(file_path, cache_name, load_cached_data=True, is_test=False):
    """
    Loads data from parquet, processes it, and caches it as .npz.
    """
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            sequences = data["sequences"]
            loop_types = data["loop_types"]
            pair_dists = data["pair_dists"]
            ids = data["ids"]

            if is_test:
                return sequences, loop_types, pair_dists, None, ids
            else:
                targets = data["targets"]
                return sequences, loop_types, pair_dists, targets, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {file_path}...")
    df = pd.read_parquet(file_path)

    sequences, loop_types, pair_dists, targets, ids = process_dataframe(
        df, is_test=is_test
    )

    # 3. Save to cache
    print(f"Saving processed data to {cache_path}...")
    save_dict = {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "ids": ids,
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return sequences, loop_types, pair_dists, targets, ids


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Args:
        debug (bool): If True, loads a subset of data.
        load_cached_data (bool): If True, attempts to use .npz cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create config instance to get parameters
    config = Config(debug=debug)

    # Define cache filenames
    # In debug mode, we might want different cache files or just slice the full data
    # To keep it simple and correct, we process full data then slice if debug
    # But for efficiency in repeated debug runs, we'll use standard cache names
    # and slice in memory.

    # 1. Load Data
    # Train
    train_seq, train_loop, train_pair, train_tgt, train_ids = load_or_process_data(
        Config.TRAIN_FILE, "train_data.npz", load_cached_data, is_test=False
    )
    # Val
    val_seq, val_loop, val_pair, val_tgt, val_ids = load_or_process_data(
        Config.VAL_FILE, "val_data.npz", load_cached_data, is_test=False
    )
    # Test
    test_seq, test_loop, test_pair, _, test_ids = load_or_process_data(
        Config.TEST_FILE, "test_data.npz", load_cached_data, is_test=True
    )

    # 2. Handle Debug Mode (Subsetting)
    if debug:
        # Slice to a small number
        subset_size = config.BATCH_SIZE * 2
        train_seq = train_seq[:subset_size]
        train_loop = train_loop[:subset_size]
        train_pair = train_pair[:subset_size]
        train_tgt = train_tgt[:subset_size]
        train_ids = train_ids[:subset_size]

        val_seq = val_seq[:subset_size]
        val_loop = val_loop[:subset_size]
        val_pair = val_pair[:subset_size]
        val_tgt = val_tgt[:subset_size]
        val_ids = val_ids[:subset_size]

        # Test usually small enough, but slice anyway
        test_seq = test_seq[:subset_size]
        test_loop = test_loop[:subset_size]
        test_pair = test_pair[:subset_size]
        test_ids = test_ids[:subset_size]

    # 3. Create Datasets
    train_dataset = RNADataset(train_seq, train_loop, train_pair, train_tgt, train_ids)
    val_dataset = RNADataset(val_seq, val_loop, val_pair, val_tgt, val_ids)
    test_dataset = RNADataset(test_seq, test_loop, test_pair, None, test_ids)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
