import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Tokenization Maps
SEQ_MAP = {c: i + 1 for i, c in enumerate(["A", "G", "C", "U"])}
LOOP_MAP = {c: i + 1 for i, c in enumerate(["S", "M", "I", "B", "H", "E", "X"])}


def parse_structure_to_distance(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to compute signed pairing distances.
    Returns a float array where:
      - Unpaired bases (.) are 0.0
      - Paired bases are (partner_index - current_index)
    """
    n = len(structure_str)
    # Ensure consistency with seq_len
    if n != seq_len:
        # In case of mismatch, we truncate or pad, but data should be consistent.
        pass

    partners = [-1] * n
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                partners[i] = j
                partners[j] = i

    distances = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if partners[i] != -1:
            distances[i] = float(partners[i] - i)

    return distances


def tokenize_sequence(seq_str, token_map, seq_len):
    """Converts a string sequence to an integer array based on the map."""
    arr = np.zeros(seq_len, dtype=np.int64)
    for i, char in enumerate(seq_str[:seq_len]):
        arr[i] = token_map.get(char, 0)
    return arr


class RNADataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays for 'seq', 'loop', 'dist', 'targets', 'ids'.
        """
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.dist = data_dict["dist"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.seq[idx], dtype=torch.long)
        loop = torch.tensor(self.loop[idx], dtype=torch.long)
        dist = torch.tensor(self.dist[idx], dtype=torch.float32)

        # Targets
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        # ID
        sample_id = self.ids[idx]

        return {
            "seq": seq,
            "loop": loop,
            "dist": dist,
            "targets": targets,
            "id": sample_id,
        }


def process_dataframe(df, mode, seq_len=107):
    """
    Process a raw dataframe into numpy arrays.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.
        seq_len (int): Sequence length.

    Returns:
        dict: Dictionary of numpy arrays.
    """
    num_samples = len(df)

    # Pre-allocate arrays
    seq_arr = np.zeros((num_samples, seq_len), dtype=np.int64)
    loop_arr = np.zeros((num_samples, seq_len), dtype=np.int64)
    dist_arr = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: (N, seq_len, 3)
    # We pad targets to seq_len even though only first 68 are valid/scored.
    target_arr = np.zeros(
        (num_samples, seq_len, len(Config.TARGET_COLS)), dtype=np.float32
    )

    ids = df["id"].values

    # Extract raw data
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    # Check if targets exist
    has_targets = all(col in df.columns for col in Config.TARGET_COLS)

    for i in range(num_samples):
        # 1. Sequence
        seq_arr[i] = tokenize_sequence(sequences[i], SEQ_MAP, seq_len)

        # 2. Loop Type
        loop_arr[i] = tokenize_sequence(loops[i], LOOP_MAP, seq_len)

        # 3. Structure Distance
        dist_arr[i] = parse_structure_to_distance(structures[i], seq_len)

        # 4. Targets
        if has_targets and mode != "test":
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = df.iloc[i][col]
                # val_list is a list or array of floats. Length usually 68.
                # Copy into the padded array
                length = min(len(val_list), seq_len)
                target_arr[i, :length, t_idx] = val_list[:length]
        else:
            # Test mode or missing targets: keep as zeros
            pass

    return {
        "seq": seq_arr,
        "loop": loop_arr,
        "dist": dist_arr,
        "targets": target_arr,
        "ids": ids,
    }


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Prepares and returns DataLoaders for train, val, and test sets.
    Handles caching of processed numpy arrays to avoid re-processing.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.
        debug_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_files = {
        "train": os.path.join(Config.WORKING_DIR, "train_data.npz"),
        "val": os.path.join(Config.WORKING_DIR, "val_data.npz"),
        "test": os.path.join(Config.WORKING_DIR, "test_data.npz"),
    }

    datasets = {}
    modes = ["train", "val", "test"]
    files = {
        "train": Config.TRAIN_FILE,
        "val": Config.VAL_FILE,
        "test": Config.TEST_FILE,
    }

    for mode in modes:
        cache_path = cache_files[mode]
        data_dict = None

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {k: loaded[k] for k in loaded.files}
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Reprocessing...")
                data_dict = None

        # 2. Process if not loaded
        if data_dict is None:
            print(f"Processing {mode} data from {files[mode]}...")
            if not os.path.exists(files[mode]):
                raise FileNotFoundError(f"Source file {files[mode]} not found.")

            df = pd.read_parquet(files[mode])

            # Debugging: subset
            if debug_size is not None:
                df = df.iloc[:debug_size]

            data_dict = process_dataframe(df, mode, seq_len=Config.SEQ_LEN)

            # Save to cache
            print(f"Saving {mode} data to cache...")
            np.savez_compressed(
                cache_path,
                seq=data_dict["seq"],
                loop=data_dict["loop"],
                dist=data_dict["dist"],
                targets=data_dict["targets"],
                ids=data_dict["ids"],
            )

        # 3. Create Dataset
        datasets[mode] = RNADataset(data_dict)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
