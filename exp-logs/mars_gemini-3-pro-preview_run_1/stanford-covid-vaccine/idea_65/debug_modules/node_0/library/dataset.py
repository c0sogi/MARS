import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Wraps processed numpy arrays and returns tensors.
    """

    def __init__(self, data):
        """
        Args:
            data (dict): Dictionary containing processed numpy arrays:
                         'sequences', 'loops', 'distances', 'targets', 'ids'.
        """
        self.sequences = data["sequences"]
        self.loops = data["loops"]
        self.distances = data["distances"]
        self.targets = data["targets"]
        self.ids = data["ids"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to appropriate torch tensors
        # Sequence and Loop are integer indices (Long)
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)

        # Distance encoding is float features
        dist = torch.tensor(self.distances[idx], dtype=torch.float32)

        # Targets are float values
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return seq, loop, dist, target


def get_sinusoidal_encoding(distances, dim):
    """
    Computes fixed sinusoidal encodings for signed distances.

    Args:
        distances (np.ndarray): Array of distances of shape (L,).
        dim (int): Embedding dimension.

    Returns:
        np.ndarray: Sinusoidal embeddings of shape (L, dim).
    """
    seq_len = len(distances)
    pe = np.zeros((seq_len, dim), dtype=np.float32)

    # Expand distances to (L, 1)
    position = distances[:, np.newaxis]

    # Compute division term: 10000^(2i/dim)
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))

    # Apply sin to even indices, cos to odd indices
    # sin(-x) = -sin(x) preserves sign information
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def parse_structure_to_distance(structure_str):
    """
    Parses dot-bracket structure to compute signed pairing distances.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of signed distances (i - j) for paired bases, 0 for unpaired.
    """
    length = len(structure_str)
    pairs = np.full(length, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i

    distances = np.zeros(length, dtype=int)
    for i in range(length):
        if pairs[i] != -1:
            # Signed distance: i - j
            # If i < j (upstream), distance is negative
            # If i > j (downstream), distance is positive
            distances[i] = i - pairs[i]

    return distances


def process_data_frame(df, mode):
    """
    Processes a pandas DataFrame into numpy arrays for the model.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    pred_len = Config.PRED_LEN

    # Initialize arrays
    sequences = np.zeros((num_samples, seq_len), dtype=np.int32)
    loops = np.zeros((num_samples, seq_len), dtype=np.int32)
    distances = np.zeros(
        (num_samples, seq_len, Config.EMBED_DIM_DIST), dtype=np.float32
    )

    # Initialize targets
    # Train/Val: (N, 68, 3)
    # Test: (N, 68, 3) filled with zeros
    targets = np.zeros(
        (num_samples, pred_len, len(Config.TARGET_COLS)), dtype=np.float32
    )

    ids = df["id"].values

    # Extract raw data
    raw_seqs = df["sequence"].values
    raw_loops = df["predicted_loop_type"].values
    raw_structs = df["structure"].values

    # Pre-fetch target columns if not test mode
    target_data = {}
    if mode != "test":
        for col in Config.TARGET_COLS:
            # These columns contain lists/arrays in the parquet file
            target_data[col] = df[col].values

    print(f"Processing {num_samples} samples for {mode}...")

    for i in range(num_samples):
        # 1. Sequence Tokenization
        sequences[i] = [seq_map.get(c, 0) for c in raw_seqs[i]]

        # 2. Loop Type Tokenization
        loops[i] = [loop_map.get(c, 0) for c in raw_loops[i]]

        # 3. Structure Distance Encoding
        dist_vals = parse_structure_to_distance(raw_structs[i])
        distances[i] = get_sinusoidal_encoding(dist_vals, Config.EMBED_DIM_DIST)

        # 4. Targets
        if mode != "test":
            for t_idx, col in enumerate(Config.TARGET_COLS):
                # Each row in target_data[col] is a list/array of length 68
                val = target_data[col][i]
                # Ensure it matches pred_len
                if len(val) >= pred_len:
                    targets[i, :, t_idx] = val[:pred_len]
                else:
                    # Pad if necessary (should not happen based on dataset spec)
                    targets[i, : len(val), t_idx] = val
        else:
            # Test mode: targets remain 0
            pass

    return {
        "sequences": sequences,
        "loops": loops,
        "distances": distances,
        "targets": targets,
        "ids": ids,
    }


def get_data(mode="train", load_cached_data=True):
    """
    Retrieves the dataset for the specified mode.
    Implements caching logic using .npz files.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary of processed numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            # Convert NpzFile object to dict to keep it in memory after closing
            data = {key: loaded[key] for key in loaded.files}
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from source...")

    if mode == "train":
        file_path = Config.TRAIN_PATH
    elif mode == "val":
        file_path = Config.VAL_PATH
    elif mode == "test":
        file_path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_parquet(file_path)

    processed_data = process_data_frame(df, mode)

    # 3. Save to cache
    print(f"Saving {mode} data to {cache_path}...")
    np.savez_compressed(cache_path, **processed_data)

    return processed_data
