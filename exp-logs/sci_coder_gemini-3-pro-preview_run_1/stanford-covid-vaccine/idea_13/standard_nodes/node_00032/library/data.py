import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Helper Functions
# =========================================================================


def get_pair_distances(structure: str) -> np.ndarray:
    """
    Parses the dot-bracket structure to calculate signed distances.
    Returns a numpy array of shape (seq_len,) where:
      - value is (j - i) if base i is paired with base j.
      - value is 0 if base i is unpaired.
    """
    n = len(structure)
    distances = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i is current (closing), j is previous (opening)
                # For j: pair is i, dist = i - j (positive)
                # For i: pair is j, dist = j - i (negative)
                distances[j] = i - j
                distances[i] = j - i

    return distances


def sequence_to_indices(sequence: str) -> np.ndarray:
    """
    Tokenizes sequence into character indices.
    Maps A,G,C,U to 1..4. 0 is padding/unknown.
    Cite solution_lesson_node_00028: Atomic tokenization > K-mers for small data.
    """
    base_map = {"A": 1, "G": 2, "C": 3, "U": 4}
    n = len(sequence)
    indices = np.zeros(n, dtype=np.int64)
    for i, char in enumerate(sequence):
        indices[i] = base_map.get(char, 0)
    return indices


def encode_loop_types(loop_str: str) -> np.ndarray:
    """
    Encodes loop type string into indices.
    """
    # Map: B, E, H, I, M, S, X -> 1..7
    loop_map = {"B": 1, "E": 2, "H": 3, "I": 4, "M": 5, "S": 6, "X": 7}
    n = len(loop_str)
    indices = np.zeros(n, dtype=np.int64)
    for i, char in enumerate(loop_str):
        indices[i] = loop_map.get(char, 0)  # 0 for unknown
    return indices


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing preprocessed arrays.
            is_test (bool): Whether this is the test set (no targets).
        """
        self.seqs = data_dict["seqs"]
        self.pair_dists = data_dict["pair_dists"]
        self.loop_types = data_dict["loop_types"]
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        pair_dist = torch.tensor(self.pair_dists[idx], dtype=torch.float32)
        loop_type = torch.tensor(self.loop_types[idx], dtype=torch.long)

        # Targets
        if self.is_test:
            # Create dummy targets for consistency in pipeline
            target = torch.zeros(
                (Config.SEQ_LENGTH, Config.NUM_TARGETS), dtype=torch.float32
            )
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return {
            "seq_inputs": seq,
            "pair_dists": pair_dist,
            "loop_types": loop_type,
            "targets": target,
        }


# =========================================================================
# Data Processing & Loading
# =========================================================================


def process_dataframe(df, is_test=False):
    """
    Extracts features and targets from a dataframe.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    all_seqs = np.zeros((num_samples, seq_len), dtype=np.int64)
    all_pair_dists = np.zeros((num_samples, seq_len), dtype=np.float32)
    all_loop_types = np.zeros((num_samples, seq_len), dtype=np.int64)

    if not is_test:
        all_targets = np.zeros(
            (num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32
        )
    else:
        all_targets = None

    print(f"Processing {num_samples} samples...")

    for idx, row in df.iterrows():
        # 1. Features
        all_seqs[idx] = sequence_to_indices(row["sequence"])
        all_pair_dists[idx] = get_pair_distances(row["structure"])
        all_loop_types[idx] = encode_loop_types(row["predicted_loop_type"])

        # 2. Targets (if available)
        if not is_test:
            # Targets are provided as lists of length seq_scored (68)
            # We need to pad them to seq_len (107)
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure it's a list or array
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    all_targets[idx, :length, t_i] = val_list
                else:
                    # Handle potential parsing issues or missing data
                    pass

    return {
        "seqs": all_seqs,
        "pair_dists": all_pair_dists,
        "loop_types": all_loop_types,
        "targets": all_targets,
    }


def get_dataloaders(load_cached_data=True, batch_size=None):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching logic.
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        "train": os.path.join(cache_dir, "train_data.pt"),
        "val": os.path.join(cache_dir, "val_data.pt"),
        "test": os.path.join(cache_dir, "test_data.pt"),
    }

    datasets = {}

    # Helper to load or process
    def load_or_process(split_name, parquet_path, is_test):
        cache_path = cache_files[split_name]

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split_name} data from {cache_path}...")
            data_dict = torch.load(cache_path, weights_only=False)
        else:
            print(f"Processing {split_name} data from {parquet_path}...")
            df = pd.read_parquet(parquet_path)
            data_dict = process_dataframe(df, is_test=is_test)
            print(f"Saving {split_name} data to cache...")
            torch.save(data_dict, cache_path)

        return RNADataset(data_dict, is_test=is_test)

    # 1. Train
    train_dataset = load_or_process("train", Config.TRAIN_DATA_PATH, is_test=False)

    # 2. Val
    val_dataset = load_or_process("val", Config.VAL_DATA_PATH, is_test=False)

    # 3. Test
    test_dataset = load_or_process("test", Config.TEST_DATA_PATH, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
