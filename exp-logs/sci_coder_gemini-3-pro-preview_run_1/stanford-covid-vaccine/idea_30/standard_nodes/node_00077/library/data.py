import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Mappings
# =========================================================================
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# =========================================================================
# Dataset Class
# =========================================================================
class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing processed numpy arrays/lists.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data_dict
        self.mode = mode

    def __len__(self):
        return len(self.data["sequence"])

    def __getitem__(self, idx):
        # Inputs
        item = {
            "sequence": torch.tensor(self.data["sequence"][idx], dtype=torch.long),
            "loop_type": torch.tensor(self.data["loop_type"][idx], dtype=torch.long),
            "pair_dist": torch.tensor(self.data["pair_dist"][idx], dtype=torch.float32),
        }

        # Targets (only for train/val)
        if self.mode != "test":
            item["targets"] = torch.tensor(
                self.data["targets"][idx], dtype=torch.float32
            )
            item["mask"] = torch.tensor(self.data["mask"][idx], dtype=torch.float32)

        # Metadata (ID) - useful for submission file generation
        if "ids" in self.data:
            item["id"] = self.data["ids"][idx]

        return item


# =========================================================================
# Processing Functions
# =========================================================================
def parse_structure(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to calculate signed pair distances.

    Args:
        structure_str (str): Dot-bracket string (e.g., '..((..))..').
        seq_len (int): Length of the sequence.

    Returns:
        np.ndarray: Array of shape (seq_len,) containing signed distances.
                    If i is paired with j, val[i] = j - i.
                    If unpaired, val[i] = 0.
    """
    stack = []
    pair_dists = np.zeros(seq_len, dtype=np.float32)

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is upstream (smaller index), i is downstream (larger index)
                # Distance for j (opening): i - j (positive)
                # Distance for i (closing): j - i (negative)
                pair_dists[j] = i - j
                pair_dists[i] = j - i

    return pair_dists


def process_data(df, mode="train"):
    """
    Processes a pandas DataFrame into a dictionary of numpy arrays suitable for the Dataset.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays for inputs
    sequences = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_types = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_dists = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = []

    # Pre-allocate arrays for targets (if applicable)
    if mode != "test":
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    else:
        targets = None
        masks = None

    # Iterate over the dataframe
    # Using enumerate ensures we fill the numpy arrays at the correct 0..N-1 index
    for i, (_, row) in enumerate(df.iterrows()):
        ids.append(row["id"])

        # 1. Tokenize Sequence
        sequences[i] = [NUCLEOTIDE_MAP.get(c, 0) for c in row["sequence"]]

        # 2. Tokenize Loop Type
        # Default to 'X' (6) if unknown
        loop_types[i] = [LOOP_TYPE_MAP.get(c, 6) for c in row["predicted_loop_type"]]

        # 3. Parse Structure
        pair_dists[i] = parse_structure(row["structure"], seq_len)

        # 4. Handle Targets
        if mode != "test":
            scored_len = Config.SCORED_LEN

            # Fill targets for each column
            for t_i, col_name in enumerate(Config.TARGET_COLS):
                val_array = row[col_name]
                # The raw data provides lists of length `scored_len` (68)
                # We copy this into the first 68 positions of the target array
                length = len(val_array)
                targets[i, :length, t_i] = val_array

            # Create binary mask (1 for scored positions, 0 for unscored)
            masks[i, :scored_len] = 1.0

    # Construct result dictionary
    data_dict = {
        "ids": ids,
        "sequence": sequences,
        "loop_type": loop_types,
        "pair_dist": pair_dists,
    }

    if mode != "test":
        data_dict["targets"] = targets
        data_dict["mask"] = masks

    return data_dict


# =========================================================================
# Main Loader Function
# =========================================================================
def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles caching of processed data to speed up experiments.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.
        debug (bool): If True, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_data.pt")
    val_cache = os.path.join(Config.CACHE_DIR, "val_data.pt")
    test_cache = os.path.join(Config.CACHE_DIR, "test_data.pt")

    def load_or_process(source_path, cache_path, mode):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            data = torch.load(cache_path, weights_only=False)
        else:
            # 2. Process from scratch
            print(f"Processing {mode} data from {source_path}...")
            df = pd.read_parquet(source_path)
            data = process_data(df, mode=mode)
            print(f"Saving {mode} data to {cache_path}...")
            torch.save(data, cache_path)

        # 3. Handle Debug Slicing
        if debug:
            print(
                f"Debug mode: slicing {mode} data to {Config.DEBUG_SUBSET_SIZE} samples."
            )
            limit = Config.DEBUG_SUBSET_SIZE
            sliced_data = {}
            for k, v in data.items():
                if v is not None:
                    sliced_data[k] = v[:limit]
                else:
                    sliced_data[k] = None
            return sliced_data

        return data

    # Load Data Dictionaries
    train_data = load_or_process(Config.TRAIN_DATA_PATH, train_cache, "train")
    val_data = load_or_process(Config.VAL_DATA_PATH, val_cache, "val")
    test_data = load_or_process(Config.TEST_DATA_PATH, test_cache, "test")

    # Initialize Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
