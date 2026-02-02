import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Token Maps
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCTURE_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_pair_index_and_mask(structure):
    """
    Parses a dot-bracket structure string to generate pair indices and a mask.

    Args:
        structure (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        pair_indices (np.ndarray): Array of shape (L,) where pair_indices[i] is the index
                                   of the base paired with i. If unpaired, defaults to i.
        pair_mask (np.ndarray): Array of shape (L,) where 1 indicates paired, 0 unpaired.
    """
    length = len(structure)
    pair_indices = np.arange(
        length
    )  # Default to self-index for unpaired to ensure valid gather
    pair_mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0

    return pair_indices, pair_mask


def one_hot_encode(sequence, token_map, num_classes):
    """
    One-hot encodes a sequence string based on a token map.

    Args:
        sequence (str): Input string.
        token_map (dict): Mapping from character to integer index.
        num_classes (int): Total number of classes.

    Returns:
        np.ndarray: One-hot encoded array of shape (L, num_classes).
    """
    length = len(sequence)
    encoding = np.zeros((length, num_classes), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in token_map:
            encoding[i, token_map[char]] = 1.0
    return encoding


def process_dataframe(df, is_test=False):
    """
    Processes a pandas DataFrame into numpy arrays suitable for the model.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): Whether processing test data (no targets).

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize containers
    # Features: (N, L, 14) -> 4 (Seq) + 3 (Struct) + 7 (Loop)
    features = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: (N, L, 5)
    # Even if seq_scored is 68, we pad to 107 to match sequence length for simplicity in Dataset
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    # Track IDs
    ids = df["id"].values

    for idx, row in df.iterrows():
        # 1. Parse Inputs
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Ensure lengths match Config.SEQ_LEN (107)
        # In case of any discrepancy, we truncate or pad (though data should be clean)
        current_len = len(seq)

        # One-Hot Encoding
        # Channel 0-3: Nucleotide
        features[idx, :current_len, 0:4] = one_hot_encode(seq, NUCLEOTIDE_MAP, 4)
        # Channel 4-6: Structure
        features[idx, :current_len, 4:7] = one_hot_encode(struct, STRUCTURE_MAP, 3)
        # Channel 7-13: Loop Type
        features[idx, :current_len, 7:14] = one_hot_encode(loop, LOOP_TYPE_MAP, 7)

        # Structure Pairing
        p_idx, p_mask = get_pair_index_and_mask(struct)
        pair_indices[idx, :current_len] = p_idx
        pair_masks[idx, :current_len] = p_mask

        # 2. Parse Targets (if not test)
        if not is_test:
            # Targets are provided as lists of length seq_scored (68)
            # We place them into the (107, 5) array.
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            target_cols = Config.TARGET_COLS
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                # val_list is a list or array. Length is seq_scored.
                if isinstance(val_list, (list, np.ndarray)):
                    length_t = len(val_list)
                    targets[idx, :length_t, t_i] = val_list
                else:
                    # Fallback for unexpected format
                    pass

    return {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict, is_test=False):
        """
        Args:
            data_dict (dict): Dictionary containing processed numpy arrays.
            is_test (bool): Flag indicating if this is a test set (dummy targets).
        """
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]
        self.is_test = is_test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert to torch tensors
        features = torch.from_numpy(self.features[idx])
        pair_indices = torch.from_numpy(self.pair_indices[idx])
        pair_masks = torch.from_numpy(self.pair_masks[idx])
        targets = torch.from_numpy(self.targets[idx])

        # For test set, targets are zeros, which is fine as they are ignored.

        return {
            "features": features,  # (107, 14)
            "pair_indices": pair_indices,  # (107,)
            "pair_masks": pair_masks,  # (107,)
            "targets": targets,  # (107, 5)
            "id": self.ids[idx],
        }


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Implements caching to avoid re-processing data on every run.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define Cache Paths
    # We include DEBUG in the filename to avoid mixing full and debug datasets
    suffix = "_debug" if Config.DEBUG else ""
    train_cache = os.path.join(Config.CACHE_DIR, f"train_data{suffix}.npz")
    val_cache = os.path.join(Config.CACHE_DIR, f"val_data{suffix}.npz")
    test_cache = os.path.join(Config.CACHE_DIR, f"test_data{suffix}.npz")

    # Helper to load or process
    def load_or_process(path_parquet, path_cache, is_test=False):
        data_dict = None

        # Try Loading Cache
        if load_cached_data and os.path.exists(path_cache):
            try:
                loaded = np.load(path_cache, allow_pickle=True)
                data_dict = {
                    "features": loaded["features"],
                    "pair_indices": loaded["pair_indices"],
                    "pair_masks": loaded["pair_masks"],
                    "targets": loaded["targets"],
                    "ids": loaded["ids"],
                }
                print(f"Loaded cached data from {path_cache}")
            except Exception as e:
                print(f"Failed to load cache {path_cache}: {e}")
                data_dict = None

        # Process if not loaded
        if data_dict is None:
            print(f"Processing data from {path_parquet}...")
            df = pd.read_parquet(path_parquet)

            # Debug Subsampling
            if Config.DEBUG:
                df = df.head(Config.DEBUG_SUBSET_SIZE)

            data_dict = process_dataframe(df, is_test=is_test)

            # Save to Cache
            np.savez_compressed(
                path_cache,
                features=data_dict["features"],
                pair_indices=data_dict["pair_indices"],
                pair_masks=data_dict["pair_masks"],
                targets=data_dict["targets"],
                ids=data_dict["ids"],
            )
            print(f"Saved processed data to {path_cache}")

        return data_dict

    # 1. Load Data
    train_data = load_or_process(Config.TRAIN_DATA_PATH, train_cache, is_test=False)
    val_data = load_or_process(Config.VAL_DATA_PATH, val_cache, is_test=False)
    test_data = load_or_process(Config.TEST_DATA_PATH, test_cache, is_test=True)

    # 2. Create Datasets
    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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
