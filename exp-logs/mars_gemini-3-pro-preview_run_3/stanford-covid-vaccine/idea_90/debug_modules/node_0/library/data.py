import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# ==========================================
# Constants & Mappings
# ==========================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# ==========================================
# Helper Functions
# ==========================================
def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate adjacency indices and masks.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        indices (np.ndarray): Array of shape (L,) where indices[i] is the index of the base paired with i.
                              If unpaired, indices[i] = 0 (safe index for gather).
        mask (np.ndarray): Array of shape (L,) where mask[i] = 1 if paired, 0 if unpaired.
    """
    length = len(structure)
    indices = np.zeros(length, dtype=np.int64)
    mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return indices, mask


def one_hot_encode(seq, mapping, length, num_channels):
    """
    One-hot encodes a string sequence.
    """
    encoding = np.zeros((length, num_channels), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def process_data(df, mode="train"):
    """
    Process a pandas DataFrame into numpy arrays for the dataset.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary of numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Input channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    targets = None
    if mode != "test":
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    for idx, row in df.iterrows():
        # 1. Inputs
        # Sequence (4)
        seq_enc = one_hot_encode(row["sequence"], SEQ_MAP, seq_len, 4)
        # Structure (3)
        struct_enc = one_hot_encode(row["structure"], STRUCT_MAP, seq_len, 3)
        # Loop Type (7)
        loop_enc = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len, 7)

        inputs[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 2. Adjacency
        p_idx, p_mask = get_structure_adj(row["structure"])
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets (if not test)
        if mode != "test":
            # Targets are lists of length seq_scored (68)
            # We pad them to 107 with zeros to maintain tensor shape consistency.
            # The loss function handles slicing to the correct scored length.
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure it's a list or array
                if isinstance(val_list, (list, np.ndarray)):
                    length = min(len(val_list), seq_len)
                    targets[idx, :length, t_i] = val_list

    data_dict = {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "ids": ids,
    }
    if targets is not None:
        data_dict["targets"] = targets

    return data_dict


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing processed numpy arrays.
            mode (str): 'train', 'val', or 'test'.
        """
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (107, 14)
        # pair_indices: (107,)
        # pair_masks: (107,)

        sample = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
            "pair_masks": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
            "ids": self.ids[idx],
        }

        if self.mode != "test":
            # targets: (107, 5)
            sample["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return sample


# ==========================================
# Data Loading & Caching
# ==========================================
def load_data(mode="train", load_cached_data=True):
    """
    Loads data for the specified mode. Uses caching to speed up subsequent loads.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The dataset object.
    """
    # Determine paths
    if mode == "train":
        parquet_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif mode == "val":
        parquet_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif mode == "test":
        parquet_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data_dict = {key: loaded[key] for key in loaded.files}
            return RNADataset(data_dict, mode=mode)
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

    # Process from scratch
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Debug subset
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    data_dict = process_data(df, mode=mode)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data_dict)

    return RNADataset(data_dict, mode=mode)


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test sets.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = load_data("train")
    val_dataset = load_data("val")
    test_dataset = load_data("test")

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
