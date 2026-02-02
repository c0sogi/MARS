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
TOKEN_TO_INDEX = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_TO_INDEX = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def get_sinusoidal_encoding(positions, dim):
    """
    Generates sinusoidal encodings for signed distances.
    Args:
        positions (np.array): Array of signed distances of shape (seq_len,).
        dim (int): Embedding dimension.
    Returns:
        np.array: Positional encodings of shape (seq_len, dim).
    """
    # positions: (L,)
    pos_tensor = torch.from_numpy(positions).float().unsqueeze(1)  # (L, 1)

    # div_term: (dim/2,)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))

    pe = torch.zeros(len(positions), dim)
    pe[:, 0::2] = torch.sin(pos_tensor * div_term)
    pe[:, 1::2] = torch.cos(pos_tensor * div_term)

    return pe.numpy()


def parse_structure_distances(structure_str):
    """
    Parses a dot-bracket structure string and calculates signed distances for paired bases.
    Unpaired bases have a distance of 0.

    Args:
        structure_str (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        np.array: Array of signed distances of length len(structure_str).
    """
    n = len(structure_str)
    distances = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is upstream (smaller index), i is downstream (larger index)
                # Distance at j (opening): i - j (positive)
                # Distance at i (closing): j - i (negative)
                dist = i - j
                distances[j] = dist
                distances[i] = -dist

    return distances


def process_dataframe(df, mode="train"):
    """
    Processes the raw dataframe into numpy arrays suitable for training/inference.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    seq_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    loop_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    dist_embeddings = np.zeros(
        (num_samples, seq_len, Config.EMBED_DIM_DIST), dtype=np.float32
    )

    # Targets are only present in train/val
    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # Process each sample
    for idx, row in df.iterrows():
        # 1. Sequence Tokenization
        seq = row["sequence"]
        seq_indices[idx] = np.array([TOKEN_TO_INDEX.get(c, 0) for c in seq])

        # 2. Loop Type Tokenization
        loop = row["predicted_loop_type"]
        loop_indices[idx] = np.array([LOOP_TO_INDEX.get(c, 0) for c in loop])

        # 3. Structure Distance Embedding
        struct = row["structure"]
        distances = parse_structure_distances(struct)
        dist_embeddings[idx] = get_sinusoidal_encoding(distances, Config.EMBED_DIM_DIST)

        # 4. Targets
        if mode in ["train", "val"]:
            # Targets are lists of length 68. We need to pad them to 107.
            # The scored columns are defined in Config.TARGET_COLS
            for t_i, col_name in enumerate(Config.TARGET_COLS):
                val_list = row[col_name]
                # Assign the first 68 values
                length = len(val_list)
                targets[idx, :length, t_i] = val_list
                # Remaining positions are already 0.0

    # Store IDs for submission
    ids = df["id"].values

    data_dict = {
        "seq_indices": seq_indices,
        "loop_indices": loop_indices,
        "dist_embeddings": dist_embeddings,
        "ids": ids,
    }

    if targets is not None:
        data_dict["targets"] = targets

    return data_dict


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.seq_indices = data_dict["seq_indices"]
        self.loop_indices = data_dict["loop_indices"]
        self.dist_embeddings = data_dict["dist_embeddings"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode in ["train", "val"]:
            self.targets = data_dict["targets"]
        else:
            self.targets = None

        self.seq_scored = Config.PRED_LEN

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to tensors
        seq = torch.tensor(self.seq_indices[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_indices[idx], dtype=torch.long)
        dist = torch.tensor(self.dist_embeddings[idx], dtype=torch.float32)

        # Create mask for loss calculation (only first 68 positions are scored)
        # Shape: (seq_len,)
        mask = torch.zeros(Config.SEQ_LEN, dtype=torch.float32)
        mask[: self.seq_scored] = 1.0

        if self.mode in ["train", "val"]:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return seq, loop, dist, target, mask
        else:
            # For test, return dummy target
            dummy_target = torch.zeros(
                (Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32
            )
            return seq, loop, dist, dummy_target, mask, self.ids[idx]


# =========================================================================
# Data Loading & Caching Logic
# =========================================================================


def get_data(mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches it using npz.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The instantiated dataset.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            data_dict = {key: loaded[key] for key in loaded.files}
            return RNADataset(data_dict, mode=mode)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Determine source file
    if mode == "train":
        path = Config.TRAIN_DATA_PATH
    elif mode == "val":
        path = Config.VAL_DATA_PATH
    else:
        path = Config.TEST_DATA_PATH

    print(f"Processing {mode} data from: {path}")
    df = pd.read_parquet(path)

    # Debug mode: subset data
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLES)
        print(f"DEBUG MODE: Reduced {mode} data to {len(df)} samples.")

    # Process
    data_dict = process_dataframe(df, mode=mode)

    # Save to cache
    print(f"Saving {mode} data to cache: {cache_file}")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_file, **data_dict)

    return RNADataset(data_dict, mode=mode)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Train Loader
    train_dataset = get_data("train", load_cached_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val Loader
    val_dataset = get_data("val", load_cached_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Test Loader
    test_dataset = get_data("test", load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
