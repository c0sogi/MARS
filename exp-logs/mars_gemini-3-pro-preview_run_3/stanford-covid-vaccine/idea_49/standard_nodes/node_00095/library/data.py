import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# ==========================================
# Constants & Dictionaries
# ==========================================
TOKEN_DICT = {
    "sequence": {"A": 0, "G": 1, "C": 2, "U": 3},
    "structure": {"(": 0, ")": 1, ".": 2},
    "predicted_loop_type": {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6},
}


# ==========================================
# Helper Functions
# ==========================================
def get_couples(structure):
    """
    Generates an adjacency array for the RNA secondary structure.

    Args:
        structure (str): Dot-bracket notation string.

    Returns:
        np.ndarray: Array of shape (Seq_Len,) where arr[i] = j if i is paired with j.
                    If i is unpaired, arr[i] = -1.
    """
    seq_len = len(structure)
    pairing = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairing[i] = j
                pairing[j] = i

    return pairing


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, features, bppm_indices, targets, ids):
        """
        Args:
            features (np.ndarray): Input features (N, 107, 14).
            bppm_indices (np.ndarray): Pairing indices (N, 107).
            targets (np.ndarray): Target values (N, 68, 5).
            ids (np.ndarray): Sample IDs (N,).
        """
        self.features = features
        self.bppm_indices = bppm_indices
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert inputs to float32 tensor
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        # Convert indices to long tensor
        bppm = torch.tensor(self.bppm_indices[idx], dtype=torch.long)

        # Convert targets to float32 tensor
        y = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Get ID
        sample_id = self.ids[idx]

        return x, bppm, y, sample_id


# ==========================================
# Data Processing
# ==========================================
def preprocess_data(df, mode="train"):
    """
    Converts DataFrame into numpy arrays for the dataset.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        tuple: (features, bppm_indices, targets, ids)
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # 1. Features: (N, 107, 14)
    # Channels: 0-3 (Seq), 4-6 (Struct), 7-13 (Loop)
    features = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)

    # 2. Adjacency Indices: (N, 107)
    bppm_indices = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Pre-fetch dictionaries
    seq_map = TOKEN_DICT["sequence"]
    struct_map = TOKEN_DICT["structure"]
    loop_map = TOKEN_DICT["predicted_loop_type"]

    # Iterate efficiently
    # Using itertuples is faster than iterrows
    for i, row in enumerate(df.itertuples(index=False)):
        # Extract strings
        # Note: Depending on pandas version/loading, columns are attributes
        # We assume standard column names from metadata
        sequence = row.sequence
        structure = row.structure
        loop_type = row.predicted_loop_type

        # Encode Sequence
        for j, char in enumerate(sequence):
            if char in seq_map:
                features[i, j, seq_map[char]] = 1.0

        # Encode Structure
        for j, char in enumerate(structure):
            if char in struct_map:
                features[i, j, 4 + struct_map[char]] = 1.0

        # Encode Loop Type
        for j, char in enumerate(loop_type):
            if char in loop_map:
                features[i, j, 7 + loop_map[char]] = 1.0

        # Compute Adjacency
        bppm_indices[i] = get_couples(structure)

    # 3. Targets: (N, 68, 5)
    target_len = Config.SEQ_SCORED
    num_targets = Config.NUM_TARGETS
    targets = np.zeros((num_samples, target_len, num_targets), dtype=np.float32)

    if mode in ["train", "val"]:
        # Extract target lists
        target_cols = Config.TARGET_COLS
        for t_idx, col_name in enumerate(target_cols):
            # df[col_name] contains lists. Stack them into a 2D array.
            # We use np.vstack on the Series values which are lists
            col_data = np.vstack(df[col_name].values)
            targets[:, :, t_idx] = col_data

    # 4. IDs
    ids = df["id"].values

    return features, bppm_indices, targets, ids


def get_dataloaders(
    debug=Config.DEBUG, load_cached_data=True, batch_size=None, num_workers=None
):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles caching and preprocessing.

    Args:
        debug (bool): If True, use a small subset of data.
        load_cached_data (bool): If True, attempt to load from disk cache.
        batch_size (int): Batch size override.
        num_workers (int): Num workers override.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set defaults if None
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    loaders = {}
    modes = ["train", "val", "test"]

    # Map modes to file paths
    # We use .npz for cache to avoid pickle issues with np.save
    file_map = {
        "train": (Config.TRAIN_FILE, Config.TRAIN_CACHE),
        "val": (Config.VAL_FILE, Config.VAL_CACHE),
        "test": (Config.TEST_FILE, Config.TEST_CACHE),
    }

    for mode in modes:
        parquet_path, cache_path_base = file_map[mode]

        # Adjust cache path to force .npz extension for np.savez compatibility
        # If config path ends in .npy, we replace it or append .npz
        if cache_path_base.endswith(".npy"):
            cache_path = cache_path_base.replace(".npy", ".npz")
        else:
            cache_path = cache_path_base + ".npz"

        data_loaded = False

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # Load uncompressed or compressed npz
                data = np.load(cache_path)
                features = data["features"]
                bppm_indices = data["bppm_indices"]
                targets = data["targets"]
                ids = data["ids"]
                data_loaded = True
            except Exception as e:
                print(f"[{mode}] Cache load failed: {e}. Reprocessing...")
                data_loaded = False

        # 2. Process from Scratch if needed
        if not data_loaded:
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Source file {parquet_path} not found.")

            # Load metadata
            df = pd.read_parquet(parquet_path)

            # Debug subset
            if debug and mode == "train":
                df = df.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

            # Preprocess
            features, bppm_indices, targets, ids = preprocess_data(df, mode=mode)

            # Save to cache (only if not debugging to avoid corrupting cache with partial data)
            if not debug:
                np.savez(
                    cache_path,
                    features=features,
                    bppm_indices=bppm_indices,
                    targets=targets,
                    ids=ids,
                )

        # 3. Create Dataset and Loader
        dataset = RNADataset(features, bppm_indices, targets, ids)

        is_train = mode == "train"

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=is_train,  # Drop last batch in training to maintain batch statistics
        )

        loaders[mode] = loader

    return loaders["train"], loaders["val"], loaders["test"]
