import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Mappings
# =============================================================================
NUC_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


# =============================================================================
# Helper Functions
# =============================================================================
def get_couples(structure):
    """
    Parses a dot-bracket structure string to generate pair indices and masks.

    Args:
        structure (str): Dot-bracket string (e.g., "((...))").

    Returns:
        tuple:
            - pair_indices (np.ndarray): Array of shape (Seq_Len,).
              If i is paired with j, pair_indices[i] = j.
              If i is unpaired, pair_indices[i] = i (self-loop, to be masked).
            - pair_mask (np.ndarray): Array of shape (Seq_Len,).
              1.0 if paired, 0.0 if unpaired.
    """
    seq_len = len(structure)
    pair_indices = np.arange(seq_len)  # Default to self
    pair_mask = np.zeros(seq_len, dtype=np.float32)

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


def one_hot_encode(sequence, structure, loop_type):
    """
    Generates the 14-channel one-hot encoded input feature.

    Channels:
    0-3: Nucleotide (A, G, C, U)
    4-6: Structure ((, ), .)
    7-13: Loop Type (S, M, I, B, H, E, X)

    Args:
        sequence (str): RNA sequence.
        structure (str): Dot-bracket structure.
        loop_type (str): Predicted loop type.

    Returns:
        np.ndarray: Shape (Seq_Len, 14).
    """
    seq_len = len(sequence)
    encoding = np.zeros((seq_len, 14), dtype=np.float32)

    for i in range(seq_len):
        # Nucleotide
        if sequence[i] in NUC_MAP:
            encoding[i, NUC_MAP[sequence[i]]] = 1.0

        # Structure
        if structure[i] in STRUCT_MAP:
            encoding[i, 4 + STRUCT_MAP[structure[i]]] = 1.0

        # Loop Type
        if loop_type[i] in LOOP_MAP:
            encoding[i, 7 + LOOP_MAP[loop_type[i]]] = 1.0

    return encoding


def process_dataframe(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for features, structural info, and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    features = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = []

    # Targets are only present in train/val
    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for idx, row in df.iterrows():
        # 1. Input Features & Structure
        seq = row["sequence"]
        struct = row["structure"]
        loop = row["predicted_loop_type"]

        # Ensure lengths match Config.SEQ_LEN (107)
        # The dataset guarantees 107, but safety check is good practice or just assume correct.

        features[idx] = one_hot_encode(seq, struct, loop)
        p_idx, p_mask = get_couples(struct)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask
        ids.append(row["id"])

        # 2. Targets (if available)
        if mode in ["train", "val"]:
            # Targets are lists of length seq_scored (68).
            # We pad them to 107 with zeros.
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                # Convert to numpy array
                val_arr = np.array(val_list, dtype=np.float32)
                # Assign to the first len(val_arr) positions
                length = len(val_arr)
                targets[idx, :length, t_i] = val_arr

    data_dict = {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "ids": np.array(ids),
    }

    if targets is not None:
        data_dict["targets"] = targets

    return data_dict


# =============================================================================
# Dataset Class
# =============================================================================
class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays.
            mode (str): 'train', 'val', or 'test'.
        """
        self.features = torch.tensor(data_dict["features"], dtype=torch.float32)
        self.pair_indices = torch.tensor(data_dict["pair_indices"], dtype=torch.long)
        self.pair_masks = torch.tensor(data_dict["pair_masks"], dtype=torch.float32)
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode in ["train", "val"]:
            self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns:
            tuple:
                - features (Tensor): (107, 14)
                - pair_indices (Tensor): (107,)
                - pair_masks (Tensor): (107,)
                - target (Tensor) or id (str): (107, 5) or ID string
        """
        feat = self.features[idx]
        p_idx = self.pair_indices[idx]
        p_mask = self.pair_masks[idx]

        if self.mode in ["train", "val"]:
            target = self.targets[idx]
            return feat, p_idx, p_mask, target
        else:
            # For test, we return the ID to construct the submission
            sample_id = self.ids[idx]
            return feat, p_idx, p_mask, sample_id


# =============================================================================
# Data Loading & Caching
# =============================================================================
def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for train, validation, and test sets.
    Handles caching of processed data to .npy files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "train": Config.TRAIN_CACHE_PATH,
        "val": Config.VAL_CACHE_PATH,
        "test": Config.TEST_CACHE_PATH,
    }

    metadata_paths = {
        "train": Config.TRAIN_METADATA_PATH,
        "val": Config.VAL_METADATA_PATH,
        "test": Config.TEST_METADATA_PATH,
    }

    datasets = {}

    for mode in ["train", "val", "test"]:
        cache_file = cache_paths[mode]
        meta_file = metadata_paths[mode]

        data_loaded = False
        data_dict = None

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_file):
            try:
                # Allow pickle=True because we are saving a dictionary of arrays
                data_dict = np.load(cache_file, allow_pickle=True).item()
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

        # 2. Process from scratch if needed
        if not data_loaded:
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"Metadata file not found: {meta_file}")

            df = pd.read_parquet(meta_file)

            # Debug mode: Subset data
            if Config.DEBUG:
                df = df.head(Config.DEBUG_SUBSET_SIZE)

            # Reset index to ensure alignment
            df = df.reset_index(drop=True)

            data_dict = process_dataframe(df, mode=mode)

            # Save to cache
            np.save(cache_file, data_dict)

        # 3. Create Dataset
        datasets[mode] = RNADataset(data_dict, mode=mode)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
