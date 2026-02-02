import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, features, targets=None):
        """
        Args:
            features (np.ndarray): Input features of shape (N, Seq_Len, 3).
            targets (np.ndarray, optional): Targets of shape (N, Seq_Scored, 5).
        """
        # Features are indices (integers) for embeddings
        self.features = torch.tensor(features, dtype=torch.long)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


def _encode_sequence(sequence, token_map):
    """
    Encodes a sequence string into indices based on the provided token map.
    """
    return [token_map.get(char, 0) for char in sequence]


def _process_features(df):
    """
    Generates the feature matrix of indices from the dataframe.

    Processing steps:
    1. Iterate over each sample.
    2. Convert sequence, structure, and loop type to integer indices.
    3. Stack them to form (N, Seq_Len, 3).

    Returns:
        np.ndarray: Feature matrix of shape (N, 107, 3).
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Pre-allocate memory: (N, 107, 3)
    # 3 channels: Sequence Index, Structure Index, Loop Type Index
    features = np.zeros((n_samples, seq_len, 3), dtype=np.int32)

    for i, (_, row) in enumerate(df.iterrows()):
        # Sequence
        features[i, :, 0] = _encode_sequence(row["sequence"], Config.TOKEN_MAP_SEQ)

        # Structure
        features[i, :, 1] = _encode_sequence(row["structure"], Config.TOKEN_MAP_STRUCT)

        # Loop Type
        features[i, :, 2] = _encode_sequence(
            row["predicted_loop_type"], Config.TOKEN_MAP_LOOP
        )

    return features


def _process_targets(df):
    """
    Generates the flattened target matrix from the dataframe.

    Processing steps:
    1. Iterate over each sample.
    2. Extract the 5 target lists (length 68 each).
    3. Stack them to form a (68, 5) matrix per sample.
    4. Flatten into a 1D vector.

    Returns:
        np.ndarray: Target matrix of shape (N, 340).
    """
    n_samples = len(df)
    seq_scored = Config.SEQ_SCORED
    num_targets = Config.NUM_TARGETS

    # Pre-allocate: (N, 68, 5)
    targets = np.zeros((n_samples, seq_scored, num_targets), dtype=np.float32)

    target_cols = Config.TARGET_COLS

    for i, (_, row) in enumerate(df.iterrows()):
        for t_idx, col in enumerate(target_cols):
            # The parquet loader preserves lists/arrays
            val = row[col]
            targets[i, :, t_idx] = val

    return targets


def get_data(split="train", load_cached_data=True):
    """
    Loads data, processes it into features/targets, and caches the result.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from .npy cache first.

    Returns:
        tuple: (features, targets) for 'train'/'val', or (features, None) for 'test'.
    """
    # Determine file paths based on split
    if split == "train":
        data_path = Config.TRAIN_DATA_PATH
        cache_feat_path = Config.TRAIN_CACHE_PATH
        cache_target_path = Config.TRAIN_TARGETS_PATH
    elif split == "val":
        data_path = Config.VAL_DATA_PATH
        cache_feat_path = Config.VAL_CACHE_PATH
        cache_target_path = Config.VAL_TARGETS_PATH
    elif split == "test":
        data_path = Config.TEST_DATA_PATH
        cache_feat_path = Config.TEST_CACHE_PATH
        cache_target_path = None
    else:
        raise ValueError(f"Invalid split: {split}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try loading from cache
    if load_cached_data:
        feat_exists = os.path.exists(cache_feat_path)
        target_exists = (cache_target_path is None) or os.path.exists(cache_target_path)

        if feat_exists and target_exists:
            print(f"Loading {split} data from cache...")
            features = np.load(cache_feat_path)
            targets = np.load(cache_target_path) if cache_target_path else None
            return features, targets

    # 2. Process from scratch
    print(f"Processing {split} data from source...")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Source data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    # Generate features
    features = _process_features(df)

    # Generate targets (if applicable)
    targets = None
    if split != "test":
        targets = _process_targets(df)

    # 3. Save to cache
    np.save(cache_feat_path, features)
    if targets is not None and cache_target_path:
        np.save(cache_target_path, targets)

    return features, targets
