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
            features (np.ndarray): Input features of shape (N, Input_Dim).
            targets (np.ndarray, optional): Targets of shape (N, Output_Dim).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
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


def _encode_sequence(sequence, token_map, vocab_size):
    """
    One-hot encodes a sequence string based on the provided token map.
    """
    indices = [token_map.get(char, 0) for char in sequence]
    return np.eye(vocab_size)[indices]


def _process_features(df):
    """
    Generates the flattened feature matrix from the dataframe.

    Processing steps:
    1. Iterate over each sample.
    2. One-hot encode sequence, structure, and predicted_loop_type.
    3. Concatenate these encodings along the channel axis.
    4. Flatten the result into a 1D vector.

    Returns:
        np.ndarray: Feature matrix of shape (N, 1498).
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Pre-allocate memory: (N, 107, 14)
    features = np.zeros((n_samples, seq_len, Config.CHANNELS_PER_POS), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        # Sequence (4 channels)
        seq_oh = _encode_sequence(
            row["sequence"], Config.TOKEN_MAP_SEQ, Config.VOCAB_SIZE_SEQ
        )

        # Structure (3 channels)
        struct_oh = _encode_sequence(
            row["structure"], Config.TOKEN_MAP_STRUCT, Config.VOCAB_SIZE_STRUCT
        )

        # Loop Type (7 channels)
        loop_oh = _encode_sequence(
            row["predicted_loop_type"], Config.TOKEN_MAP_LOOP, Config.VOCAB_SIZE_LOOP
        )

        # Concatenate: (107, 4) + (107, 3) + (107, 7) -> (107, 14)
        features[i] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

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

    # Filter training data for high signal-to-noise
    # Cite solution_lesson_node_00002: Improving data quality to help model distinguish signal from noise
    if split == "train":
        print("Filtering training data for SN_filter == 1...")
        df = df[df["SN_filter"] == 1].reset_index(drop=True)

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
