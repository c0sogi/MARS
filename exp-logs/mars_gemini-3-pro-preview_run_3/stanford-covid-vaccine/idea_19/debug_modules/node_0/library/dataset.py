import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# ==========================================
# Constants & Maps
# ==========================================
TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGUC")}
TOKEN2INT_STRUCT = {x: i for i, x in enumerate(".()")}
TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays of features and targets.
            mode (str): 'train', 'val', or 'test'.
        """
        self.features = data_dict["features"]
        self.pair_indices = data_dict["pair_indices"]
        self.distances = data_dict["distances"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = data_dict["targets"]
            self.masks = data_dict["masks"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features: (Seq_Len, 14)
        # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop)
        feat = torch.from_numpy(self.features[idx]).float()

        # Pair Indices: (Seq_Len,)
        # Indices of paired bases. Unpaired bases map to themselves.
        pair_idx = torch.from_numpy(self.pair_indices[idx]).long()

        # Relative Distances: (Seq_Len,)
        # Distance |i-j|. Unpaired bases have distance 0.
        dist = torch.from_numpy(self.distances[idx]).long()

        if self.mode == "test":
            # For test, we don't return targets
            return feat, pair_idx, dist

        # Targets: (Seq_Len, 5)
        # Padded with zeros beyond seq_scored
        targets = torch.from_numpy(self.targets[idx]).float()

        # Mask: (Seq_Len,)
        # 1.0 for scored positions, 0.0 otherwise
        mask = torch.from_numpy(self.masks[idx]).float()

        return feat, pair_idx, dist, targets, mask


def get_one_hot(sequence, token_map, length):
    """
    Generates one-hot encoding for a sequence.
    """
    arr = np.zeros((length, len(token_map)), dtype=np.float32)
    for i, char in enumerate(sequence):
        if i >= length:
            break
        if char in token_map:
            arr[i, token_map[char]] = 1.0
    return arr


def parse_structure_pairs(structure_str, length):
    """
    Parses dot-bracket structure to find paired indices and distances.
    Unpaired bases are mapped to themselves with distance 0.
    """
    pair_indices = np.arange(length, dtype=np.int16)  # Default to self
    distances = np.zeros(length, dtype=np.int16)  # Default to 0

    stack = []
    for i, char in enumerate(structure_str):
        if i >= length:
            break

        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Set pair connections
                pair_indices[i] = j
                pair_indices[j] = i

                # Set distances
                d = abs(i - j)
                # Clamp distance to avoid issues if larger than expected (though 107 fits in 128)
                d = min(d, Config.MAX_DISTANCE - 1)
                distances[i] = d
                distances[j] = d

    return pair_indices, distances


def process_dataframe(df, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    seq_scored = Config.SEQ_SCORED

    # Pre-allocate arrays
    # Features: (N, L, 4+3+7=14)
    features = np.zeros(
        (num_samples, seq_len, Config.NUM_NODE_FEATURES), dtype=np.float32
    )
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int16)
    distances = np.zeros((num_samples, seq_len), dtype=np.int16)

    # IDs
    ids = df["id"].values

    # Targets & Masks (only for train/val)
    targets = None
    masks = None
    if mode != "test":
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
        masks = np.zeros((num_samples, seq_len), dtype=np.float32)
        masks[:, :seq_scored] = 1.0

    print(f"Processing {num_samples} samples for {mode}...")

    for idx, row in df.iterrows():
        # 1. Features
        # Sequence (4)
        seq_oh = get_one_hot(row["sequence"], TOKEN2INT_SEQ, seq_len)
        # Structure (3)
        struct_oh = get_one_hot(row["structure"], TOKEN2INT_STRUCT, seq_len)
        # Loop Type (7)
        loop_oh = get_one_hot(row["predicted_loop_type"], TOKEN2INT_LOOP, seq_len)

        # Concatenate
        features[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Structural Context (Pairs & Distances)
        p_idx, dists = parse_structure_pairs(row["structure"], seq_len)
        pair_indices[idx] = p_idx
        distances[idx] = dists

        # 3. Targets (Train/Val only)
        if mode != "test":
            # Targets are provided as lists of length seq_scored (68)
            # We place them into the (107,) buffer
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure we handle cases where val_list might be shorter/longer or numpy array
                length = min(len(val_list), seq_scored)
                targets[idx, :length, t_i] = val_list[:length]

    data_dict = {
        "features": features,
        "pair_indices": pair_indices,
        "distances": distances,
        "ids": ids,
    }

    if mode != "test":
        data_dict["targets"] = targets
        data_dict["masks"] = masks

    return data_dict


def load_data(split="train", load_cached_data=True, debug=Config.DEBUG):
    """
    Loads data for a specific split. Handles caching.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npz cache.
        debug (bool): If True, loads a subset of data.

    Returns:
        RNADataset: The dataset object.
    """
    # Determine cache path
    if split == "train":
        base_path = Config.TRAIN_CACHE
        meta_path = Config.TRAIN_METADATA
    elif split == "val":
        base_path = Config.VAL_CACHE
        meta_path = Config.VAL_METADATA
    elif split == "test":
        base_path = Config.TEST_CACHE
        meta_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    # Adjust cache path for debug mode to avoid polluting full cache
    if debug:
        base_path = base_path.replace(".npy", "_debug.npy")

    # Use .npz extension for array storage
    cache_path = base_path.replace(".npy", ".npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        try:
            loaded = np.load(cache_path)
            # Convert NpzFile to dict
            data_dict = {k: loaded[k] for k in loaded.files}
            return RNADataset(data_dict, mode=split)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Metadata
    print(f"Loading metadata from {meta_path}...")
    df = pd.read_parquet(meta_path)

    if debug:
        df = df.head(Config.DEBUG_SAMPLES)
        print(f"Debug mode: Reduced {split} data to {len(df)} samples.")

    data_dict = process_dataframe(df, mode=split)

    # 3. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Saving processed {split} data to {cache_path}...")
    np.savez(cache_path, **data_dict)

    return RNADataset(data_dict, mode=split)
