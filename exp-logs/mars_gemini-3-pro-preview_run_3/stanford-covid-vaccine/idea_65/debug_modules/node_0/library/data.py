import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_md5_hash


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Delivers input features, structural adjacency info, and targets.
    """

    def __init__(self, features, pair_indices, pair_masks, targets=None):
        self.features = features
        self.pair_indices = pair_indices
        self.pair_masks = pair_masks
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        # Features: (Seq_Len, 14)
        feat = torch.tensor(self.features[idx], dtype=torch.float32)

        # Pair Indices: (Seq_Len,) - Long tensor for indexing/gathering
        p_idx = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        # Pair Mask: (Seq_Len,) - Float tensor for masking operations
        p_mask = torch.tensor(self.pair_masks[idx], dtype=torch.float32)

        item = {"features": feat, "pair_indices": p_idx, "pair_mask": p_mask}

        # Targets: (Seq_Len, 5) - Optional (not present in test set)
        if self.targets is not None:
            t = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = t

        return item


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string to generate adjacency indices and masks.

    Args:
        structure (str): Dot-bracket string e.g. "((..))"

    Returns:
        indices (np.ndarray): Array where indices[i] is the index of the base paired with i.
                              Unpaired bases are set to 0 (safe for gathering, masked later).
        mask (np.ndarray): Array where mask[i] is 1.0 if paired, 0.0 if unpaired.
    """
    n = len(structure)
    indices = np.zeros(n, dtype=np.int32)
    mask = np.zeros(n, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Register the pair (i, j)
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
            else:
                # Unbalanced closing parenthesis - treat as unpaired
                pass
        # '.' characters are already 0 in indices and mask

    return indices, mask


def preprocess_data(df, mode="train"):
    """
    Converts raw DataFrame into structured numpy arrays for the model.
    """
    # Channel Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {".": 0, "(": 1, ")": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays for efficiency
    features = np.zeros((num_samples, seq_len, Config.INPUT_DIM), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Sequence Features (Channels 0-3)
        seq = row["sequence"]
        for i, char in enumerate(seq):
            if i >= seq_len:
                break
            if char in seq_map:
                features[idx, i, seq_map[char]] = 1.0

        # 2. Structure Features (Channels 4-6)
        struct = row["structure"]
        for i, char in enumerate(struct):
            if i >= seq_len:
                break
            if char in struct_map:
                features[idx, i, 4 + struct_map[char]] = 1.0

        # 3. Loop Type Features (Channels 7-13)
        loop = row["predicted_loop_type"]
        for i, char in enumerate(loop):
            if i >= seq_len:
                break
            if char in loop_map:
                features[idx, i, 7 + loop_map[char]] = 1.0

        # 4. Structural Adjacency
        p_idx, p_mask = get_structure_indices(struct)
        L = min(len(p_idx), seq_len)
        pair_indices[idx, :L] = p_idx[:L]
        pair_masks[idx, :L] = p_mask[:L]

        # 5. Targets (Train/Val only)
        if mode in ["train", "val"]:
            # Targets are provided as lists/arrays of length 68
            # We pad them to 107
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                if isinstance(val_list, (list, np.ndarray)):
                    length = min(len(val_list), seq_len)
                    targets[idx, :length, t_i] = val_list[:length]

    return features, pair_indices, pair_masks, targets


def load_and_process(metadata_path, mode="train", load_cached_data=True, debug=False):
    """
    Orchestrates data loading with caching mechanism.
    """
    # 1. Load Metadata
    try:
        df = pd.read_parquet(metadata_path)
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        raise

    if debug:
        df = df.head(Config.SUBSET_SIZE).reset_index(drop=True)

    # 2. Generate Cache Key
    # Hash based on IDs to ensure data consistency
    ids_hash = get_md5_hash(df["id"].tolist())
    cache_filename = f"{mode}_data_{ids_hash}.npz"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 3. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            features = data["features"]
            pair_indices = data["pair_indices"]
            pair_masks = data["pair_masks"]
            targets = data["targets"]
            return features, pair_indices, pair_masks, targets
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 4. Process Data from Scratch
    features, pair_indices, pair_masks, targets = preprocess_data(df, mode)

    # 5. Save Cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez(
        cache_path,
        features=features,
        pair_indices=pair_indices,
        pair_masks=pair_masks,
        targets=targets,
    )

    return features, pair_indices, pair_masks, targets


def get_dataloaders(load_cached_data=True):
    """
    Creates Train, Validation, and Test DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Train Loader
    train_feat, train_pidx, train_mask, train_y = load_and_process(
        Config.TRAIN_METADATA,
        mode="train",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )
    train_dataset = RNADataset(train_feat, train_pidx, train_mask, train_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Stability for batch norm / deeper networks
    )

    # Validation Loader
    val_feat, val_pidx, val_mask, val_y = load_and_process(
        Config.VAL_METADATA,
        mode="val",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )
    val_dataset = RNADataset(val_feat, val_pidx, val_mask, val_y)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test Loader
    test_feat, test_pidx, test_mask, _ = load_and_process(
        Config.TEST_METADATA,
        mode="test",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )
    # Test set has no targets
    test_dataset = RNADataset(test_feat, test_pidx, test_mask, targets=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
