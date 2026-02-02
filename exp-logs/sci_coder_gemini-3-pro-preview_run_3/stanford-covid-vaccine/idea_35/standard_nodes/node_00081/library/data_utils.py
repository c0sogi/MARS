import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Constants & Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

# =========================================================================
# Preprocessing Functions
# =========================================================================


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate adjacency indices and a pair mask.

    This is critical for the Decoupled Structural Interaction Module.

    Args:
        structure (str): Dot-bracket string (e.g., "((...))").

    Returns:
        adj_indices (np.ndarray): Array of shape (L,) where adj_indices[i] is the index
                                  of the base paired with i. If unpaired, set to 0 (safe index).
        pair_mask (np.ndarray): Array of shape (L, 1) where 1 indicates paired, 0 unpaired.
    """
    length = len(structure)
    adj_indices = np.zeros(length, dtype=np.int64)
    pair_mask = np.zeros((length, 1), dtype=np.float32)

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i and j are paired
                adj_indices[i] = j
                adj_indices[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0
            else:
                # Unbalanced closing parenthesis (shouldn't happen in valid data but handle safely)
                pass

    # Note: adj_indices[k] = 0 for unpaired k is safe because we multiply by pair_mask[k] = 0
    # in the model forward pass (Zero-Masking).

    return adj_indices, pair_mask


def one_hot_encode(seq, struct, loop):
    """
    Generates a (L, 14) one-hot encoded feature matrix.

    Channels:
    0-3: Sequence (A, G, C, U)
    4-6: Structure (., (, ))
    7-13: Loop Type (S, M, I, B, H, E, X)
    """
    length = len(seq)
    features = np.zeros((length, Config.INPUT_DIM), dtype=np.float32)

    for i in range(length):
        # Sequence (0-3)
        if seq[i] in SEQ_MAP:
            features[i, SEQ_MAP[seq[i]]] = 1.0

        # Structure (4-6)
        if struct[i] in STRUCT_MAP:
            features[i, 4 + STRUCT_MAP[struct[i]]] = 1.0

        # Loop Type (7-13)
        if loop[i] in LOOP_MAP:
            features[i, 7 + LOOP_MAP[loop[i]]] = 1.0

    return features


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, features, adj_indices, pair_masks, targets=None, ids=None):
        """
        Args:
            features: (N, 107, 14)
            adj_indices: (N, 107)
            pair_masks: (N, 107, 1)
            targets: (N, 68, 5) or None
            ids: List of IDs
        """
        self.features = features
        self.adj_indices = adj_indices
        self.pair_masks = pair_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "adj_indices": torch.tensor(self.adj_indices[idx], dtype=torch.long),
            "pair_mask": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


# =========================================================================
# Data Loading & Caching Logic
# =========================================================================


def process_and_cache_data(mode, load_cached_data=True):
    """
    Loads raw data from metadata parquet files, processes features/targets,
    and caches numpy arrays to disk.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict containing numpy arrays: features, adj_indices, pair_masks, targets, ids
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "features": data["features"],
                "adj_indices": data["adj_indices"],
                "pair_masks": data["pair_masks"],
                "targets": data["targets"] if "targets" in data else None,
                "ids": data["ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load Metadata
    print(f"Processing {mode} data from scratch...")
    if mode == "train":
        df = pd.read_parquet(Config.TRAIN_PATH)
    elif mode == "val":
        df = pd.read_parquet(Config.VAL_PATH)
    elif mode == "test":
        df = pd.read_parquet(Config.TEST_PATH)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3. Preallocate Arrays
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    pred_len = Config.PRED_LEN

    features = np.zeros((num_samples, seq_len, Config.INPUT_DIM), dtype=np.float32)
    adj_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len, 1), dtype=np.float32)
    ids = df["id"].values

    has_targets = mode != "test"
    targets = (
        np.zeros((num_samples, pred_len, Config.NUM_TARGETS), dtype=np.float32)
        if has_targets
        else None
    )

    # 4. Process Rows
    for idx, row in df.iterrows():
        # Features
        feat = one_hot_encode(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )
        features[idx] = feat

        # Adjacency
        adj, mask = get_structure_adj(row["structure"])
        adj_indices[idx] = adj
        pair_masks[idx] = mask

        # Targets
        if has_targets:
            # Targets are lists of floats in the dataframe
            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_matrix = np.zeros((pred_len, Config.NUM_TARGETS), dtype=np.float32)
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure length matches pred_len
                length = min(len(val_list), pred_len)
                t_matrix[:length, t_i] = val_list[:length]
            targets[idx] = t_matrix

    # 5. Save to Cache
    save_dict = {
        "features": features,
        "adj_indices": adj_indices,
        "pair_masks": pair_masks,
        "ids": ids,
    }
    if has_targets:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved {mode} data to cache: {cache_path}")

    return {
        "features": features,
        "adj_indices": adj_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=False,
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        load_cached_data (bool): Whether to use cached .npz files.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Process Data
    train_data = process_and_cache_data("train", load_cached_data)
    val_data = process_and_cache_data("val", load_cached_data)
    test_data = process_and_cache_data("test", load_cached_data)

    # Debug Subsetting
    if debug:
        print(f"Debug mode: Reducing dataset size to {Config.DEBUG_SAMPLES}")
        limit = Config.DEBUG_SAMPLES
        for data in [train_data, val_data, test_data]:
            data["features"] = data["features"][:limit]
            data["adj_indices"] = data["adj_indices"][:limit]
            data["pair_masks"] = data["pair_masks"][:limit]
            data["ids"] = data["ids"][:limit]
            if data["targets"] is not None:
                data["targets"] = data["targets"][:limit]

    # Create Datasets
    train_dataset = RNADataset(
        train_data["features"],
        train_data["adj_indices"],
        train_data["pair_masks"],
        train_data["targets"],
        train_data["ids"],
    )

    val_dataset = RNADataset(
        val_data["features"],
        val_data["adj_indices"],
        val_data["pair_masks"],
        val_data["targets"],
        val_data["ids"],
    )

    test_dataset = RNADataset(
        test_data["features"],
        test_data["adj_indices"],
        test_data["pair_masks"],
        targets=None,
        ids=test_data["ids"],
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
