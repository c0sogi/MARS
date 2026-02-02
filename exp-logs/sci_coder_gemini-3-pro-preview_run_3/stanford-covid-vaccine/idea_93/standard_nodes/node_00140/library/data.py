import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_structure_indices


def process_data(df, mode="train"):
    """
    Process the dataframe into input features and targets.

    Args:
        df (pd.DataFrame): The dataframe containing sequences and structures.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Input features {'X', 'adj', 'mask', 'ids'}.
        np.ndarray or None: Targets (N, 68, 5) if available.
    """
    # Mappings for One-Hot Encoding
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    struct_map = {"(": 0, ")": 1, ".": 2}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    N = len(df)
    L = Config.SEQ_LEN

    # Initialize feature tensors
    # Channels: 0-3 (Seq), 4-6 (Struct), 7-13 (Loop) -> Total 14
    X = np.zeros((N, L, Config.INPUT_CHANNELS), dtype=np.float32)

    # Initialize adjacency and mask for interaction module
    adj = np.zeros((N, L), dtype=np.int32)
    mask = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        seq = sequences[i]
        struct = structures[i]
        loop = loops[i]

        # 1. Sequence Encoding
        for j, char in enumerate(seq):
            if char in seq_map:
                X[i, j, seq_map[char]] = 1.0

        # 2. Structure Encoding
        for j, char in enumerate(struct):
            if char in struct_map:
                X[i, j, 4 + struct_map[char]] = 1.0

        # 3. Loop Type Encoding
        for j, char in enumerate(loop):
            if char in loop_map:
                X[i, j, 7 + loop_map[char]] = 1.0

        # 4. Adjacency Map & Mask
        # get_structure_indices returns -1 for unpaired bases
        indices = get_structure_indices(struct)

        # Mask: 1.0 if paired, 0.0 if unpaired
        m = (indices != -1).astype(np.float32)
        mask[i] = m

        # Adj: Replace -1 with 0 to ensure valid indices for torch.gather.
        # The gathered value at index 0 will be zeroed out by the mask later.
        a = indices.copy()
        a[a == -1] = 0
        adj[i] = a

    # 5. Process Targets
    targets = None
    if mode in ["train", "val"]:
        # Targets shape: (N, SEQ_SCORED, 5)
        targets = np.zeros((N, Config.SEQ_SCORED, 5), dtype=np.float32)
        for t_idx, col in enumerate(Config.TARGET_COLS):
            # df[col] contains lists; stack them into a matrix
            vals = np.vstack(df[col].values)
            targets[:, :, t_idx] = vals

    return {"X": X, "adj": adj, "mask": mask, "ids": ids}, targets


def get_dataset(mode="train", load_cached_data=True):
    """
    Load dataset from cache or process from scratch.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (inputs, targets) for train/val, or (inputs) for test.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            inputs = {k: data[k] for k in ["X", "adj", "mask", "ids"]}

            if mode != "test":
                targets = data["targets"]
                return inputs, targets
            else:
                return inputs
        except Exception as e:
            print(f"Cache load failed for {mode}: {e}. Processing from scratch.")

    # 2. Process from scratch
    meta_path = os.path.join(Config.METADATA_DIR, f"{mode}.parquet")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)
    inputs, targets = process_data(df, mode)

    # 3. Save to cache
    save_dict = {**inputs}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez(cache_path, **save_dict)

    if mode != "test":
        return inputs, targets
    else:
        return inputs


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA data.
    """

    def __init__(self, inputs, targets=None):
        self.X = inputs["X"]
        self.adj = inputs["adj"]
        self.mask = inputs["mask"]
        self.targets = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert to tensors
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        adj = torch.tensor(self.adj[idx], dtype=torch.long)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)

        item = {"X": x, "adj": adj, "mask": mask}

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return item, y

        return item
