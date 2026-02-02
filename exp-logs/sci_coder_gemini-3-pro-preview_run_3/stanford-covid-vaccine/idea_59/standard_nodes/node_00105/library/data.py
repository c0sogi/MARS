import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Token maps for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to generate pair indices and a mask.

    Args:
        structure_str: Dot-bracket string (e.g., "((..))").
        seq_len: Length of the sequence.

    Returns:
        pair_indices: Array of shape (seq_len,). If i is paired with j, indices[i] = j.
                      If unpaired, indices[i] = i (self-loop placeholder, masked out later).
        pair_mask: Array of shape (seq_len,). 1.0 if paired, 0.0 if unpaired.
    """
    pair_indices = np.arange(seq_len)  # Default to self if unpaired (will be masked)
    pair_mask = np.zeros(seq_len, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
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


def one_hot_encode(seq, token_map, num_classes):
    """
    One-hot encodes a sequence string based on a token map.
    """
    arr = np.zeros((len(seq), num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in token_map:
            arr[i, token_map[char]] = 1.0
    return arr


def process_data(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for features, adjacency, and targets.

    Args:
        df: Input dataframe.
        mode: 'train', 'val', or 'test'.

    Returns:
        Dictionary containing numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    pred_len = Config.PRED_LEN

    # Pre-allocate arrays
    # Features: (N, 107, 14)
    features = np.zeros((num_samples, seq_len, Config.NUM_FEATURES), dtype=np.float32)
    # Adjacency
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    # IDs
    ids = df["id"].values

    # Targets: (N, 68, 5) - Only for train/val
    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros(
            (num_samples, pred_len, Config.NUM_TARGETS), dtype=np.float32
        )

    for idx, row in df.iterrows():
        # 1. Features
        # Sequence (4)
        seq_oh = one_hot_encode(row["sequence"], SEQ_MAP, 4)
        # Structure (3)
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, 3)
        # Loop Type (7)
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate: (107, 14)
        features[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 2. Adjacency / Structure Graph
        p_idx, p_mask = get_structure_adj(row["structure"], seq_len)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets
        if mode in ["train", "val"]:
            # Extract lists
            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_list = []
            for col in Config.TARGET_COLS:
                val = row[col]
                # Ensure it's a list/array of length 68
                if len(val) < pred_len:
                    # Pad if necessary (though data should be clean)
                    val = list(val) + [0.0] * (pred_len - len(val))
                t_list.append(val[:pred_len])

            # Stack to (5, 68) then transpose to (68, 5)
            targets[idx] = np.array(t_list, dtype=np.float32).T

    data_dict = {
        "features": features,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "ids": ids,
    }

    if targets is not None:
        data_dict["targets"] = targets

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.features = torch.tensor(data_dict["features"], dtype=torch.float32)
        self.pair_indices = torch.tensor(data_dict["pair_indices"], dtype=torch.long)
        self.pair_masks = torch.tensor(data_dict["pair_masks"], dtype=torch.float32)
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode in ["train", "val"] and "targets" in data_dict:
            self.targets = torch.tensor(data_dict["targets"], dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "features": self.features[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_mask": self.pair_masks[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        return sample


def get_dataloaders(load_cached_data=True):
    """
    Loads data, processes it (or loads from cache), and returns DataLoaders.

    Args:
        load_cached_data: If True, attempts to load .npz files from working dir.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_files = {
        "train": Config.TRAIN_CACHE,
        "val": Config.VAL_CACHE,
        "test": Config.TEST_CACHE,
    }

    # Define metadata paths
    meta_files = {
        "train": Config.TRAIN_METADATA,
        "val": Config.VAL_METADATA,
        "test": Config.TEST_METADATA,
    }

    datasets = {}

    for mode in ["train", "val", "test"]:
        cache_path = cache_files[mode]
        meta_path = meta_files[mode]

        data_loaded = False

        # 1. Try Loading Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                print(f"Loading {mode} data from cache: {cache_path}")
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {
                    "features": loaded["features"],
                    "pair_indices": loaded["pair_indices"],
                    "pair_masks": loaded["pair_masks"],
                    "ids": loaded["ids"],
                }
                if "targets" in loaded:
                    data_dict["targets"] = loaded["targets"]
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Reprocessing.")

        # 2. Process from Scratch if needed
        if not data_loaded:
            print(f"Processing {mode} data from metadata: {meta_path}")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")

            df = pd.read_parquet(meta_path)
            data_dict = process_data(df, mode=mode)

            # Save to cache
            print(f"Saving {mode} data to cache: {cache_path}")
            save_kwargs = {
                "features": data_dict["features"],
                "pair_indices": data_dict["pair_indices"],
                "pair_masks": data_dict["pair_masks"],
                "ids": data_dict["ids"],
            }
            if "targets" in data_dict:
                save_kwargs["targets"] = data_dict["targets"]

            np.savez_compressed(cache_path, **save_kwargs)

        # 3. Create Dataset
        datasets[mode] = RNADataset(data_dict, mode=mode)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        datasets["train"],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,
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
