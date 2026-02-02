import os
import numpy as np
import pandas as pd
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
# Helper Functions
# =========================================================================


def parse_structure(structure_str):
    """
    Parses a dot-bracket structure string to generate pair indices and masks.

    Args:
        structure_str (str): Dot-bracket notation string (e.g., "((..))").

    Returns:
        pair_index (np.ndarray): Shape (L,), indices of paired bases. Unpaired = 0.
        pair_mask (np.ndarray): Shape (L,), 1.0 if paired, 0.0 if unpaired.
    """
    n = len(structure_str)
    pair_index = np.zeros(n, dtype=np.int32)
    pair_mask = np.zeros(n, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Bidirectional linkage
                pair_index[i] = j
                pair_index[j] = i
                # Mask valid pairs
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0
            else:
                # Unbalanced structure (rare in this dataset, but safe to ignore)
                pass

    return pair_index, pair_mask


def one_hot_encode(seq, mapping, dim):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((len(seq), dim), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def process_data(df, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays suitable for the model.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    input_dim = Config.INPUT_DIM  # 14

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, input_dim), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Check if targets exist (Train/Val)
    has_targets = mode in ["train", "val"]
    if has_targets:
        targets = np.zeros(
            (num_samples, Config.PRED_LEN, Config.NUM_TARGETS), dtype=np.float32
        )
    else:
        targets = None

    # Iterate and process
    # Note: reset_index is assumed to have been called upstream or indices are standard
    for idx, (_, row) in enumerate(df.iterrows()):
        # 1. Input Features
        # Sequence (4 channels)
        seq_feat = one_hot_encode(row["sequence"], SEQ_MAP, 4)
        # Structure (3 channels)
        struct_feat = one_hot_encode(row["structure"], STRUCT_MAP, 3)
        # Loop Type (7 channels)
        loop_feat = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, 7)

        # Concatenate features along channel dimension
        inputs[idx] = np.concatenate([seq_feat, struct_feat, loop_feat], axis=1)

        # 2. Structural Interaction Indices
        p_idx, p_mask = parse_structure(row["structure"])
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets
        if has_targets:
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_arr = np.zeros((Config.PRED_LEN, Config.NUM_TARGETS), dtype=np.float32)
            for t_i, col in enumerate(Config.TARGET_COLS):
                val = row[col]
                if isinstance(val, (list, np.ndarray)):
                    # Ensure length matches Config.PRED_LEN (68)
                    length = min(len(val), Config.PRED_LEN)
                    t_arr[:length, t_i] = val[:length]
            targets[idx] = t_arr

    return {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "targets": targets,
        "ids": ids,
    }


def get_or_create_data(mode, load_cached_data=True):
    """
    Retrieves data from cache or processes it from source Parquet files.
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            data_dict = {
                "inputs": loaded["inputs"],
                "pair_indices": loaded["pair_indices"],
                "pair_masks": loaded["pair_masks"],
                "ids": loaded["ids"],
                "targets": loaded["targets"] if "targets" in loaded else None,
            }
            # Fix for test set where targets might be None or missing
            if mode == "test":
                data_dict["targets"] = None
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from source
    print(f"Processing {mode} data from metadata...")

    if mode == "train":
        path = Config.TRAIN_PARQUET
    elif mode == "val":
        path = Config.VAL_PARQUET
    elif mode == "test":
        path = Config.TEST_PARQUET
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_parquet(path)

    # Debugging Subset
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)
        print(f"DEBUG MODE: Subsampled {mode} to {len(df)} rows.")

    data_dict = process_data(df, mode)

    # 3. Save to cache
    print(f"Saving {mode} data to cache: {cache_file}")
    save_dict = {
        "inputs": data_dict["inputs"],
        "pair_indices": data_dict["pair_indices"],
        "pair_masks": data_dict["pair_masks"],
        "ids": data_dict["ids"],
    }
    if data_dict["targets"] is not None:
        save_dict["targets"] = data_dict["targets"]

    np.savez_compressed(cache_file, **save_dict)

    return data_dict


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, data_dict, mode="train"):
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.pair_masks = data_dict["pair_masks"]
        self.ids = data_dict["ids"]
        self.targets = data_dict.get("targets")
        self.mode = mode

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "pair_indices": torch.tensor(self.pair_indices[idx], dtype=torch.long),
            "pair_masks": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
            "ids": str(self.ids[idx]),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


# =========================================================================
# Data Loaders
# =========================================================================


def get_loaders(load_cached_data=True):
    """
    Initializes datasets and returns DataLoaders for Train, Val, and Test.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Load Data
    train_data = get_or_create_data("train", load_cached_data)
    val_data = get_or_create_data("val", load_cached_data)
    test_data = get_or_create_data("test", load_cached_data)

    # 2. Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # 3. Create DataLoaders
    # Pin memory speeds up host-to-device transfer for CUDA
    pin_memory = Config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,  # Avoid small last batches affecting BatchNorm/Stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
