import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config


# ==========================================
# Dataset Class
# ==========================================
class RNADataset(Dataset):
    def __init__(self, features, pair_indices, pair_masks, targets, ids):
        """
        Args:
            features: Tensor of shape (N, 107, 14)
            pair_indices: LongTensor of shape (N, 107)
            pair_masks: FloatTensor of shape (N, 107)
            targets: Tensor of shape (N, 107, 5)
            ids: List of sample IDs
        """
        self.features = features
        self.pair_indices = pair_indices
        self.pair_masks = pair_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "pair_indices": self.pair_indices[idx],
            "pair_masks": self.pair_masks[idx],
            "targets": self.targets[idx],
            "ids": self.ids[idx],
        }


# ==========================================
# Helper Functions
# ==========================================
def get_structure_indices(structure_str, length):
    """
    Parses dot-bracket structure to generate pair indices and masks.
    Unpaired bases point to themselves (index i) but have mask 0.
    Paired bases point to their partner (index j) and have mask 1.
    """
    pair_indices = np.arange(length)  # Default: point to self
    mask = np.zeros(length, dtype=np.float32)

    stack = []
    for i, char in enumerate(structure_str):
        if i >= length:
            break
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0
    return pair_indices, mask


def one_hot_encode(seq, mapping, length):
    """
    One-hot encodes a sequence string based on a mapping dictionary.
    """
    arr = np.zeros((length, len(mapping)), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


def preprocess_dataframe(df):
    """
    Converts a pandas DataFrame into tensors required for the model.
    """
    # Mappings
    SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    STRUCT_MAP = {".": 0, "(": 1, ")": 2}
    LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Pre-allocate arrays
    features = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].tolist()

    # Check if targets exist (Train/Val) or not (Test)
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    has_targets = all(col in df.columns for col in target_cols)

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Features
        # Sequence (4 channels)
        f_seq = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)
        # Structure (3 channels)
        f_struct = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)
        # Loop Type (7 channels)
        f_loop = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        features[idx] = np.concatenate([f_seq, f_struct, f_loop], axis=1)

        # 2. Structure Adjacency
        p_idx, p_mask = get_structure_indices(row["structure"], seq_len)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

        # 3. Targets
        if has_targets:
            for t_i, col in enumerate(target_cols):
                val = row[col]
                # val is a list/array. Usually length 68.
                # We pad to 107 with zeros (masked out during scoring anyway)
                if isinstance(val, (list, np.ndarray)):
                    length_scored = min(len(val), seq_len)
                    targets[idx, :length_scored, t_i] = val[:length_scored]

    return (
        torch.tensor(features),
        torch.tensor(pair_indices),
        torch.tensor(pair_masks),
        torch.tensor(targets),
        ids,
    )


# ==========================================
# Main Data Loading Function
# ==========================================
def get_dataloaders(load_cached_data=True):
    """
    Generates DataLoaders for Train, Val, and Test sets.
    Handles caching of processed tensors to disk.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    splits = {
        "train": Config.TRAIN_PATH,
        "val": Config.VAL_PATH,
        "test": Config.TEST_PATH,
    }

    datasets = {}

    for split_name, file_path in splits.items():
        cache_path = os.path.join(cache_dir, f"{split_name}_data.pt")

        # Logic: Load cache if requested and exists, else process from scratch
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {split_name} data from {cache_path}")
            data_tuple = torch.load(cache_path)
        else:
            print(f"Processing {split_name} data from {file_path}...")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Source file not found: {file_path}")

            df = pd.read_parquet(file_path)
            data_tuple = preprocess_dataframe(df)

            print(f"Saving {split_name} data to {cache_path}")
            torch.save(data_tuple, cache_path)

        # Unpack tuple and create Dataset
        datasets[split_name] = RNADataset(*data_tuple)

    # Create DataLoaders
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
