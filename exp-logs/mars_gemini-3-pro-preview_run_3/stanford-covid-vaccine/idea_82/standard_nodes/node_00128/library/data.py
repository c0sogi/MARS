import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import parse_structure_to_adj

# Dictionaries for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    def __init__(self, inputs, neighbor_indices, pair_masks, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, 14)
            neighbor_indices (np.ndarray): Shape (N, 107)
            pair_masks (np.ndarray): Shape (N, 107)
            targets (np.ndarray, optional): Shape (N, 107, 5)
            ids (list, optional): List of sequence IDs
        """
        self.inputs = inputs
        self.neighbor_indices = neighbor_indices
        self.pair_masks = pair_masks
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to FloatTensor
        item = {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "neighbor_indices": torch.tensor(
                self.neighbor_indices[idx], dtype=torch.long
            ),
            "pair_masks": torch.tensor(self.pair_masks[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def one_hot_encode(seq, mapping, depth):
    """Encodes a string sequence into a one-hot numpy array."""
    encoding = np.zeros((len(seq), depth), dtype=np.float32)
    for i, char in enumerate(seq):
        if char in mapping:
            encoding[i, mapping[char]] = 1.0
    return encoding


def process_dataframe(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for features and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Initialize arrays
    # 4 (Seq) + 3 (Struct) + 7 (Loop) = 14 channels
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    neighbor_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    targets = None
    if mode in ["train", "val"]:
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)

    ids = df["id"].tolist()

    for idx, row in df.iterrows():
        # 1. Features
        sequence = row["sequence"]
        structure = row["structure"]
        loop_type = row["predicted_loop_type"]

        # Ensure lengths match Config
        if len(sequence) != seq_len:
            # Should not happen based on dataset description, but safety check
            continue

        # One-hot encoding
        enc_seq = one_hot_encode(sequence, SEQ_MAP, 4)
        enc_struct = one_hot_encode(structure, STRUCT_MAP, 3)
        enc_loop = one_hot_encode(loop_type, LOOP_MAP, 7)

        inputs[idx] = np.concatenate([enc_seq, enc_struct, enc_loop], axis=1)

        # 2. Adjacency / Structural Interaction
        # parse_structure_to_adj returns -1 for unpaired, index for paired
        adj = parse_structure_to_adj(structure)

        # Create mask: 1 if paired (adj != -1), 0 otherwise
        mask = (adj != -1).astype(np.float32)
        pair_masks[idx] = mask

        # Replace -1 with 0 for safe gathering (masked out later anyway)
        safe_adj = adj.copy()
        safe_adj[adj == -1] = 0
        neighbor_indices[idx] = safe_adj

        # 3. Targets (only for train/val)
        if targets is not None:
            # Targets are provided as lists of length 68
            # We pad them to 107 with zeros
            for t_i, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Assign the first 68 values
                # Note: val_list is a list or numpy array
                length_scored = len(val_list)
                targets[idx, :length_scored, t_i] = val_list

    data_dict = {
        "inputs": inputs,
        "neighbor_indices": neighbor_indices,
        "pair_masks": pair_masks,
        "ids": ids,
    }

    if targets is not None:
        data_dict["targets"] = targets

    return data_dict


def load_or_process_data(mode, metadata_path, load_cached_data=True):
    """
    Loads data from cache or processes from metadata parquet file.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_data.npz")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading {mode} data from cache: {cache_file}")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            data_dict = {
                "inputs": loaded["inputs"],
                "neighbor_indices": loaded["neighbor_indices"],
                "pair_masks": loaded["pair_masks"],
                "ids": loaded["ids"].tolist(),
            }
            if "targets" in loaded:
                data_dict["targets"] = loaded["targets"]
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from scratch
    print(f"Processing {mode} data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLES)
        print(f"DEBUG MODE: Subsampled {mode} to {len(df)} rows.")

    data_dict = process_dataframe(df, mode=mode)

    # Save to cache
    save_dict = {
        "inputs": data_dict["inputs"],
        "neighbor_indices": data_dict["neighbor_indices"],
        "pair_masks": data_dict["pair_masks"],
        "ids": data_dict["ids"],
    }
    if "targets" in data_dict:
        save_dict["targets"] = data_dict["targets"]

    np.savez_compressed(cache_file, **save_dict)
    print(f"Saved {mode} data to cache: {cache_file}")

    return data_dict


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Data
    train_data = load_or_process_data(
        "train", Config.TRAIN_METADATA_PATH, load_cached_data
    )
    val_data = load_or_process_data("val", Config.VAL_METADATA_PATH, load_cached_data)
    test_data = load_or_process_data(
        "test", Config.TEST_METADATA_PATH, load_cached_data
    )

    # 2. Create Datasets
    train_dataset = RNADataset(
        inputs=train_data["inputs"],
        neighbor_indices=train_data["neighbor_indices"],
        pair_masks=train_data["pair_masks"],
        targets=train_data["targets"],
        ids=train_data["ids"],
    )

    val_dataset = RNADataset(
        inputs=val_data["inputs"],
        neighbor_indices=val_data["neighbor_indices"],
        pair_masks=val_data["pair_masks"],
        targets=val_data["targets"],
        ids=val_data["ids"],
    )

    test_dataset = RNADataset(
        inputs=test_data["inputs"],
        neighbor_indices=test_data["neighbor_indices"],
        pair_masks=test_data["pair_masks"],
        targets=None,  # Test set has no targets
        ids=test_data["ids"],
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    print(
        f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
