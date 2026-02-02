import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# ==========================================
# Mappings for One-Hot Encoding
# ==========================================
TOKEN_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_bpp_features(structure_str, seq_len):
    """
    Parses dot-bracket structure to generate adjacency indices and masks.

    Strategy: Unified GLU-Decoupled Interaction Module
    - bpp_indices: If paired, index of partner. If unpaired, defaults to 0.
    - bpp_mask: 1.0 if paired, 0.0 if unpaired.

    Args:
        structure_str (str): Dot-bracket string.
        seq_len (int): Length of sequence.

    Returns:
        indices (np.ndarray): Shape (seq_len,)
        mask (np.ndarray): Shape (seq_len,)
    """
    stack = []
    indices = np.zeros(seq_len, dtype=np.int64)  # Default to 0 for unpaired
    mask = np.zeros(seq_len, dtype=np.float32)  # Default to 0 for unpaired

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i
                mask[i] = 1.0
                mask[j] = 1.0

    return indices, mask


def one_hot_encode(seq, mapping, length, num_classes):
    """
    One-hot encodes a sequence string based on a mapping.
    """
    arr = np.zeros((length, num_classes), dtype=np.float32)
    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in mapping:
            arr[i, mapping[char]] = 1.0
    return arr


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Features:
    - Sequence (One-Hot, 4)
    - Structure (One-Hot, 3)
    - Loop Type (One-Hot, 7)
    - BPP Indices and Mask for Interaction Module

    Targets:
    - 5 regression targets padded to sequence length.
    """

    def __init__(self, data_dict, mode="train"):
        self.inputs = data_dict["inputs"]
        self.bpp_indices = data_dict["bpp_indices"]
        self.bpp_masks = data_dict["bpp_masks"]
        self.ids = data_dict["ids"]
        self.mode = mode

        if self.mode != "test":
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Inputs: (Seq_Len, 14)
        input_tensor = torch.from_numpy(self.inputs[idx])

        # BPP Features
        bpp_idx = torch.from_numpy(self.bpp_indices[idx])
        bpp_mask = torch.from_numpy(self.bpp_masks[idx])

        sample_id = self.ids[idx]

        if self.mode != "test":
            # Targets: (Seq_Len, 5)
            target_tensor = torch.from_numpy(self.targets[idx])
            return input_tensor, bpp_idx, bpp_mask, target_tensor
        else:
            return input_tensor, bpp_idx, bpp_mask, sample_id


def process_dataframe(df, config, mode="train"):
    """
    Processes a pandas DataFrame into numpy arrays for the dataset.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Initialize arrays
    # 14 channels: 4 (Seq) + 3 (Struct) + 7 (Loop)
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    bpp_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    bpp_masks = np.zeros((num_samples, seq_len), dtype=np.float32)
    ids = df["id"].values

    # Process features
    for idx, row in df.iterrows():
        # 1. Sequence One-Hot
        seq_oh = one_hot_encode(row["sequence"], TOKEN_MAP, seq_len, 4)

        # 2. Structure One-Hot
        struct_oh = one_hot_encode(row["structure"], STRUCT_MAP, seq_len, 3)

        # 3. Loop Type One-Hot
        loop_oh = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len, 7)

        # Concatenate features
        inputs[idx] = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)

        # 4. BPP Adjacency
        b_idx, b_mask = get_bpp_features(row["structure"], seq_len)
        bpp_indices[idx] = b_idx
        bpp_masks[idx] = b_mask

    # Process targets if not test mode
    targets = None
    if mode != "test":
        targets = np.zeros((num_samples, seq_len, 5), dtype=np.float32)
        target_cols = config.target_cols

        for t_i, col in enumerate(target_cols):
            # Each row[col] is a list/array of length seq_scored (68)
            # We stack them and pad to seq_len (107)

            # Extract column data as a list of arrays
            col_data = df[col].values

            for idx, val_array in enumerate(col_data):
                # Ensure it's a list or array
                if isinstance(val_array, (list, np.ndarray)):
                    length = len(val_array)
                    # Copy data to the beginning of the sequence
                    targets[idx, :length, t_i] = val_array
                else:
                    # Fallback for unexpected data types
                    pass

    return {
        "inputs": inputs,
        "bpp_indices": bpp_indices,
        "bpp_masks": bpp_masks,
        "targets": targets,
        "ids": ids,
    }


def load_or_process_data(
    file_path, cache_path, load_cached_data, config, mode="train", max_samples=None
):
    """
    Loads data from cache or processes it from scratch.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}...")
            # Allow pickle is required for loading object arrays if any,
            # but we used savez which stores numpy arrays.
            # However, np.load returns a NpzFile wrapper, we convert to dict.
            data = np.load(cache_path)
            data_dict = {
                "inputs": data["inputs"],
                "bpp_indices": data["bpp_indices"],
                "bpp_masks": data["bpp_masks"],
                "ids": data["ids"],
            }
            if mode != "test":
                data_dict["targets"] = data["targets"]

            # Handle max_samples for debugging after loading cache
            if max_samples is not None:
                for k in data_dict:
                    if data_dict[k] is not None:
                        data_dict[k] = data_dict[k][:max_samples]

            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {file_path}...")
    df = pd.read_parquet(file_path)

    # Debugging subsample
    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    data_dict = process_dataframe(df, config, mode=mode)

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    save_dict = {
        "inputs": data_dict["inputs"],
        "bpp_indices": data_dict["bpp_indices"],
        "bpp_masks": data_dict["bpp_masks"],
        "ids": data_dict["ids"],
    }
    if mode != "test":
        save_dict["targets"] = data_dict["targets"]
    else:
        # For test, we can't save None in np.savez easily without object dtype
        # Just don't save targets key
        pass

    np.savez(cache_path, **save_dict)
    print(f"Data cached to {cache_path}.")

    return data_dict


def get_dataloaders(config, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    """
    # Load Train Data
    train_data = load_or_process_data(
        config.train_file,
        config.train_cache.replace(".npy", ".npz"),  # Use .npz for savez
        load_cached_data,
        config,
        mode="train",
        max_samples=config.max_train_samples,
    )

    # Load Val Data
    val_data = load_or_process_data(
        config.val_file,
        config.val_cache.replace(".npy", ".npz"),
        load_cached_data,
        config,
        mode="train",  # Val has targets, so treat as train mode
        max_samples=config.max_val_samples,
    )

    # Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="train")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(config, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    test_data = load_or_process_data(
        config.test_file,
        config.test_cache.replace(".npy", ".npz"),
        load_cached_data,
        config,
        mode="test",
        max_samples=None,  # Always predict full test set
    )

    test_dataset = RNADataset(test_data, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
