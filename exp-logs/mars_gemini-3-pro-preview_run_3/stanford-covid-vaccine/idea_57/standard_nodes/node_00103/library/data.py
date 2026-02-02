import os
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    def __init__(self, inputs, pair_indices, pair_masks, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, 14)
            pair_indices (np.ndarray): Shape (N, 107)
            pair_masks (np.ndarray): Shape (N, 107)
            targets (np.ndarray, optional): Shape (N, 68, 5)
            ids (list, optional): List of sequence IDs
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.pair_masks = torch.tensor(pair_masks, dtype=torch.float32)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        self.ids = ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            "inputs": self.inputs[idx],
            "pair_index": self.pair_indices[idx],
            "pair_mask": self.pair_masks[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def get_structure_adj(structure_str, length):
    """
    Parses dot-bracket structure to find pairs.
    Returns:
        pair_idx: Array where pair_idx[i] = j if (i, j) are paired.
                  If unpaired, pair_idx[i] = i (self-loop for safe gathering).
        pair_mask: Array where pair_mask[i] = 1 if paired, 0 if unpaired.
    """
    pair_idx = np.arange(length)  # Default to self
    pair_mask = np.zeros(length)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_idx[i] = j
                pair_idx[j] = i
                pair_mask[i] = 1.0
                pair_mask[j] = 1.0

    return pair_idx, pair_mask


def one_hot_encode(seq, token_map, length):
    """
    One-hot encodes a sequence string based on a token map.
    Returns shape (length, num_tokens)
    """
    num_tokens = len(token_map)
    encoding = np.zeros((length, num_tokens), dtype=np.float32)

    for i, char in enumerate(seq):
        if i >= length:
            break
        if char in token_map:
            encoding[i, token_map[char]] = 1.0

    return encoding


def process_dataframe(df, mode="train"):
    """
    Process raw dataframe into numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH
    input_channels = Config.INPUT_CHANNELS

    # Initialize arrays
    inputs = np.zeros((num_samples, seq_len, input_channels), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Process inputs
    for idx, row in df.iterrows():
        # 1. Sequence (4 dims)
        seq_enc = one_hot_encode(row["sequence"], Config.TOKEN_MAP_SEQ, seq_len)

        # 2. Structure (3 dims)
        struct_enc = one_hot_encode(row["structure"], Config.TOKEN_MAP_STRUCT, seq_len)

        # 3. Loop Type (7 dims)
        loop_enc = one_hot_encode(
            row["predicted_loop_type"], Config.TOKEN_MAP_LOOP, seq_len
        )

        # Concatenate features
        inputs[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # 4. Adjacency
        p_idx, p_mask = get_structure_adj(row["structure"], seq_len)
        pair_indices[idx] = p_idx
        pair_masks[idx] = p_mask

    # Process targets if available
    targets = None
    if mode in ["train", "val"]:
        target_cols = Config.TARGET_COLS
        # Targets are lists of length 68. Stack them.
        # Shape: (N, 68, 5)

        # Extract lists efficiently
        t_arrays = []
        for col in target_cols:
            # Convert column of lists to 2D array
            col_data = np.vstack(df[col].values)
            t_arrays.append(col_data)

        # Stack along last axis -> (N, 68, 5)
        targets = np.stack(t_arrays, axis=2).astype(np.float32)

    ids = df["id"].tolist()

    return inputs, pair_indices, pair_masks, targets, ids


def load_or_process_data(parquet_path, cache_name, load_cached_data=True):
    """
    Loads data from parquet, processes it, and caches it.
    """
    Config.setup_directories()

    # Load raw dataframe
    df = pd.read_parquet(parquet_path)

    # Create hash of the dataframe IDs to ensure data consistency
    ids_hash = hashlib.md5(pd.util.hash_pandas_object(df["id"]).values).hexdigest()
    cache_filename = f"{cache_name}_{ids_hash}.npz"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        inputs = data["inputs"]
        pair_indices = data["pair_indices"]
        pair_masks = data["pair_masks"]
        # Handle targets being None (saved as None object array or omitted)
        if "targets" in data and data["targets"].shape != ():
            targets = data["targets"]
        else:
            targets = None
        ids = data["ids"].tolist()
        return inputs, pair_indices, pair_masks, targets, ids

    print(f"Processing data from {parquet_path}...")
    mode = "test" if "test" in cache_name else "train"
    inputs, pair_indices, pair_masks, targets, ids = process_dataframe(df, mode=mode)

    print(f"Saving cache to {cache_path}")
    save_dict = {
        "inputs": inputs,
        "pair_indices": pair_indices,
        "pair_masks": pair_masks,
        "ids": np.array(ids),
    }
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return inputs, pair_indices, pair_masks, targets, ids


def get_dataloaders(
    load_cached_data=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Returns train, val, test dataloaders.
    """
    # Load Train
    train_inputs, train_pairs, train_masks, train_targets, train_ids = (
        load_or_process_data(Config.TRAIN_METADATA, "train_data", load_cached_data)
    )
    train_dataset = RNADataset(
        train_inputs, train_pairs, train_masks, train_targets, train_ids
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Val
    val_inputs, val_pairs, val_masks, val_targets, val_ids = load_or_process_data(
        Config.VAL_METADATA, "val_data", load_cached_data
    )
    val_dataset = RNADataset(val_inputs, val_pairs, val_masks, val_targets, val_ids)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Test
    test_inputs, test_pairs, test_masks, test_targets, test_ids = load_or_process_data(
        Config.TEST_METADATA, "test_data", load_cached_data
    )
    test_dataset = RNADataset(
        test_inputs, test_pairs, test_masks, targets=None, ids=test_ids
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
