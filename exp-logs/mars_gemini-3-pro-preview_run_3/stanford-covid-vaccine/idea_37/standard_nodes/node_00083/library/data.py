import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.

    Returns:
        inputs (torch.Tensor): Shape (seq_len, 14) - One-hot encoded features.
        pair_indices (torch.Tensor): Shape (seq_len,) - Indices of paired bases.
        pair_mask (torch.Tensor): Shape (seq_len,) - 1.0 if paired, 0.0 otherwise.
        targets (torch.Tensor): Shape (pred_len, 5) - Ground truth values (or dummies).
        ids (str): Sample ID.
    """

    def __init__(self, inputs, pair_indices, pair_mask, targets, ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.pair_mask = torch.tensor(pair_mask, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.pair_indices[idx],
            self.pair_mask[idx],
            self.targets[idx],
            self.ids[idx],
        )


def get_structure_adj(structure, seq_len):
    """
    Parses dot-bracket structure to get pair indices and mask.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").
        seq_len (int): Length of sequence.

    Returns:
        pair_indices (np.array): Index of partner. Points to self if unpaired.
        mask (np.array): 1.0 if paired, 0.0 if unpaired.
    """
    pair_indices = np.arange(seq_len)
    mask = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
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


def preprocess_data(df, config, is_test=False):
    """
    Converts DataFrame columns to numpy arrays for the model.
    """
    num_samples = len(df)
    seq_len = config.seq_len
    pred_len = config.pred_len

    # Mappings
    seq_map = {c: i for i, c in enumerate("AGUC")}
    struct_map = {c: i for i, c in enumerate("().")}
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    # Initialize arrays
    # 14 channels: 4 seq + 3 struct + 7 loop
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_mask = np.zeros((num_samples, seq_len), dtype=np.float32)
    targets = np.zeros((num_samples, pred_len, 5), dtype=np.float32)
    ids = df["id"].values

    # Process each sample
    for idx, row in df.iterrows():
        # 1. Sequence One-Hot (0-3)
        seq = row["sequence"]
        for i, char in enumerate(seq):
            if char in seq_map:
                inputs[idx, i, seq_map[char]] = 1.0

        # 2. Structure One-Hot (4-6)
        struct = row["structure"]
        for i, char in enumerate(struct):
            if char in struct_map:
                inputs[idx, i, 4 + struct_map[char]] = 1.0

        # 3. Loop Type One-Hot (7-13)
        loop = row["predicted_loop_type"]
        for i, char in enumerate(loop):
            if char in loop_map:
                inputs[idx, i, 7 + loop_map[char]] = 1.0

        # 4. Adjacency / Pairing
        p_idx, p_mask = get_structure_adj(struct, seq_len)
        pair_indices[idx] = p_idx
        pair_mask[idx] = p_mask

        # 5. Targets
        if not is_test:
            # Targets are lists of length 68
            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            t_list = [
                row["reactivity"],
                row["deg_Mg_pH10"],
                row["deg_pH10"],
                row["deg_Mg_50C"],
                row["deg_50C"],
            ]
            # Transpose to (68, 5)
            targets[idx] = np.array(t_list, dtype=np.float32).T
        else:
            # Keep zeros for test
            pass

    return inputs, pair_indices, pair_mask, targets, ids


def load_or_process_data(split_name, parquet_path, config, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from Parquet and caches it.
    """
    cache_path = os.path.join(config.working_dir, f"{split_name}_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["inputs"],
            data["pair_indices"],
            data["pair_mask"],
            data["targets"],
            data["ids"],
        )

    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Determine if test set (no targets in metadata)
    # The test.parquet schema does not have target columns.
    is_test = split_name == "test"

    inputs, pair_indices, pair_mask, targets, ids = preprocess_data(
        df, config, is_test=is_test
    )

    # Save to cache
    os.makedirs(config.working_dir, exist_ok=True)
    np.savez(
        cache_path,
        inputs=inputs,
        pair_indices=pair_indices,
        pair_mask=pair_mask,
        targets=targets,
        ids=ids,
    )
    print(f"Saved {split_name} data to cache: {cache_path}")

    return inputs, pair_indices, pair_mask, targets, ids


def get_dataloaders(config, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Train
    train_data = load_or_process_data(
        "train", config.train_file, config, load_cached_data
    )
    train_dataset = RNADataset(*train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Validation
    val_data = load_or_process_data("val", config.val_file, config, load_cached_data)
    val_dataset = RNADataset(*val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Test
    test_data = load_or_process_data("test", config.test_file, config, load_cached_data)
    test_dataset = RNADataset(*test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
