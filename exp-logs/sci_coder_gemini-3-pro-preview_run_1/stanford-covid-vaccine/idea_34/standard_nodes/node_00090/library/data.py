import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_bond_type_tokens(sequence, structure):
    """
    Generates bond type tokens for each nucleotide based on the secondary structure.

    Mappings (from Config.bond2id):
    - A-U: 0, U-A: 1
    - G-C: 2, C-G: 3
    - G-U: 4, U-G: 5
    - Mismatch: 6
    - Unpaired: 7
    """
    length = len(sequence)
    tokens = np.full(length, Config.bond2id["Unpaired"], dtype=np.int64)

    # Parse structure to find pairs
    stack = []
    pairs = {}
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pairs[start] = i
                pairs[i] = start

    # Assign tokens for paired bases
    for i, j in pairs.items():
        # Determine pair type based on sequence identity
        base_i = sequence[i]
        base_j = sequence[j]

        # Construct key for bond2id (e.g., "A-U")
        # Note: Direction implies specific token (A-U vs U-A)
        pair_key = f"{base_i}-{base_j}"

        if pair_key in Config.bond2id:
            tokens[i] = Config.bond2id[pair_key]
        else:
            tokens[i] = Config.bond2id["Mismatch"]

    return tokens


def get_distance_vector(structure):
    """
    Computes the signed distance (j - i) for each paired nucleotide.
    Unpaired bases have a distance of 0.
    """
    length = len(structure)
    dist = np.zeros(length, dtype=np.float32)

    stack = []
    pairs = {}
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                pairs[start] = i
                pairs[i] = start

    for i, j in pairs.items():
        dist[i] = float(j - i)

    return dist


def process_data(parquet_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes features, and caches the result.

    Args:
        parquet_path (str): Path to the input parquet file.
        cache_path (str): Path to save/load the processed .pt file.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed tensors.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = torch.load(cache_path)
            # Verify cache schema matches current code expectations
            required_keys = ["seq", "loop", "bond", "dist", "ids", "targets"]
            if all(key in data for key in required_keys):
                # print(f"Loaded {mode} data from cache: {cache_path}")
                return data
            print(f"Cache {cache_path} has stale schema. Reprocessing...")
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Initialize containers
    num_samples = len(df)
    seq_len = Config.seq_len

    # Feature arrays
    seq_tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    loop_tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    bond_tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    dist_vectors = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Iterate and process each sample
    for idx, row in df.iterrows():
        # Sequence
        seq_tokens[idx] = np.array(
            [Config.token2id.get(base, 0) for base in row["sequence"]]
        )

        # Loop Type
        loop_tokens[idx] = np.array(
            [Config.loop2id.get(loop, 0) for loop in row["predicted_loop_type"]]
        )

        # Bond Type (Soft Feature)
        bond_tokens[idx] = get_bond_type_tokens(row["sequence"], row["structure"])

        # Distance Vector
        dist_vectors[idx] = get_distance_vector(row["structure"])

    data_dict = {
        "seq": torch.tensor(seq_tokens, dtype=torch.long),
        "loop": torch.tensor(loop_tokens, dtype=torch.long),
        "bond": torch.tensor(bond_tokens, dtype=torch.long),
        "dist": torch.tensor(dist_vectors, dtype=torch.float32),
        "ids": df["id"].values.tolist(),
    }

    # Process Targets (only if available)
    # Check if target columns exist in the dataframe
    has_targets = all(col in df.columns for col in Config.target_cols)

    if has_targets:
        # Stack targets: (N, 68, 3)
        target_arrays = []
        for col in Config.target_cols:
            # Each row contains a list/array of length 68
            # np.vstack converts list of arrays to (N, 68)
            col_data = np.vstack(df[col].values)
            target_arrays.append(col_data)

        # Stack along last dimension -> (N, 68, Num_Targets)
        targets = np.stack(target_arrays, axis=2)
        data_dict["targets"] = torch.tensor(targets, dtype=torch.float32)
    else:
        # For test set, create dummy targets or None
        # We create dummy targets of shape (N, 68, 3) to prevent errors in generic loops
        data_dict["targets"] = torch.zeros(
            (num_samples, Config.pred_len, Config.num_targets), dtype=torch.float32
        )

    # 3. Save to cache
    torch.save(data_dict, cache_path)
    # print(f"Processed and saved {mode} data to {cache_path}")

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.bond = data_dict["bond"]
        self.dist = data_dict["dist"]
        self.targets = data_dict["targets"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return {
            "seq": self.seq[idx],
            "loop": self.loop[idx],
            "bond": self.bond[idx],
            "dist": self.dist[idx],
            "targets": self.targets[idx],
            "id": self.ids[idx],
        }


def get_dataloaders(debug=Config.debug):
    """
    Prepares DataLoaders for train, val, and test sets.
    Handles caching and debug slicing.
    """
    # 1. Process Data
    train_data = process_data(Config.train_file, Config.train_cache, mode="train")
    val_data = process_data(Config.val_file, Config.val_cache, mode="val")
    test_data = process_data(Config.test_file, Config.test_cache, mode="test")

    # 2. Handle Debug Mode (Slice data)
    if debug:
        subset_size = Config.batch_size * 2

        def slice_dict(d, size):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    new_d[k] = v[:size]
                elif isinstance(v, list):
                    new_d[k] = v[:size]
            return new_d

        train_data = slice_dict(train_data, subset_size)
        val_data = slice_dict(val_data, subset_size)
        test_data = slice_dict(test_data, subset_size)
        # print(f"Debug mode: Sliced datasets to {subset_size} samples.")

    # 3. Create Datasets
    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)
    test_dataset = RNADataset(test_data)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
