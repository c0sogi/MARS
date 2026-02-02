import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config, parse_structure_distances


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Stores pre-processed tensors for sequence, loop type, and distance map.
    """

    def __init__(self, ids, seq, loop, dist, targets=None):
        self.ids = ids
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.loop = torch.tensor(loop, dtype=torch.long)
        self.dist = torch.tensor(dist, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {"seq": self.seq[idx], "loop": self.loop[idx], "dist": self.dist[idx]}
        if self.targets is not None:
            sample["targets"] = self.targets[idx]
        return sample


def collate_fn(batch):
    """
    Custom collate function to stack dictionary items into batches.
    """
    seq = torch.stack([item["seq"] for item in batch])
    loop = torch.stack([item["loop"] for item in batch])
    dist = torch.stack([item["dist"] for item in batch])

    collated = {"seq": seq, "loop": loop, "dist": dist}

    if "targets" in batch[0]:
        targets = torch.stack([item["targets"] for item in batch])
        collated["targets"] = targets

    return collated


def process_dataframe(df, mode="train"):
    """
    Tokenizes sequences and loop types, and parses structure distances.
    Returns numpy arrays ready for the Dataset.
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    ids = df["id"].values

    # Sequences: Map chars to indices
    sequences = df["sequence"].values
    seq_encoded = np.array(
        [[seq_map.get(c, 0) for c in s] for s in sequences], dtype=np.int32
    )

    # Loop Types: Map chars to indices
    loops = df["predicted_loop_type"].values
    loop_encoded = np.array(
        [[loop_map.get(c, 0) for c in l] for l in loops], dtype=np.int32
    )

    # Structure Distances: Parse dot-bracket to signed distances
    structures = df["structure"].values
    dist_encoded = np.array(
        [parse_structure_distances(s) for s in structures], dtype=np.int32
    )

    if mode in ["train", "val"]:
        # Targets: Stack specific target columns
        targets = []
        for col in Config.TARGET_COLS:
            # Each row in df[col] is a list/array of length 68
            col_data = np.vstack(df[col].values)
            targets.append(col_data)
        targets = np.stack(targets, axis=2)  # Shape: (N, 68, 3)
        return ids, seq_encoded, loop_encoded, dist_encoded, targets
    else:
        return ids, seq_encoded, loop_encoded, dist_encoded


def get_data_split(mode, load_cached_data=True):
    """
    Loads, processes, and caches data for a specific split (train/val/test).
    Strictly follows caching logic using numpy archives.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"cached_{mode}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if mode in ["train", "val"]:
            return data["ids"], data["seq"], data["loop"], data["dist"], data["targets"]
        else:
            return data["ids"], data["seq"], data["loop"], data["dist"]

    # 2. Process from scratch if cache missing or forced reload
    print(f"Processing {mode} data from scratch...")
    parquet_path = os.path.join(Config.METADATA_DIR, f"{mode}.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    processed = process_dataframe(df, mode)

    # 3. Save to cache
    if mode in ["train", "val"]:
        ids, seq, loop, dist, targets = processed
        np.savez(cache_path, ids=ids, seq=seq, loop=loop, dist=dist, targets=targets)
        return ids, seq, loop, dist, targets
    else:
        ids, seq, loop, dist = processed
        np.savez(cache_path, ids=ids, seq=seq, loop=loop, dist=dist)
        return ids, seq, loop, dist


def get_loaders(load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test splits.
    """
    # Train Loader
    train_ids, train_seq, train_loop, train_dist, train_targets = get_data_split(
        "train", load_cached_data
    )
    train_ds = RNADataset(train_ids, train_seq, train_loop, train_dist, train_targets)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Validation Loader
    val_ids, val_seq, val_loop, val_dist, val_targets = get_data_split(
        "val", load_cached_data
    )
    val_ds = RNADataset(val_ids, val_seq, val_loop, val_dist, val_targets)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test Loader
    test_ids, test_seq, test_loop, test_dist = get_data_split("test", load_cached_data)
    test_ds = RNADataset(test_ids, test_seq, test_loop, test_dist, targets=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
