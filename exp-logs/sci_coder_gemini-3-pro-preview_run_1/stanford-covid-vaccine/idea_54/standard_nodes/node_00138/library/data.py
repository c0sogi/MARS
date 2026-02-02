import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def get_structure_distance(structure):
    """
    Parses dot-bracket structure and returns a signed distance array.
    If base i is paired with j, dist = j - i.
    If unpaired, dist = 0.
    """
    stack = []
    indices = np.zeros(len(structure), dtype=np.int32)

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is upstream (smaller), i is downstream (larger)
                # For j: pair is i, dist = i - j (positive)
                # For i: pair is j, dist = j - i (negative)
                indices[j] = i - j
                indices[i] = j - i
    return indices


def process_data(df, is_test=False):
    """
    Tokenizes sequences and loops, computes structure distances, and extracts targets.
    """
    # Token Maps
    token2int = {x: i for i, x in enumerate("AGUC")}
    loop2int = {x: i for i, x in enumerate("BEHIMSX")}

    ids = df["id"].values
    sequences = []
    loops = []
    distances = []

    for _, row in df.iterrows():
        # Sequence Tokenization
        seq = [token2int.get(x, 0) for x in row["sequence"]]
        sequences.append(seq)

        # Loop Tokenization
        lp = [loop2int.get(x, 0) for x in row["predicted_loop_type"]]
        loops.append(lp)

        # Structure Distance
        dist = get_structure_distance(row["structure"])
        distances.append(dist)

    sequences = np.array(sequences, dtype=np.int32)
    loops = np.array(loops, dtype=np.int32)
    distances = np.array(distances, dtype=np.int32)

    if is_test:
        return ids, sequences, loops, distances

    # Process Targets
    # Config.TARGET_COLS defines the subset of columns to train on
    targets = []
    for col in Config.TARGET_COLS:
        # Each row in df[col] is a list/array of floats
        val = np.vstack(df[col].values)
        targets.append(val)

    # Stack to shape (N, 68, 3)
    # Original lists are 1x68, vstack makes them N x 68.
    # We stack along axis 2 to get channels.
    targets = np.stack(targets, axis=2)

    return ids, sequences, loops, distances, targets


def load_data(mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches it using .npy files.
    mode: 'train', 'val', or 'test'
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_prefix = os.path.join(Config.CACHE_DIR, f"{mode}_data")
    files = {
        "ids": f"{cache_prefix}_ids.npy",
        "seq": f"{cache_prefix}_seq.npy",
        "loop": f"{cache_prefix}_loop.npy",
        "dist": f"{cache_prefix}_dist.npy",
        "tgt": f"{cache_prefix}_tgt.npy",
    }

    # Determine required keys
    required_keys = ["ids", "seq", "loop", "dist"]
    if mode != "test":
        required_keys.append("tgt")

    # Check if cache exists
    all_exist = True
    if load_cached_data:
        for k in required_keys:
            if not os.path.exists(files[k]):
                all_exist = False
                break
    else:
        all_exist = False

    # Load from cache
    if all_exist:
        print(f"Loading cached {mode} data...")
        ids = np.load(files["ids"], allow_pickle=True)
        seq = np.load(files["seq"])
        loop = np.load(files["loop"])
        dist = np.load(files["dist"])

        if mode == "test":
            return ids, seq, loop, dist
        else:
            tgt = np.load(files["tgt"])
            return ids, seq, loop, dist, tgt

    # Process from scratch
    print(f"Processing {mode} data from scratch...")
    if mode == "train":
        source_path = Config.TRAIN_METADATA
    elif mode == "val":
        source_path = Config.VAL_METADATA
    else:
        source_path = Config.TEST_METADATA

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_parquet(source_path)

    if mode == "test":
        ids, seq, loop, dist = process_data(df, is_test=True)
        np.save(files["ids"], ids)
        np.save(files["seq"], seq)
        np.save(files["loop"], loop)
        np.save(files["dist"], dist)
        return ids, seq, loop, dist
    else:
        ids, seq, loop, dist, tgt = process_data(df, is_test=False)
        np.save(files["ids"], ids)
        np.save(files["seq"], seq)
        np.save(files["loop"], loop)
        np.save(files["dist"], dist)
        np.save(files["tgt"], tgt)
        return ids, seq, loop, dist, tgt


class RNADataset(Dataset):
    def __init__(self, sequences, loops, distances, targets=None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.loops = torch.tensor(loops, dtype=torch.long)
        # Distances are converted to float for Sinusoidal Embeddings
        self.distances = torch.tensor(distances, dtype=torch.float)
        self.targets = (
            torch.tensor(targets, dtype=torch.float) if targets is not None else None
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "sequence": self.sequences[idx],
            "loop": self.loops[idx],
            "distance": self.distances[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item
