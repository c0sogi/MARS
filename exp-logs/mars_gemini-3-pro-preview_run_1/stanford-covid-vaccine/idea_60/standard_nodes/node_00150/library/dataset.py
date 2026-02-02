import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config, get_structure_pairs


def process_data(df, mode="train"):
    """
    Processes DataFrame into numpy arrays for the model.
    Tokenizes sequences and loop types, and computes signed distance matrices.
    For training/validation, extracts and stacks specific target columns.

    Args:
        df (pd.DataFrame): Input dataframe containing sequences and metadata.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        Tuple of numpy arrays: (ids, X_seq, X_loop, X_dist, [y])
    """
    # Mappings
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    loop_map = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}

    ids = df["id"].values
    sequences = df["sequence"].values
    structures = df["structure"].values
    loops = df["predicted_loop_type"].values

    # Pre-allocate arrays
    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    X_seq = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_loop = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_dist = np.zeros((n_samples, seq_len), dtype=np.float32)

    for i in range(n_samples):
        # Sequence Tokenization
        # Truncate or pad if necessary (though data is fixed length)
        curr_seq = sequences[i][:seq_len]
        X_seq[i, : len(curr_seq)] = [seq_map.get(c, 0) for c in curr_seq]

        # Loop Type Tokenization
        curr_loop = loops[i][:seq_len]
        X_loop[i, : len(curr_loop)] = [loop_map.get(c, 0) for c in curr_loop]

        # Structure Distance Matrix
        # get_structure_pairs returns {index: paired_index}
        pairs = get_structure_pairs(structures[i])
        for j in range(seq_len):
            if j in pairs:
                # Signed distance: current_index - paired_index
                X_dist[i, j] = float(j - pairs[j])
            else:
                # Unpaired bases have 0 distance
                X_dist[i, j] = 0.0

    if mode in ["train", "val"]:
        # Targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Explicitly discard deg_pH10 and deg_50C as per instructions

        # Stack lists into numpy arrays
        y_reactivity = np.vstack(df["reactivity"].values)
        y_deg_Mg_pH10 = np.vstack(df["deg_Mg_pH10"].values)
        y_deg_Mg_50C = np.vstack(df["deg_Mg_50C"].values)

        # Stack channels: (N, 68, 3)
        y = np.stack([y_reactivity, y_deg_Mg_pH10, y_deg_Mg_50C], axis=2)

        return ids, X_seq, X_loop, X_dist, y
    else:
        return ids, X_seq, X_loop, X_dist


def get_dataset(path, mode="train", load_cached_data=True):
    """
    Loads data from Parquet files or retrieves it from the cache.

    Args:
        path (str): Path to the parquet file.
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Tuple of numpy arrays ready for the Dataset class.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True)
            if mode in ["train", "val"]:
                return (
                    data["ids"],
                    data["X_seq"],
                    data["X_loop"],
                    data["X_dist"],
                    data["y"],
                )
            else:
                return data["ids"], data["X_seq"], data["X_loop"], data["X_dist"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from {path}...")
    df = pd.read_parquet(path)
    processed = process_data(df, mode)

    # 3. Save to cache and return
    if mode in ["train", "val"]:
        ids, X_seq, X_loop, X_dist, y = processed
        np.savez(cache_file, ids=ids, X_seq=X_seq, X_loop=X_loop, X_dist=X_dist, y=y)
        return ids, X_seq, X_loop, X_dist, y
    else:
        ids, X_seq, X_loop, X_dist = processed
        np.savez(cache_file, ids=ids, X_seq=X_seq, X_loop=X_loop, X_dist=X_dist)
        return ids, X_seq, X_loop, X_dist


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA data.
    """

    def __init__(self, X_seq, X_loop, X_dist, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.long)
        self.X_loop = torch.tensor(X_loop, dtype=torch.long)
        self.X_dist = torch.tensor(X_dist, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_seq[idx], self.X_loop[idx], self.X_dist[idx], self.y[idx]
        return self.X_seq[idx], self.X_loop[idx], self.X_dist[idx]
