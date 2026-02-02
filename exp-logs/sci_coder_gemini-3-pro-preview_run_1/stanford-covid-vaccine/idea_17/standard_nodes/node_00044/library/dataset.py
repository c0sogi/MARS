import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import ModelConfig, get_distance_vector


def process_data(df, mode="train"):
    """
    Converts dataframe columns to numpy arrays suitable for training/inference.

    Args:
        df (pd.DataFrame): The input dataframe containing sequences and structures.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: A dictionary containing numpy arrays for inputs and targets.
    """
    # Mappings
    token_map = {c: i for i, c in enumerate(["A", "G", "U", "C"])}
    loop_map = {c: i for i, c in enumerate(["S", "M", "I", "B", "H", "E", "X"])}

    seqs = df["sequence"].values
    structs = df["structure"].values
    loops = df["predicted_loop_type"].values
    ids = df["id"].values

    n_samples = len(seqs)
    seq_len = 107

    # Pre-allocate arrays
    X_seq = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_loop = np.zeros((n_samples, seq_len), dtype=np.int32)
    X_dist = np.zeros((n_samples, seq_len), dtype=np.float32)
    X_mask = np.zeros((n_samples, seq_len), dtype=np.float32)

    for i in range(n_samples):
        # Sequence Tokenization
        X_seq[i] = [token_map.get(c, 0) for c in seqs[i]]

        # Loop Type Tokenization
        X_loop[i] = [loop_map.get(c, 0) for c in loops[i]]

        # Distance Vector and Mask (using helper from library.config)
        # d[k] = j - k if k is paired with j, else 0
        d, m = get_distance_vector(structs[i], seq_len)
        X_dist[i] = d
        X_mask[i] = m

    data_dict = {
        "seq": X_seq,
        "loop": X_loop,
        "dist": X_dist,
        "mask": X_mask,
        "id": ids,
    }

    if mode in ["train", "val"]:
        # Stack targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        # The parquet files store these as lists/arrays within the cell.
        # np.vstack converts the column of lists into a 2D array (N, 68)
        reactivity = np.vstack(df["reactivity"].values)
        deg_Mg_pH10 = np.vstack(df["deg_Mg_pH10"].values)
        deg_Mg_50C = np.vstack(df["deg_Mg_50C"].values)

        # Shape: (N, 68, 3)
        targets = np.stack([reactivity, deg_Mg_pH10, deg_Mg_50C], axis=2).astype(
            np.float32
        )
        data_dict["target"] = targets

    return data_dict


def load_or_process_data(load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes raw parquet files
    and saves to cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_data, val_data, test_data) dictionaries.
    """
    # Ensure output directory exists
    os.makedirs(ModelConfig.output_dir, exist_ok=True)

    files = {
        "train": (
            ModelConfig.train_file,
            os.path.join(ModelConfig.output_dir, "train_data.npz"),
        ),
        "val": (
            ModelConfig.val_file,
            os.path.join(ModelConfig.output_dir, "val_data.npz"),
        ),
        "test": (
            ModelConfig.test_file,
            os.path.join(ModelConfig.output_dir, "test_data.npz"),
        ),
    }

    datasets = {}

    for mode, (input_path, cache_path) in files.items():
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached {mode} data from {cache_path}...")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                # Convert NpzFile object back to dict
                data_dict = {k: loaded[k] for k in loaded.files}
                datasets[mode] = data_dict
                continue
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Reprocessing...")

        # 2. Process from scratch
        print(f"Processing {mode} data from {input_path}...")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_parquet(input_path)
        data_dict = process_data(df, mode=mode)

        # Save to cache
        np.savez(cache_path, **data_dict)
        datasets[mode] = data_dict

    return datasets["train"], datasets["val"], datasets["test"]


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.dist = data_dict["dist"]
        self.mask = data_dict["mask"]
        self.ids = data_dict["id"]

        if mode in ["train", "val"]:
            self.target = data_dict["target"]

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        item = {
            "seq": torch.tensor(self.seq[idx], dtype=torch.long),
            "loop": torch.tensor(self.loop[idx], dtype=torch.long),
            "dist": torch.tensor(self.dist[idx], dtype=torch.float32),
            "mask": torch.tensor(self.mask[idx], dtype=torch.float32),
            "id": str(self.ids[idx]),
        }

        if self.mode in ["train", "val"]:
            item["target"] = torch.tensor(self.target[idx], dtype=torch.float32)

        return item
