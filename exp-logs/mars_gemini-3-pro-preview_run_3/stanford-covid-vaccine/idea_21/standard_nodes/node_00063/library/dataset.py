import os
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string into an adjacency array.
    adj[i] = j if base i is paired with base j.
    adj[i] = -1 if base i is unpaired.
    """
    n = len(structure)
    adj = np.full(n, -1, dtype=np.int32)
    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i
    return adj


def process_dataframe(df, mode="train"):
    """
    Converts a pandas DataFrame into a dictionary of numpy arrays
    suitable for training or inference.
    """
    # 1. Sequence Encoding (One-Hot)
    # A:0, G:1, C:2, U:3
    seq_map = {"A": 0, "G": 1, "C": 2, "U": 3}
    seqs = np.array([[seq_map.get(c, 0) for c in s] for s in df["sequence"]])

    # 2. Structure Encoding (One-Hot)
    # .:0, (:1, ):2
    struct_map = {".": 0, "(": 1, ")": 2}
    structs = np.array([[struct_map.get(c, 0) for c in s] for s in df["structure"]])

    # 3. Loop Type Encoding (One-Hot)
    # S:0, M:1, I:2, B:3, H:4, E:5, X:6
    loop_map = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    loops = np.array(
        [[loop_map.get(c, 0) for c in s] for s in df["predicted_loop_type"]]
    )

    # 4. Adjacency Matrix for Structural Injection
    adj = np.array([get_structure_adj(s) for s in df["structure"]])

    # Combine Features into (N, L, 14)
    N, L = seqs.shape
    features = np.zeros((N, L, Config.INPUT_DIM), dtype=np.float32)

    # Fill Sequence (0-3)
    for i in range(4):
        features[:, :, i] = seqs == i

    # Fill Structure (4-6)
    for i in range(3):
        features[:, :, 4 + i] = structs == i

    # Fill Loop (7-13)
    for i in range(7):
        features[:, :, 7 + i] = loops == i

    data = {"features": features, "adj": adj, "ids": df["id"].values}

    # Process Targets if not test
    if mode != "test":
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Targets are stored as lists in the dataframe, need to stack them
        # Shape: (N, SEQ_SCORED, 5)
        y = np.zeros((N, Config.SEQ_SCORED, 5), dtype=np.float32)
        for i, col in enumerate(target_cols):
            col_data = np.array(df[col].tolist())
            y[:, :, i] = col_data
        data["targets"] = y

    return data


def get_data_hash(mode):
    """
    Generates a unique hash based on the configuration and mode
    to ensure cache validity.
    """
    config_dict = {
        "SEQ_LEN": Config.SEQ_LEN,
        "SEQ_SCORED": Config.SEQ_SCORED,
        "INPUT_DIM": Config.INPUT_DIM,
        "mode": mode,
        "IDEA_NAME": Config.IDEA_NAME,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.features = torch.from_numpy(data_dict["features"]).float()
        self.adj = torch.from_numpy(data_dict["adj"]).long()
        self.ids = data_dict["ids"]
        self.mode = mode

        if mode != "test":
            self.targets = torch.from_numpy(data_dict["targets"]).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {"x": self.features[idx], "adj": self.adj[idx], "id": self.ids[idx]}
        if self.mode != "test":
            item["y"] = self.targets[idx]
        return item


def get_dataset(mode="train", load_cached_data=True):
    """
    Loads the dataset for the specified mode.
    Uses hash-based caching to speed up loading.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Generate cache filename
    data_hash = get_data_hash(mode)
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_data_{data_hash}.npz")

    if load_cached_data and os.path.exists(cache_file):
        # print(f"Loading cached {mode} data from {cache_file}...")
        try:
            loaded = np.load(cache_file, allow_pickle=True)
            data_dict = {k: loaded[k] for k in loaded.files}
            return RNADataset(data_dict, mode)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    # print(f"Processing {mode} data from metadata...")
    if mode == "train":
        df = pd.read_parquet(Config.TRAIN_META)
    elif mode == "val":
        df = pd.read_parquet(Config.VAL_META)
    else:
        df = pd.read_parquet(Config.TEST_META)

    data_dict = process_dataframe(df, mode)

    # Save to cache
    np.savez(cache_file, **data_dict)
    # print(f"Saved {mode} data cache to {cache_file}")

    return RNADataset(data_dict, mode)
