import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
# Bond types: Watson-Crick (GC, AU) and Wobble (GU), plus reverse and None
BOND_MAP = {"GC": 0, "CG": 1, "AU": 2, "UA": 3, "GU": 4, "UG": 5, "None": 6}

# =========================================================================
# Helper Functions
# =========================================================================


def process_structure(structure):
    """
    Parses a dot-bracket structure string into a pair map.
    Returns an array where arr[i] = j if i is paired with j, else -1.
    """
    n = len(structure)
    pair_map = np.full(n, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i

    return pair_map


def get_bond_types(sequence, pair_map):
    """
    Generates bond type indices for the sequence based on the pair map.
    """
    n = len(sequence)
    bond_types = np.full(n, BOND_MAP["None"], dtype=int)

    for i in range(n):
        j = pair_map[i]
        if j != -1:
            # Construct pair string, e.g., "GC"
            pair_str = sequence[i] + sequence[j]
            # Map to index, default to 'None' if non-canonical pair found
            bond_types[i] = BOND_MAP.get(pair_str, BOND_MAP["None"])

    return bond_types


def get_distance_encoding(pair_map, dim):
    """
    Generates Signed Sinusoidal Positional Encodings for pairing distances.
    Distance d = j - i. If unpaired, d = 0.

    Args:
        pair_map (np.ndarray): Array of pair indices.
        dim (int): Dimension of the embedding (must be even).

    Returns:
        np.ndarray: Shape (seq_len, dim)
    """
    seq_len = len(pair_map)
    indices = np.arange(seq_len)

    # Calculate signed distance: j - i if paired, else 0
    # Where pair_map[i] == -1, result is -1 - i, which is not 0.
    # We need to mask explicitly.
    paired_mask = pair_map != -1
    distances = np.zeros(seq_len, dtype=float)
    distances[paired_mask] = pair_map[paired_mask] - indices[paired_mask]

    # Sinusoidal encoding
    # PE(pos, 2i) = sin(pos / 10000^(2i/dim))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    encoding = np.zeros((seq_len, dim), dtype=np.float32)
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))

    # Expand dims for broadcasting
    # distances: (L, 1), div_term: (1, D/2)
    dist_expanded = distances[:, np.newaxis]
    div_term_expanded = div_term[np.newaxis, :]

    phase = dist_expanded * div_term_expanded

    encoding[:, 0::2] = np.sin(phase)
    encoding[:, 1::2] = np.cos(phase)

    return encoding


# =========================================================================
# Dataset Class
# =========================================================================


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.bond = data_dict["bond"]
        self.dist = data_dict["dist"]
        self.ids = data_dict["ids"]

        if mode != "test":
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        # Inputs
        seq_idx = torch.tensor(self.seq[idx], dtype=torch.long)
        loop_idx = torch.tensor(self.loop[idx], dtype=torch.long)
        bond_idx = torch.tensor(self.bond[idx], dtype=torch.long)
        dist_feat = torch.tensor(self.dist[idx], dtype=torch.float32)

        inputs = {"seq": seq_idx, "loop": loop_idx, "bond": bond_idx, "dist": dist_feat}

        if self.mode == "test":
            return inputs, self.ids[idx]

        # Targets
        # Shape: (68, 3)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return inputs, target


# =========================================================================
# Data Processing Pipeline
# =========================================================================


def preprocess_data(df, mode="train"):
    """
    Extracts features and targets from the DataFrame.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    seq_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    bond_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    dist_arr = np.zeros(
        (num_samples, seq_len, Config.EMBED_DIM_DISTANCE), dtype=np.float32
    )

    ids_list = df["id"].values

    # Process features
    for idx, row in df.iterrows():
        # 1. Sequence
        sequence = row["sequence"]
        seq_arr[idx] = [
            SEQ_MAP.get(c, 0) for c in sequence
        ]  # Default to A if unknown (unlikely)

        # 2. Loop Type
        loop_type = row["predicted_loop_type"]
        loop_arr[idx] = [LOOP_MAP.get(c, 6) for c in loop_type]  # Default to X

        # 3. Structure Parsing
        structure = row["structure"]
        pair_map = process_structure(structure)

        # 4. Bond Types
        bond_arr[idx] = get_bond_types(sequence, pair_map)

        # 5. Distance Encoding
        dist_arr[idx] = get_distance_encoding(pair_map, Config.EMBED_DIM_DISTANCE)

    # Process targets (only for train/val)
    targets_arr = None
    if mode != "test":
        # Targets are lists of length 68 in the parquet file
        # We need to stack them: (N, 68, 3)
        # Columns: reactivity, deg_Mg_pH10, deg_Mg_50C

        # Extract columns as list of lists then convert to array
        t1 = np.vstack(df["reactivity"].values)
        t2 = np.vstack(df["deg_Mg_pH10"].values)
        t3 = np.vstack(df["deg_Mg_50C"].values)

        # Stack along the last axis
        targets_arr = np.stack([t1, t2, t3], axis=2).astype(np.float32)

    return {
        "seq": seq_arr,
        "loop": loop_arr,
        "bond": bond_arr,
        "dist": dist_arr,
        "ids": ids_list,
        "targets": targets_arr,
    }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching of preprocessed numpy arrays.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    loaders = {}
    modes = [
        ("train", Config.TRAIN_DATA_PATH),
        ("val", Config.VAL_DATA_PATH),
        ("test", Config.TEST_DATA_PATH),
    ]

    for mode, path in modes:
        cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

        data_dict = None

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {mode} data from cache: {cache_path}")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {
                    "seq": loaded["seq"],
                    "loop": loaded["loop"],
                    "bond": loaded["bond"],
                    "dist": loaded["dist"],
                    "ids": loaded["ids"],
                }
                if mode != "test":
                    data_dict["targets"] = loaded["targets"]
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                data_dict = None

        # Process from scratch if needed
        if data_dict is None:
            print(f"Processing {mode} data from {path}...")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Metadata file not found: {path}")

            df = pd.read_parquet(path)
            data_dict = preprocess_data(df, mode=mode)

            # Save to cache
            print(f"Saving {mode} data to cache...")
            save_kwargs = {
                "seq": data_dict["seq"],
                "loop": data_dict["loop"],
                "bond": data_dict["bond"],
                "dist": data_dict["dist"],
                "ids": data_dict["ids"],
            }
            if mode != "test":
                save_kwargs["targets"] = data_dict["targets"]

            np.savez(cache_path, **save_kwargs)

        # Create Dataset
        dataset = RNADataset(data_dict, mode=mode)

        # Create DataLoader
        shuffle = mode == "train"
        drop_last = mode == "train"

        loaders[mode] = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=drop_last,
        )

    return loaders["train"], loaders["val"], loaders["test"]
