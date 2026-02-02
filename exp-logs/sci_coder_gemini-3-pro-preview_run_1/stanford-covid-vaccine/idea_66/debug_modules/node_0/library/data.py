import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Token Mappings
# =========================================================================
TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Serves sequences, structural features, and targets.
    """

    def __init__(self, sequences, loops, structures, targets=None, ids=None):
        self.sequences = sequences
        self.loops = loops
        self.structures = structures
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        # Structure is a float for sinusoidal encoding (signed distance)
        struct = torch.tensor(self.structures[idx], dtype=torch.float32)

        item = {"seq": seq, "loop": loop, "struct": struct}

        # Targets (Training/Validation only)
        if self.targets is not None:
            # Shape: (Seq_Len, 3)
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["target"] = target

        # IDs (Test/Validation tracking)
        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def parse_structure(structure_str):
    """
    Parses a dot-bracket structure string into a signed distance vector.
    For a pair (i, j), index i has value (j-i), index j has value (i-j).
    Unpaired bases have value 0.
    """
    n = len(structure_str)
    dists = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Signed distance: Target - Source
                dists[i] = float(
                    j - i
                )  # At closing bracket, point to opening (negative)
                dists[j] = float(
                    i - j
                )  # At opening bracket, point to closing (positive)

    return dists


def process_data(df, mode="train"):
    """
    Processes a dataframe into numpy arrays for the dataset.
    """
    n_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    sequences = np.zeros((n_samples, seq_len), dtype=np.int32)
    loops = np.zeros((n_samples, seq_len), dtype=np.int32)
    structures = np.zeros((n_samples, seq_len), dtype=np.float32)

    # Process inputs row by row
    for idx, (_, row) in enumerate(df.iterrows()):
        # Sequence Tokenization
        sequences[idx] = [TOKEN_MAP.get(c, 0) for c in row["sequence"]]

        # Loop Type Tokenization
        loops[idx] = [LOOP_MAP.get(c, 0) for c in row["predicted_loop_type"]]

        # Structure Parsing
        structures[idx] = parse_structure(row["structure"])

    # Process targets (only for train/val)
    targets = None
    if mode != "test":
        # Shape: (N, Seq_Len, 3)
        targets = np.zeros((n_samples, seq_len, 3), dtype=np.float32)
        target_cols = Config.TARGET_COLS

        for t_idx, col in enumerate(target_cols):
            # df[col] contains lists of length 68
            values = df[col].values
            for idx, val_array in enumerate(values):
                # Copy the available 68 values
                # The rest (68-107) remain 0.0 (masked out by loss function later)
                length = len(val_array)
                targets[idx, :length, t_idx] = val_array

    ids = df["id"].values

    return {
        "sequences": sequences,
        "loops": loops,
        "structures": structures,
        "targets": targets,
        "ids": ids,
    }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching of processed data to speed up subsequent runs.
    """
    seed_everything(Config.SEED)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    loaders = {}
    modes = ["train", "val", "test"]

    for mode in modes:
        cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")
        data_dict = None

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {mode} data from cache: {cache_path}")
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                data_dict = {
                    "sequences": loaded["sequences"],
                    "loops": loaded["loops"],
                    "structures": loaded["structures"],
                    "ids": loaded["ids"],
                }
                if mode != "test":
                    data_dict["targets"] = loaded["targets"]
                else:
                    data_dict["targets"] = None
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}")
                data_dict = None

        # 2. Process from metadata if cache miss
        if data_dict is None:
            print(f"Processing {mode} data from metadata...")
            parquet_path = os.path.join(Config.METADATA_DIR, f"{mode}.parquet")

            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

            df = pd.read_parquet(parquet_path)

            # Debugging subset
            if Config.DEBUG:
                print(f"Debug mode: sampling {Config.DEBUG_SAMPLES} rows.")
                df = df.head(Config.DEBUG_SAMPLES)

            data_dict = process_data(df, mode=mode)

            # Save to cache
            print(f"Saving {mode} data to cache: {cache_path}")
            save_dict = {
                "sequences": data_dict["sequences"],
                "loops": data_dict["loops"],
                "structures": data_dict["structures"],
                "ids": data_dict["ids"],
            }
            if mode != "test":
                save_dict["targets"] = data_dict["targets"]

            np.savez_compressed(cache_path, **save_dict)

        # 3. Create Dataset
        dataset = RNADataset(
            sequences=data_dict["sequences"],
            loops=data_dict["loops"],
            structures=data_dict["structures"],
            targets=data_dict["targets"],
            ids=data_dict["ids"],
        )

        # 4. Create DataLoader
        shuffle = mode == "train"
        drop_last = mode == "train"  # Stability for BatchNorm/Training

        loaders[mode] = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=drop_last,
        )

    return loaders["train"], loaders["val"], loaders["test"]
