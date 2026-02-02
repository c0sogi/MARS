import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import parse_structure


class RNADataset(Dataset):
    def __init__(self, seqs, loops, pair_indices, ids, targets=None):
        """
        PyTorch Dataset for RNA degradation prediction.

        Args:
            seqs (np.ndarray): (N, L) Integer-encoded sequences.
            loops (np.ndarray): (N, L) Integer-encoded predicted loops.
            pair_indices (np.ndarray): (N, L) Indices of paired bases (-1 if unpaired).
            ids (np.ndarray): (N,) Sample IDs.
            targets (np.ndarray, optional): (N, 68, 3) Target values.
        """
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.loops = torch.tensor(loops, dtype=torch.long)
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        self.ids = ids
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        item = {
            "seq": self.seqs[idx],
            "loop": self.loops[idx],
            "pair_index": self.pair_indices[idx],
            "id": str(self.ids[idx]),
        }

        if self.targets is not None:
            item["targets"] = self.targets[idx]

        return item


def process_dataframe(df, mode):
    """
    Process a pandas DataFrame into numpy arrays for the dataset.
    """
    # Mappings based on Config
    # Config.VOCAB_SIZE = 4 -> A, G, U, C
    seq_map = {c: i for i, c in enumerate("AGUC")}
    # Config.LOOP_VOCAB_SIZE = 7 -> S, M, I, B, H, E, X
    loop_map = {c: i for i, c in enumerate("SMIBHEX")}

    N = len(df)
    L = Config.SEQ_LENGTH

    # Pre-allocate arrays
    seqs = np.zeros((N, L), dtype=np.int8)
    loops = np.zeros((N, L), dtype=np.int8)
    pair_indices = np.full((N, L), -1, dtype=np.int16)
    ids = df["id"].values.astype(str)

    # Process sequences and structures
    df_seqs = df["sequence"].values
    df_loops = df["predicted_loop_type"].values
    df_structs = df["structure"].values

    for i in range(N):
        # Encode Sequence
        seqs[i] = [seq_map.get(c, 0) for c in df_seqs[i]]

        # Encode Loop
        loops[i] = [loop_map.get(c, 0) for c in df_loops[i]]

        # Parse Structure for Pair Indices
        # parse_structure returns (pair_index, distances)
        p_idx, _ = parse_structure(df_structs[i])
        pair_indices[i] = p_idx

    # Process Targets (only for train/val)
    targets = None
    if mode in ["train", "val"]:
        # We only train on the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # These columns contain lists of floats in the parquet file
        t1 = np.vstack(df["reactivity"].values)
        t2 = np.vstack(df["deg_Mg_pH10"].values)
        t3 = np.vstack(df["deg_Mg_50C"].values)

        # Stack to shape (N, 68, 3)
        targets = np.stack([t1, t2, t3], axis=2).astype(np.float32)

    return seqs, loops, pair_indices, ids, targets


def load_and_process_data(load_cached_data=True):
    """
    Loads data from cache or processes it from scratch using metadata files.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    splits = ["train", "val", "test"]
    # Define cache file paths
    cache_files = {
        split: os.path.join(Config.WORKING_DIR, f"{split}_data.npz") for split in splits
    }

    datasets = {}

    # Check if all cache files exist
    all_cached = all(os.path.exists(f) for f in cache_files.values())

    if load_cached_data and all_cached:
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        for split in splits:
            # allow_pickle=True is required to load the object array of IDs (strings)
            data = np.load(cache_files[split], allow_pickle=True)

            datasets[split] = RNADataset(
                seqs=data["seqs"],
                loops=data["loops"],
                pair_indices=data["pair_indices"],
                ids=data["ids"],
                targets=data["targets"] if "targets" in data else None,
            )
    else:
        print("Processing data from scratch (Parquet metadata)...")

        # Load DataFrames
        df_train = pd.read_parquet(os.path.join(Config.METADATA_DIR, "train.parquet"))
        df_val = pd.read_parquet(os.path.join(Config.METADATA_DIR, "val.parquet"))
        df_test = pd.read_parquet(os.path.join(Config.METADATA_DIR, "test.parquet"))

        dfs = {"train": df_train, "val": df_val, "test": df_test}

        for split, df in dfs.items():
            print(f"Processing {split} split...")
            seqs, loops, pair_indices, ids, targets = process_dataframe(df, split)

            # Save to cache
            save_dict = {
                "seqs": seqs,
                "loops": loops,
                "pair_indices": pair_indices,
                "ids": ids,
            }
            if targets is not None:
                save_dict["targets"] = targets

            np.savez(cache_files[split], **save_dict)

            # Create Dataset
            datasets[split] = RNADataset(seqs, loops, pair_indices, ids, targets)

    return datasets
