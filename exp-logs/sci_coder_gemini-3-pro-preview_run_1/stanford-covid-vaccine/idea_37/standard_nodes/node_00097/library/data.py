import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    WORKING_DIR,
    TOKEN_MAP_SEQ,
    TOKEN_MAP_LOOP,
    TARGET_COLS,
    ERROR_COLS,
    SEQ_SCORED,
    process_structure_to_pairs,
)


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    Stores pre-processed tensors for sequences, loops, structure distances,
    targets, and error estimates.
    """

    def __init__(self, seqs, loops, pair_dists, ids, targets=None, errors=None):
        self.seqs = seqs
        self.loops = loops
        self.pair_dists = pair_dists
        self.ids = ids
        self.targets = targets
        self.errors = errors

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors on the fly
        item = {
            "seq": torch.from_numpy(self.seqs[idx]),
            "loop": torch.from_numpy(self.loops[idx]),
            "pair_dist": torch.from_numpy(self.pair_dists[idx]),
        }

        if self.targets is not None:
            item["target"] = torch.from_numpy(self.targets[idx])

        if self.errors is not None:
            item["error"] = torch.from_numpy(self.errors[idx])

        return item


def get_dataset(mode="train", load_cached_data=True):
    """
    Loads the dataset for the specified mode (train, val, test).
    Implements caching using .npz files to avoid pickle and speed up loading.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The initialized dataset object.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(WORKING_DIR, f"{mode}_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {mode} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            # Extract arrays
            seqs = data["seqs"]
            loops = data["loops"]
            pair_dists = data["pair_dists"]
            ids = data["ids"]

            targets = data["targets"] if "targets" in data else None
            errors = data["errors"] if "errors" in data else None

            return RNADataset(seqs, loops, pair_dists, ids, targets, errors)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data...")

    # 2. Process from scratch
    print(f"Processing {mode} data from scratch...")

    # Determine source file
    if mode == "train":
        source_path = TRAIN_METADATA
    elif mode == "val":
        source_path = VAL_METADATA
    else:
        source_path = TEST_METADATA

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Metadata file not found: {source_path}")

    df = pd.read_parquet(source_path)

    # --- Feature Processing ---

    # 1. Sequence Tokenization
    # Map chars to integers using TOKEN_MAP_SEQ
    seqs = np.array(
        [[TOKEN_MAP_SEQ.get(c, 0) for c in seq] for seq in df["sequence"].values],
        dtype=np.int64,
    )

    # 2. Loop Type Tokenization
    # Map chars to integers using TOKEN_MAP_LOOP
    loops = np.array(
        [
            [TOKEN_MAP_LOOP.get(c, 0) for c in loop]
            for loop in df["predicted_loop_type"].values
        ],
        dtype=np.int64,
    )

    # 3. Structure Pairing Distance
    # Compute signed distances using the provided utility
    pair_dists = np.array(
        [process_structure_to_pairs(struct) for struct in df["structure"].values],
        dtype=np.float32,
    )

    ids = df["id"].values
    targets = None
    errors = None

    # --- Target Processing (Train/Val only) ---
    if mode in ["train", "val"]:
        # Stack targets into (N, 68, 3)
        targets = (
            np.vstack(df[TARGET_COLS].values.tolist())
            .reshape(-1, SEQ_SCORED, len(TARGET_COLS))
            .astype(np.float32)
        )

    # 3. Save to Cache (using np.savez_compressed)
    save_dict = {"seqs": seqs, "loops": loops, "pair_dists": pair_dists, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    print(f"Saving {mode} data to cache: {cache_path}")
    np.savez_compressed(cache_path, **save_dict)

    return RNADataset(seqs, loops, pair_dists, ids, targets, errors)
