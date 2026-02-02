import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_PARQUET,
    VAL_PARQUET,
    TEST_PARQUET,
    WORKING_DIR,
    TOKEN2INT,
    LOOP2INT,
    TARGET_COLS,
    BATCH_SIZE,
    NUM_WORKERS,
    SEQ_LEN,
)


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, data_dict, is_test=False):
        self.sequences = data_dict["sequences"]
        self.loop_types = data_dict["loop_types"]
        self.pair_dists = data_dict["pair_dists"]
        self.ids = data_dict["ids"]
        self.is_test = is_test

        if not self.is_test:
            self.targets = data_dict["targets"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        item = {
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.long),
            "loop_type": torch.tensor(self.loop_types[idx], dtype=torch.long),
            "pair_dist": torch.tensor(self.pair_dists[idx], dtype=torch.float32),
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def token_encoder(seq_str, mapping):
    """
    Converts a sequence string into an array of integers based on the provided mapping.
    """
    return np.array([mapping.get(c, 0) for c in seq_str], dtype=np.int64)


def parse_structure(structure_str):
    """
    Parses a dot-bracket structure string into a signed distance array.

    Logic:
    - Unpaired bases ('.') have a distance of 0.
    - Paired bases ('(', ')') have a distance equal to the signed difference in indices (partner_index - current_index).
      - For an opening bracket at i paired with j (where i < j): distance is j - i (Positive).
      - For a closing bracket at j paired with i (where i < j): distance is i - j (Negative).
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
                # j is the index of the opening bracket '('
                # i is the index of the closing bracket ')'

                # Distance for the opening bracket at j (partner is downstream at i)
                dists[j] = float(i - j)

                # Distance for the closing bracket at i (partner is upstream at j)
                dists[i] = float(j - i)

    return dists


def process_dataframe(df, is_test=False):
    """
    Extracts and processes features and targets from a pandas DataFrame.
    """
    # 1. Sequence Encoding
    sequences = np.array([token_encoder(s, TOKEN2INT) for s in df["sequence"].values])

    # 2. Loop Type Encoding
    loop_types = np.array(
        [token_encoder(s, LOOP2INT) for s in df["predicted_loop_type"].values]
    )

    # 3. Structure Distance Parsing
    pair_dists = np.array([parse_structure(s) for s in df["structure"].values])

    ids = df["id"].values

    data = {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "ids": ids,
    }

    # 4. Target Processing (Train/Val only)
    if not is_test:
        target_arrays = []
        for col in TARGET_COLS:
            # df[col] contains lists/arrays of length 68 (PRED_LEN).
            # We need to stack them and pad to SEQ_LEN (107) to match input dimensions.
            col_data = np.vstack(df[col].values)

            current_len = col_data.shape[1]
            pad_len = SEQ_LEN - current_len

            if pad_len > 0:
                padding = np.zeros((len(df), pad_len), dtype=col_data.dtype)
                col_data_padded = np.hstack([col_data, padding])
            else:
                col_data_padded = col_data

            target_arrays.append(col_data_padded)

        # Stack along the last dimension -> Shape: (N, SEQ_LEN, Num_Targets)
        targets = np.stack(target_arrays, axis=2)
        data["targets"] = targets

    return data


def get_dataloaders(load_cached_data=True, debug_sample_size=None):
    """
    Prepares DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from .npz files.
        debug_sample_size (int, optional): If set, limits the dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists for cache
    os.makedirs(WORKING_DIR, exist_ok=True)

    splits = ["train", "val", "test"]
    paths = [TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET]
    loaders = []

    for split, path in zip(splits, paths):
        cache_path = os.path.join(WORKING_DIR, f"{split}_data.npz")
        is_test = split == "test"

        data_dict = None

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                # Reconstruct dictionary from NpzFile
                data_dict = {k: loaded[k] for k in loaded.files}
            except Exception as e:
                print(f"Failed to load cache for {split}: {e}. Reprocessing.")
                data_dict = None

        # 2. Process from scratch if needed
        if data_dict is None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file {path} not found.")

            df = pd.read_parquet(path)
            data_dict = process_dataframe(df, is_test=is_test)

            # Save to cache
            np.savez_compressed(cache_path, **data_dict)

        # 3. Debug Subsampling (applied in-memory, does not overwrite cache)
        if debug_sample_size is not None:
            limit = min(len(data_dict["sequences"]), debug_sample_size)
            for k in data_dict:
                data_dict[k] = data_dict[k][:limit]

        # 4. Create Dataset and DataLoader
        dataset = RNADataset(data_dict, is_test=is_test)

        # Shuffle only for training
        shuffle = split == "train"

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=shuffle,  # Drop incomplete batch only during training
        )
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
