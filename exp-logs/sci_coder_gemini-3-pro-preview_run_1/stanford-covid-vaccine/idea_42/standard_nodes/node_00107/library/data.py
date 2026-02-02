import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, sequences, loop_types, pair_offsets, targets=None, ids=None):
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_offsets = pair_offsets
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert inputs to Long tensors for embedding lookups or positional encodings
        item = {
            "sequence": torch.tensor(self.sequences[idx], dtype=torch.long),
            "loop_type": torch.tensor(self.loop_types[idx], dtype=torch.long),
            "pair_offset": torch.tensor(
                self.pair_offsets[idx], dtype=torch.float
            ),  # Float for sinusoidal encoding
        }

        # Include targets if available (Training/Validation)
        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        # Include ID for submission/tracking
        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def parse_structure(structure):
    """
    Parses a dot-bracket structure string to generate a pair_offset vector.
    Value at i is j-i if paired with j, else 0.
    """
    n = len(structure)
    offsets = np.zeros(n, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is the index of '(', i is the index of ')'
                # For open bracket at j: paired with i -> value = i - j
                # For close bracket at i: paired with j -> value = j - i
                offsets[j] = i - j
                offsets[i] = j - i
    return offsets


def process_dataframe(df, is_test=False):
    """
    Converts DataFrame columns into numpy arrays suitable for the Dataset.
    """
    # 1. Process Sequences
    seq_map = Config.token2id
    sequences = []
    for seq in df["sequence"]:
        sequences.append([seq_map.get(c, 0) for c in seq])
    sequences = np.array(sequences, dtype=np.int16)

    # 2. Process Loop Types
    loop_map = Config.loop2id
    loop_types = []
    for lt in df["predicted_loop_type"]:
        loop_types.append([loop_map.get(c, 0) for c in lt])
    loop_types = np.array(loop_types, dtype=np.int16)

    # 3. Process Structure (Pair Offsets)
    pair_offsets = []
    for struct in df["structure"]:
        pair_offsets.append(parse_structure(struct))
    pair_offsets = np.array(pair_offsets, dtype=np.int16)

    # 4. Process Targets
    targets = None
    if not is_test:
        n_samples = len(df)
        seq_len = Config.seq_len
        target_cols = Config.target_cols
        n_targets = len(target_cols)

        # Initialize with zeros (padding for positions > 68)
        targets = np.zeros((n_samples, seq_len, n_targets), dtype=np.float32)

        for i, col in enumerate(target_cols):
            # Each row in df[col] is a list/array of length 68
            # We stack them into the target tensor
            col_values = df[col].values
            for idx, val_list in enumerate(col_values):
                length = len(val_list)
                # Fill the available ground truth
                targets[idx, :length, i] = val_list

    # 5. IDs
    ids = df["id"].values if "id" in df.columns else None

    return sequences, loop_types, pair_offsets, targets, ids


def get_data(mode, load_cached_data=True, max_samples=None):
    """
    Retrieves data for the specified mode (train/val/test).
    Uses caching to speed up subsequent loads.
    """
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cached_{mode}.npz")

    # Try to load from cache
    loaded = False
    if load_cached_data and os.path.exists(cache_file):
        # print(f"Loading {mode} data from cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        if "sequences" in data:
            sequences = data["sequences"]
            loop_types = data["loop_types"]
            pair_offsets = data["pair_offsets"]
            ids = data["ids"]
            targets = data["targets"] if "targets" in data else None
            loaded = True

    if not loaded:
        # print(f"Processing {mode} data from metadata...")
        if mode == "train":
            path = Config.train_metadata
        elif mode == "val":
            path = Config.val_metadata
        elif mode == "test":
            path = Config.test_metadata
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata file not found: {path}")

        df = pd.read_parquet(path)
        sequences, loop_types, pair_offsets, targets, ids = process_dataframe(
            df, is_test=(mode == "test")
        )

        # Save to cache
        save_dict = {
            "sequences": sequences,
            "loop_types": loop_types,
            "pair_offsets": pair_offsets,
            "ids": ids,
        }
        if targets is not None:
            save_dict["targets"] = targets

        np.savez_compressed(cache_file, **save_dict)
        # print(f"Saved {mode} data to cache: {cache_file}")

    # Handle max_samples for debugging
    if max_samples is not None:
        sequences = sequences[:max_samples]
        loop_types = loop_types[:max_samples]
        pair_offsets = pair_offsets[:max_samples]
        if ids is not None:
            ids = ids[:max_samples]
        if targets is not None:
            targets = targets[:max_samples]

    return sequences, loop_types, pair_offsets, targets, ids


def get_loader(
    mode,
    batch_size=None,
    shuffle=None,
    load_cached_data=True,
    num_workers=None,
    max_samples=None,
):
    """
    Creates a DataLoader for the specified mode.
    """
    if batch_size is None:
        batch_size = Config.batch_size
    if num_workers is None:
        num_workers = Config.num_workers

    # Default shuffle logic
    if shuffle is None:
        shuffle = mode == "train"

    sequences, loop_types, pair_offsets, targets, ids = get_data(
        mode, load_cached_data, max_samples
    )

    dataset = RNADataset(sequences, loop_types, pair_offsets, targets, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
