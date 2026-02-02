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


# =========================================================================
# Helper Functions
# =========================================================================
def get_structure_distance(structure_str):
    """
    Parses a dot-bracket structure string and returns an array of signed distances.

    Logic:
    - If nucleotide i is paired with j:
      - distance at i = j - i
      - distance at j = i - j
    - If nucleotide i is unpaired:
      - distance at i = 0

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of signed integer distances.
    """
    n = len(structure_str)
    distances = np.zeros(n, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                open_idx = stack.pop()
                # open_idx is paired with i (closing)
                # distance for open_idx (upstream) is positive: i - open_idx
                distances[open_idx] = i - open_idx
                # distance for i (downstream) is negative: open_idx - i
                distances[i] = open_idx - i

    return distances


def preprocess_data(df, mode="train"):
    """
    Converts dataframe columns to numpy arrays suitable for model input.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        tuple: (sequences, loops, pair_dists, targets, ids)
    """
    # 1. Process Sequences
    sequences = []
    for seq in df["sequence"]:
        tokens = [SEQ_MAP.get(c, 0) for c in seq]
        sequences.append(tokens)
    sequences = np.array(sequences, dtype=np.int32)

    # 2. Process Loop Types
    loops = []
    for lp in df["predicted_loop_type"]:
        tokens = [LOOP_MAP.get(c, 0) for c in lp]
        loops.append(tokens)
    loops = np.array(loops, dtype=np.int32)

    # 3. Process Structure (Pair Distances)
    pair_dists = []
    for struct in df["structure"]:
        dists = get_structure_distance(struct)
        pair_dists.append(dists)
    pair_dists = np.array(pair_dists, dtype=np.int32)

    # 4. Process Targets
    # Targets are only relevant for train/val and strictly for the first 68 positions
    if mode in ["train", "val"]:
        targets_list = []
        for _, row in df.iterrows():
            row_targets = []
            for col in Config.TARGET_COLS:
                # Retrieve list/array from dataframe
                val_array = np.array(row[col], dtype=np.float32)

                # Ensure fixed length (Config.PRED_LENGTH = 68)
                if len(val_array) >= Config.PRED_LENGTH:
                    val_array = val_array[: Config.PRED_LENGTH]
                else:
                    # Pad with zeros if shorter (unlikely given dataset specs)
                    pad = np.zeros(
                        Config.PRED_LENGTH - len(val_array), dtype=np.float32
                    )
                    val_array = np.concatenate([val_array, pad])

                row_targets.append(val_array)

            # Stack columns: (3, 68) -> Transpose to (68, 3)
            targets_list.append(np.stack(row_targets, axis=1))

        targets = np.array(targets_list, dtype=np.float32)
    else:
        # Test mode: Create dummy targets of shape (N, 68, 3)
        targets = np.zeros(
            (len(df), Config.PRED_LENGTH, Config.NUM_TARGETS), dtype=np.float32
        )

    # 5. IDs
    ids = df["id"].values

    return sequences, loops, pair_dists, targets, ids


def load_or_process_data(parquet_path, cache_name, mode="train", load_cached_data=True):
    """
    Handles data loading with caching mechanism using .npz files.
    """
    cache_path = os.path.join(Config.CACHE_DIR, cache_name)

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["sequences"],
                data["loops"],
                data["pair_dists"],
                data["targets"],
                data["ids"],
            )
        except Exception:
            # If load fails, fall back to processing
            pass

    # Process from scratch
    df = pd.read_parquet(parquet_path)
    sequences, loops, pair_dists, targets, ids = preprocess_data(df, mode=mode)

    # Save to cache
    np.savez(
        cache_path,
        sequences=sequences,
        loops=loops,
        pair_dists=pair_dists,
        targets=targets,
        ids=ids,
    )

    return sequences, loops, pair_dists, targets, ids


# =========================================================================
# Dataset Class
# =========================================================================
class RNADataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npz files.
        """
        self.mode = mode

        # Determine paths based on mode
        if mode == "train":
            path = Config.TRAIN_METADATA_PATH
            cache = "train_data.npz"
        elif mode == "val":
            path = Config.VAL_METADATA_PATH
            cache = "val_data.npz"
        else:
            path = Config.TEST_METADATA_PATH
            cache = "test_data.npz"

        # Load data
        self.sequences, self.loops, self.pair_dists, self.targets, self.ids = (
            load_or_process_data(path, cache, mode, load_cached_data)
        )

        # Handle Debug Mode (Subset data)
        if Config.DEBUG:
            subset = min(len(self.ids), Config.SUBSET_SIZE)
            self.sequences = self.sequences[:subset]
            self.loops = self.loops[:subset]
            self.pair_dists = self.pair_dists[:subset]
            self.targets = self.targets[:subset]
            self.ids = self.ids[:subset]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns:
            seq (torch.LongTensor): (107,) Nucleotide indices
            loop (torch.LongTensor): (107,) Loop type indices
            dist (torch.LongTensor): (107,) Signed pair distances
            target (torch.FloatTensor): (68, 3) Target values
            id (str): Sample ID
        """
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        dist = torch.tensor(self.pair_dists[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return seq, loop, dist, target, self.ids[idx]


# =========================================================================
# DataLoader Factory
# =========================================================================
def get_dataloader(
    mode="train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
):
    """
    Creates a DataLoader for the specified mode.
    """
    dataset = RNADataset(mode=mode, load_cached_data=load_cached_data)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=(mode == "train"),  # Drop last incomplete batch only during training
    )

    return loader
