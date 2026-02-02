import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def parse_structure_distances(structure):
    """
    Parses a dot-bracket structure string and calculates the signed distance
    to the paired base for each nucleotide.

    Args:
        structure (str): Dot-bracket notation string (e.g., '..((..))..').

    Returns:
        np.ndarray: Array of shape (L,) containing signed distances.
                    Unpaired bases are 0.
                    If i pairs with j:
                        val at i = j - i
                        val at j = i - j
    """
    L = len(structure)
    dists = np.zeros(L, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is index of opening '(', i is index of closing ')'
                # j < i
                # For position j: pair is i, dist = i - j (positive)
                # For position i: pair is j, dist = j - i (negative)
                dists[j] = float(i - j)
                dists[i] = float(j - i)

    return dists


def process_data(df, mode="train"):
    """
    Processes the dataframe into numpy arrays for the dataset.

    Args:
        df (pd.DataFrame): Input dataframe loaded from Parquet.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN
    pred_len = Config.PRED_LEN

    # Initialize arrays
    sequences = np.zeros((num_samples, seq_len), dtype=np.int64)
    loop_types = np.zeros((num_samples, seq_len), dtype=np.int64)
    pair_dists = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets and masks
    # Shape: (N, L, 3) for targets, (N, L) for mask
    targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    masks = np.zeros((num_samples, seq_len), dtype=np.bool_)
    ids = []

    # Pre-fetch maps for speed
    nuc_map = Config.NUCLEOTIDE_MAP
    loop_map = Config.LOOP_TYPE_MAP

    # Iterate and process
    for idx, row in df.iterrows():
        # Use implicit integer index for arrays if index is not reset
        array_idx = (
            idx
            if isinstance(idx, int) and idx < num_samples
            else list(df.index).index(idx)
        )

        # 1. Sequence
        seq_str = row["sequence"]
        sequences[array_idx] = [nuc_map.get(c, 0) for c in seq_str]

        # 2. Loop Type
        loop_str = row["predicted_loop_type"]
        loop_types[array_idx] = [loop_map.get(c, 0) for c in loop_str]

        # 3. Structure / Pair Distance
        struct_str = row["structure"]
        pair_dists[array_idx] = parse_structure_distances(struct_str)

        # 4. Targets (only for train/val)
        if mode in ["train", "val"]:
            # Extract scored targets
            # Each column in the parquet is a list/array of length 68
            row_targets = []
            for t_col in Config.SCORED_TARGETS:
                val = row[t_col]
                # Ensure it's a list or array
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                row_targets.append(val)

            # Stack to (3, 68) -> Transpose to (68, 3)
            # Note: row_targets is list of lists
            t_matrix = np.array(row_targets, dtype=np.float32).T

            # Fill the first 68 positions
            targets[array_idx, :pred_len, :] = t_matrix
            masks[array_idx, :pred_len] = True

        # Store ID
        ids.append(row["id"])

    return {
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "targets": targets,
        "masks": masks,
        "ids": np.array(ids),
    }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays from process_data.
        """
        self.sequences = torch.from_numpy(data_dict["sequences"]).long()
        self.loop_types = torch.from_numpy(data_dict["loop_types"]).long()
        self.pair_dists = torch.from_numpy(data_dict["pair_dists"]).float()
        self.targets = torch.from_numpy(data_dict["targets"]).float()
        self.masks = torch.from_numpy(data_dict["masks"]).bool()
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "loop_type": self.loop_types[idx],
            "pair_dist": self.pair_dists[idx],
            "targets": self.targets[idx],
            "mask": self.masks[idx],
            "id": self.ids[idx],
        }


def get_dataset(mode, load_cached_data=True):
    """
    Loads data, processes it (with caching), and returns an RNADataset.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        RNADataset: The ready-to-use dataset.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data_dict = {
                "sequences": loaded["sequences"],
                "loop_types": loaded["loop_types"],
                "pair_dists": loaded["pair_dists"],
                "targets": loaded["targets"],
                "masks": loaded["masks"],
                "ids": loaded["ids"],
            }
            return RNADataset(data_dict)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from source...")

    if mode == "train":
        source_path = Config.TRAIN_FILE
    elif mode == "val":
        source_path = Config.VAL_FILE
    elif mode == "test":
        source_path = Config.TEST_FILE
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Load Parquet
    df = pd.read_parquet(source_path)

    # Process
    data_dict = process_data(df, mode=mode)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sequences=data_dict["sequences"],
        loop_types=data_dict["loop_types"],
        pair_dists=data_dict["pair_dists"],
        targets=data_dict["targets"],
        masks=data_dict["masks"],
        ids=data_dict["ids"],
    )
    print(f"Saved {mode} data to cache: {cache_path}")

    return RNADataset(data_dict)
