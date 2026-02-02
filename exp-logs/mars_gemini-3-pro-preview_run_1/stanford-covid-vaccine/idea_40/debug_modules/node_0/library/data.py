import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =========================================================================
# Token Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_distance(structure_str, seq_len):
    """
    Parses a dot-bracket structure string to compute signed pairing distances.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").
        seq_len (int): Length of the sequence.

    Returns:
        np.array: Array of signed distances.
                  If i pairs with j, val = j - i.
                  If unpaired, val = 0.
    """
    stack = []
    indices = np.zeros(seq_len, dtype=np.int32)

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Signed distance
                indices[i] = j - i
                indices[j] = i - j
            else:
                # Unbalanced closing bracket (should not happen in valid data)
                pass

    return indices


def encode_sequence(seq_str, mapping):
    return np.array([mapping.get(c, 0) for c in seq_str], dtype=np.int32)


def process_dataframe(df, mode="train"):
    """
    Converts a pandas DataFrame into numpy arrays for inputs and targets.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    sequences = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_types = np.zeros((num_samples, seq_len), dtype=np.int32)
    pair_dists = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets are only present in train/val
    has_targets = mode != "test"
    targets = None
    if has_targets:
        # 3 target channels: reactivity, deg_Mg_pH10, deg_Mg_50C
        targets = np.zeros((num_samples, seq_len, 3), dtype=np.float32)

    # Iterate and process
    for idx, row in df.iterrows():
        # 0 is the start index for numpy arrays if df index is not reset
        # We use enumerate on the dataframe values or reset index before
        # Here we assume df is passed from process_data which loads fresh
        # so we use integer index i
        pass

    # Efficient vectorization is hard with strings, using loop
    # Re-initializing to be safe with enumeration

    # We will collect lists then stack to avoid index confusion
    seq_list = []
    loop_list = []
    dist_list = []
    target_list = []
    ids_list = []

    for _, row in df.iterrows():
        # Inputs
        s_arr = encode_sequence(row["sequence"], SEQ_MAP)
        l_arr = encode_sequence(row["predicted_loop_type"], LOOP_MAP)
        d_arr = parse_structure_distance(row["structure"], seq_len)

        seq_list.append(s_arr)
        loop_list.append(l_arr)
        dist_list.append(d_arr)
        ids_list.append(row["id"])

        if has_targets:
            # Extract targets (length 68)
            t_react = np.array(row["reactivity"], dtype=np.float32)
            t_mg_ph10 = np.array(row["deg_Mg_pH10"], dtype=np.float32)
            t_mg_50c = np.array(row["deg_Mg_50C"], dtype=np.float32)

            # Stack channels
            # Shape: (68, 3)
            t_sample = np.stack([t_react, t_mg_ph10, t_mg_50c], axis=1)

            # Pad to 107
            pad_len = seq_len - len(t_sample)
            if pad_len > 0:
                t_padded = np.pad(t_sample, ((0, pad_len), (0, 0)), "constant")
            else:
                t_padded = t_sample

            target_list.append(t_padded)

    sequences = np.stack(seq_list)
    loop_types = np.stack(loop_list)
    pair_dists = np.stack(dist_list)
    ids = np.array(ids_list)

    if has_targets:
        targets = np.stack(target_list)
        return sequences, loop_types, pair_dists, targets, ids
    else:
        return sequences, loop_types, pair_dists, ids


def get_processed_data(mode, load_cached_data=True):
    """
    Loads data from cache or processes from source Parquet files.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: Numpy arrays (sequences, loop_types, pair_dists, targets, ids)
               Note: targets is None for 'test' mode.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"cached_{mode}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {mode} data from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            sequences = data["sequences"]
            loop_types = data["loop_types"]
            pair_dists = data["pair_dists"]
            ids = data["ids"]

            if "targets" in data:
                targets = data["targets"]
                return sequences, loop_types, pair_dists, targets, ids
            else:
                return sequences, loop_types, pair_dists, None, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from source...")

    if mode == "train":
        file_path = Config.TRAIN_METADATA
    elif mode == "val":
        file_path = Config.VAL_METADATA
    elif mode == "test":
        file_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df = pd.read_parquet(file_path)

    result = process_dataframe(df, mode)

    # Unpack result to save
    if mode == "test":
        sequences, loop_types, pair_dists, ids = result
        targets = None
        save_dict = {
            "sequences": sequences,
            "loop_types": loop_types,
            "pair_dists": pair_dists,
            "ids": ids,
        }
    else:
        sequences, loop_types, pair_dists, targets, ids = result
        save_dict = {
            "sequences": sequences,
            "loop_types": loop_types,
            "pair_dists": pair_dists,
            "targets": targets,
            "ids": ids,
        }

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(cache_path, **save_dict)
    print(f"Saved {mode} data to cache: {cache_path}")

    return sequences, loop_types, pair_dists, targets, ids


class RNADataset(Dataset):
    def __init__(self, sequences, loop_types, pair_dists, targets=None, ids=None):
        self.sequences = sequences
        self.loop_types = loop_types
        self.pair_dists = pair_dists
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert to tensors
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_types[idx], dtype=torch.long)
        dist = torch.tensor(self.pair_dists[idx], dtype=torch.long)

        item = {"sequence": seq, "loop_type": loop, "pair_dist": dist}

        if self.ids is not None:
            item["id"] = self.ids[idx]

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["target"] = target

        return item


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed data
    train_data = get_processed_data("train", load_cached_data)
    val_data = get_processed_data("val", load_cached_data)
    test_data = get_processed_data("test", load_cached_data)

    # Unpack
    train_seq, train_loop, train_dist, train_tgt, train_ids = train_data
    val_seq, val_loop, val_dist, val_tgt, val_ids = val_data
    test_seq, test_loop, test_dist, _, test_ids = test_data

    # Debug subsetting
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG mode: Subsetting to {subset_size} samples.")

        train_seq = train_seq[:subset_size]
        train_loop = train_loop[:subset_size]
        train_dist = train_dist[:subset_size]
        train_tgt = train_tgt[:subset_size]
        train_ids = train_ids[:subset_size]

        val_seq = val_seq[:subset_size]
        val_loop = val_loop[:subset_size]
        val_dist = val_dist[:subset_size]
        val_tgt = val_tgt[:subset_size]
        val_ids = val_ids[:subset_size]

        test_seq = test_seq[:subset_size]
        test_loop = test_loop[:subset_size]
        test_dist = test_dist[:subset_size]
        test_ids = test_ids[:subset_size]

    # Create Datasets
    train_dataset = RNADataset(train_seq, train_loop, train_dist, train_tgt, train_ids)
    val_dataset = RNADataset(val_seq, val_loop, val_dist, val_tgt, val_ids)
    test_dataset = RNADataset(test_seq, test_loop, test_dist, None, test_ids)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
