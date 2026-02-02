import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import tokenize_sequence, tokenize_loop_type


def process_structure_to_distance(structure):
    """
    Parses a dot-bracket structure string and calculates the signed distance
    between paired bases.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.array: Array of integers where value at index k is (pair_index - k).
                  Unpaired bases have value 0.
    """
    n = len(structure)
    pair_dist = np.zeros(n, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair is (j, i) where j < i
                # Distance for j is i - j (positive)
                # Distance for i is j - i (negative)
                pair_dist[j] = i - j
                pair_dist[i] = j - i

    return pair_dist


def load_and_process_data(mode, load_cached_data=True):
    """
    Loads data from Parquet, processes it, and caches it to .npz.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{mode}_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached {mode} data from {cache_file}...")
        loaded = np.load(cache_file, allow_pickle=True)
        # Convert NpzFile to dict
        data_dict = {k: loaded[k] for k in loaded.files}
        return data_dict

    print(f"Processing {mode} data from scratch...")

    # Load source parquet
    if mode == "train":
        df = pd.read_parquet(Config.TRAIN_PATH)
    elif mode == "val":
        df = pd.read_parquet(Config.VAL_PATH)
    elif mode == "test":
        df = pd.read_parquet(Config.TEST_PATH)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Initialize containers
    sequences = []
    loop_types = []
    pair_dists = []

    # Targets for train/val
    targets_list = []

    # IDs for test
    ids_list = []

    for _, row in df.iterrows():
        # 1. Sequence
        seq_arr = tokenize_sequence(row["sequence"])
        sequences.append(seq_arr)

        # 2. Loop Type
        loop_arr = tokenize_loop_type(row["predicted_loop_type"])
        loop_types.append(loop_arr)

        # 3. Structure / Pair Distance
        dist_arr = process_structure_to_distance(row["structure"])
        pair_dists.append(dist_arr)

        if mode in ["train", "val"]:
            # 4. Targets
            # Extract specific target columns defined in Config
            target_arrays = []
            for col in Config.TARGET_COLS:
                # Metadata columns are lists/arrays of floats
                target_arrays.append(np.array(row[col], dtype=np.float32))

            # Stack to shape (seq_scored, num_targets) -> (68, 3)
            target_matrix = np.stack(target_arrays, axis=1)
            targets_list.append(target_matrix)
        else:
            ids_list.append(row["id"])

    # Convert lists to numpy arrays
    data_dict = {
        "sequences": np.array(sequences),
        "loop_types": np.array(loop_types),
        "pair_dists": np.array(pair_dists),
    }

    if mode in ["train", "val"]:
        data_dict["targets"] = np.array(targets_list)
    else:
        data_dict["ids"] = np.array(ids_list)

    # Save to cache
    print(f"Saving {mode} data to {cache_file}...")
    np.savez_compressed(cache_file, **data_dict)

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.sequences = data_dict["sequences"]
        self.loop_types = data_dict["loop_types"]
        self.pair_dists = data_dict["pair_dists"]

        if mode in ["train", "val"]:
            self.targets = data_dict["targets"]
        else:
            self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Inputs
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        loop = torch.tensor(self.loop_types[idx], dtype=torch.long)
        # Distance is kept as float for potential sinusoidal encoding in the model
        dist = torch.tensor(self.pair_dists[idx], dtype=torch.float32)

        if self.mode in ["train", "val"]:
            # Targets: The raw data is (68, 3).
            # We need to return a tensor of shape (107, 3) to match the sequence length.
            # The loss function will handle masking the unscored positions.
            target_data = self.targets[idx]  # Shape (68, 3)

            target_tensor = torch.zeros(
                (Config.SEQ_LEN, Config.NUM_TARGETS), dtype=torch.float32
            )

            # Fill the scored positions
            seq_scored = min(target_data.shape[0], Config.SEQ_LEN)
            target_tensor[:seq_scored, :] = torch.tensor(
                target_data[:seq_scored], dtype=torch.float32
            )

            return seq, loop, dist, target_tensor
        else:
            # Test mode: Return ID for submission mapping
            sample_id = str(self.ids[idx])
            return seq, loop, dist, sample_id


def get_dataloader(
    mode,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates a DataLoader for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    data_dict = load_and_process_data(mode, load_cached_data=load_cached_data)
    dataset = RNADataset(data_dict, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
