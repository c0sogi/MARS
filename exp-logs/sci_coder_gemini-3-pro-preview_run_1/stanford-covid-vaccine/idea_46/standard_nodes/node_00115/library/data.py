import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed(Config.SEED)


class RNADataset(Dataset):
    def __init__(self, sequences, loop_types, structures, targets=None, ids=None):
        """
        PyTorch Dataset for RNA degradation prediction using Discrete Topological Tokens.

        Args:
            sequences (np.ndarray): (N, 107) Integer tokens for nucleotide identity.
            loop_types (np.ndarray): (N, 107) Integer tokens for predicted loop type.
            structures (np.ndarray): (N, 107) Integer tokens for discrete topological distances.
            targets (np.ndarray, optional): (N, 107, 3) Float values for reactivity and degradation.
            ids (np.ndarray, optional): (N,) String IDs for the samples.
        """
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.loop_types = torch.tensor(loop_types, dtype=torch.long)
        self.structures = torch.tensor(structures, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

        self.ids = ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = {
            "sequence": self.sequences[idx],
            "loop_type": self.loop_types[idx],
            "structure_dist": self.structures[idx],
        }

        if self.targets is not None:
            sample["target"] = self.targets[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def parse_structure_to_distance(structure_str, seq_len=107, clip=32):
    """
    Parses dot-bracket structure into discrete topological distance tokens.

    Logic:
    1. Identify pairs (i, j).
    2. Calculate signed distance d = j - i.
    3. Clip d to [-clip, clip].
    4. Offset by +clip to map to [0, 2*clip].
    5. Unpaired bases are assigned index 'clip' (equivalent to d=0).

    Args:
        structure_str (str): Dot-bracket notation string.
        seq_len (int): Length of the sequence.
        clip (int): Maximum distance to encode explicitly.

    Returns:
        np.ndarray: (seq_len,) array of integer tokens.
    """
    # Initialize with index 'clip' (32), which corresponds to distance 0.
    # Since distance 0 is impossible for valid pairs, this effectively acts
    # as the "Unpaired" token.
    distances = np.full(seq_len, clip, dtype=int)

    stack = []
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start_idx = stack.pop()
                # Calculate signed distances
                # Forward: j - i (positive)
                # Backward: i - j (negative)
                dist_forward = i - start_idx
                dist_backward = start_idx - i

                # Clip and Offset
                # Range [-32, 32] -> +32 -> [0, 64]
                distances[start_idx] = np.clip(dist_forward, -clip, clip) + clip
                distances[i] = np.clip(dist_backward, -clip, clip) + clip
            else:
                # Unbalanced closing bracket, treat as unpaired
                pass

    return distances


def process_dataframe(df, is_test=False):
    """
    Converts DataFrame columns into Numpy arrays suitable for the model.
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LENGTH

    # Pre-allocate arrays
    seq_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    loop_arr = np.zeros((num_samples, seq_len), dtype=np.int32)
    dist_arr = np.zeros((num_samples, seq_len), dtype=np.int32)

    # Targets: (N, 107, 3) - Pad with zeros
    target_arr = None
    if not is_test:
        target_arr = np.zeros(
            (num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32
        )

    base_map = Config.BASE_TO_INT
    loop_map = Config.LOOP_TO_INT

    for idx, row in df.iterrows():
        # 1. Sequence Tokenization
        seq_arr[idx] = [base_map.get(c, 0) for c in row["sequence"]]

        # 2. Loop Type Tokenization
        loop_arr[idx] = [loop_map.get(c, 0) for c in row["predicted_loop_type"]]

        # 3. Discrete Distance Tokenization
        dist_arr[idx] = parse_structure_to_distance(
            row["structure"], seq_len=seq_len, clip=Config.DIST_CLIP
        )

        # 4. Targets
        if not is_test:
            # We train on: reactivity, deg_Mg_pH10, deg_Mg_50C
            # Data provided is usually length 68. We fill the beginning of the 107 array.

            # Reactivity
            r = row["reactivity"]
            if r is not None:
                length = min(len(r), seq_len)
                target_arr[idx, :length, 0] = r[:length]

            # deg_Mg_pH10
            d1 = row["deg_Mg_pH10"]
            if d1 is not None:
                length = min(len(d1), seq_len)
                target_arr[idx, :length, 1] = d1[:length]

            # deg_Mg_50C
            d2 = row["deg_Mg_50C"]
            if d2 is not None:
                length = min(len(d2), seq_len)
                target_arr[idx, :length, 2] = d2[:length]

    return seq_arr, loop_arr, dist_arr, target_arr


def load_and_process_data(data_path, cache_name, load_cached_data=True, is_test=False):
    """
    Loads data, processes it, and caches the result to disk.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            seq_arr = data["seq"]
            loop_arr = data["loop"]
            dist_arr = data["dist"]
            ids = data["ids"]

            if is_test:
                return seq_arr, loop_arr, dist_arr, None, ids
            else:
                target_arr = data["target"]
                return seq_arr, loop_arr, dist_arr, target_arr, ids
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing data from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    seq_arr, loop_arr, dist_arr, target_arr = process_dataframe(df, is_test=is_test)
    ids = df["id"].values

    # 3. Save Cache
    save_dict = {"seq": seq_arr, "loop": loop_arr, "dist": dist_arr, "ids": ids}
    if target_arr is not None:
        save_dict["target"] = target_arr

    np.savez(cache_path, **save_dict)
    print(f"Cached processed data to {cache_path}")

    return seq_arr, loop_arr, dist_arr, target_arr, ids


def get_dataloaders(
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Constructs and returns DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Load Data
    # Updated cache names to force reprocessing with new DIST_CLIP
    train_seq, train_loop, train_dist, train_targets, train_ids = load_and_process_data(
        Config.TRAIN_PATH, "train_data_v2", load_cached_data, is_test=False
    )

    val_seq, val_loop, val_dist, val_targets, val_ids = load_and_process_data(
        Config.VAL_PATH, "val_data_v2", load_cached_data, is_test=False
    )

    test_seq, test_loop, test_dist, _, test_ids = load_and_process_data(
        Config.TEST_PATH, "test_data_v2", load_cached_data, is_test=True
    )

    # Debug Subset
    if debug:
        subset = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG MODE: Subsetting data to {subset} samples.")
        train_seq = train_seq[:subset]
        train_loop = train_loop[:subset]
        train_dist = train_dist[:subset]
        train_targets = train_targets[:subset]
        train_ids = train_ids[:subset]

        val_seq = val_seq[:subset]
        val_loop = val_loop[:subset]
        val_dist = val_dist[:subset]
        val_targets = val_targets[:subset]
        val_ids = val_ids[:subset]

        test_seq = test_seq[:subset]
        test_loop = test_loop[:subset]
        test_dist = test_dist[:subset]
        test_ids = test_ids[:subset]

    # Instantiate Datasets
    train_dataset = RNADataset(
        train_seq, train_loop, train_dist, train_targets, train_ids
    )
    val_dataset = RNADataset(val_seq, val_loop, val_dist, val_targets, val_ids)
    test_dataset = RNADataset(test_seq, test_loop, test_dist, None, test_ids)

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
