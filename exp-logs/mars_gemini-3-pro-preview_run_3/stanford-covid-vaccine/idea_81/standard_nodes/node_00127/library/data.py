import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Define mappings for One-Hot Encoding
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_MAP = {".": 0, "(": 1, ")": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_adj(structure):
    """
    Parses a dot-bracket structure string to generate an adjacency index array.

    Args:
        structure (str): Dot-bracket structure string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (L,) where arr[i] is the index of the base paired with i.
                    If i is unpaired, arr[i] = -1 (sentinel value).
    """
    length = len(structure)
    adj = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                adj[i] = j
                adj[j] = i

    return adj


def process_data(parquet_path, split_name, load_cached_data=True):
    """
    Loads data from Parquet, processes features and targets, and handles caching.

    Args:
        parquet_path (str): Path to the parquet metadata file.
        split_name (str): Name of the split ('train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for inputs and targets.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{split_name}_data.npz")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split_name} data from cache: {cache_path}")
            loaded = np.load(cache_path)
            data_dict = {
                "ids": loaded["ids"],
                "inputs": loaded["inputs"],
                "adjacency": loaded["adjacency"],
            }
            if "targets" in loaded:
                data_dict["targets"] = loaded["targets"]
            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing {split_name} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    ids = []
    inputs_list = []
    adj_list = []
    targets_list = []

    has_targets = "reactivity" in df.columns

    for idx, row in df.iterrows():
        # ID
        ids.append(row["id"])

        # Sequence Encoding (107, 4)
        seq_ints = [SEQ_MAP.get(c, 0) for c in row["sequence"]]
        seq_oh = np.eye(4)[seq_ints]

        # Structure Encoding (107, 3)
        struct_ints = [STRUCT_MAP.get(c, 0) for c in row["structure"]]
        struct_oh = np.eye(3)[struct_ints]

        # Loop Type Encoding (107, 7)
        loop_ints = [LOOP_MAP.get(c, 0) for c in row["predicted_loop_type"]]
        loop_oh = np.eye(7)[loop_ints]

        # Concatenate Input Features (107, 14)
        # Order: Sequence (4), Structure (3), Loop (7)
        feat = np.concatenate([seq_oh, struct_oh, loop_oh], axis=1)
        inputs_list.append(feat)

        # Adjacency Indices
        adj = get_structure_adj(row["structure"])
        adj_list.append(adj)

        # Targets (68, 5)
        if has_targets:
            # Targets are lists of floats
            t_react = np.array(row["reactivity"], dtype=np.float32)
            t_mg_ph10 = np.array(row["deg_Mg_pH10"], dtype=np.float32)
            t_ph10 = np.array(row["deg_pH10"], dtype=np.float32)
            t_mg_50c = np.array(row["deg_Mg_50C"], dtype=np.float32)
            t_50c = np.array(row["deg_50C"], dtype=np.float32)

            # Stack columns: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            # Shape: (68, 5)
            t_stack = np.stack([t_react, t_mg_ph10, t_ph10, t_mg_50c, t_50c], axis=1)
            targets_list.append(t_stack)

    # Convert lists to numpy arrays
    ids_arr = np.array(ids)
    inputs_arr = np.array(inputs_list, dtype=np.float32)  # (N, 107, 14)
    adj_arr = np.array(adj_list, dtype=np.int32)  # (N, 107)

    save_dict = {"ids": ids_arr, "inputs": inputs_arr, "adjacency": adj_arr}

    data_dict = {"ids": ids_arr, "inputs": inputs_arr, "adjacency": adj_arr}

    if has_targets:
        targets_arr = np.array(targets_list, dtype=np.float32)  # (N, 68, 5)
        save_dict["targets"] = targets_arr
        data_dict["targets"] = targets_arr

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved processed data to {cache_path}")

    return data_dict


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays.
            mode (str): 'train', 'val', or 'test'.
        """
        self.ids = data_dict["ids"]
        self.inputs = data_dict["inputs"]
        self.adjacency = data_dict["adjacency"]
        self.mode = mode

        if mode != "test":
            self.targets = data_dict["targets"]
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Inputs: (107, 14)
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)

        # Adjacency: (107,)
        adj = torch.tensor(self.adjacency[idx], dtype=torch.long)

        # Create a mask for paired bases (1 if paired, 0 if unpaired/-1)
        # This helps the model handle the sentinel value -1
        pair_mask = (adj != -1).float()

        sample = {
            "input": x,
            "adjacency": adj,
            "pair_mask": pair_mask,
            "id": self.ids[idx],
        }

        if self.mode != "test":
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            sample["target"] = y

        return sample


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        debug (bool): If True, subsamples the data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # Process Data
    train_data = process_data(Config.TRAIN_FILE, "train", load_cached_data)
    val_data = process_data(Config.VAL_FILE, "val", load_cached_data)
    test_data = process_data(Config.TEST_FILE, "test", load_cached_data)

    # Debug Subsampling
    if debug:
        print(f"DEBUG MODE: Subsampling to {Config.DEBUG_SUBSET_SIZE} samples.")
        limit = Config.DEBUG_SUBSET_SIZE

        def slice_dict(d, n):
            return {k: v[:n] for k, v in d.items()}

        train_data = slice_dict(train_data, limit)
        val_data = slice_dict(val_data, limit)
        test_data = slice_dict(test_data, limit)

    # Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # Create DataLoaders
    # Note: drop_last=True for train to avoid unstable batch norm/stats on small last batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
