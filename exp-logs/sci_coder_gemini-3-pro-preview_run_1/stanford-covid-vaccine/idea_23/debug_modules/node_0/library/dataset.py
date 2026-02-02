import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Tokenization Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_to_distance(structure_str):
    """
    Parses a dot-bracket structure string into a signed pairing distance vector.

    Args:
        structure_str (str): Dot-bracket string (e.g., "((..))").

    Returns:
        np.ndarray: Array of shape (len,) containing signed distances.
                    If i is paired with j:
                        dist[i] = j - i
                        dist[j] = i - j
                    If unpaired: 0
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
                # Calculate distance
                dist = i - open_idx

                # Assign signed distances
                distances[open_idx] = dist  # Positive for opening (points forward)
                distances[i] = -dist  # Negative for closing (points backward)

    return distances


def preprocess_dataframe(df, mode="train"):
    """
    Converts a pandas DataFrame into numpy arrays for features and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train', 'val', or 'test'.

    Returns:
        dict: Dictionary containing numpy arrays for 'ids', 'seq', 'loop', 'dist', and optionally 'targets'.
    """
    # 1. Process IDs
    ids = df["id"].values

    # 2. Process Sequence and Loop Types
    # Vectorized mapping is faster, but list comprehension is clearer for strings
    seq_list = []
    loop_list = []
    dist_list = []

    sequences = df["sequence"].values
    loops = df["predicted_loop_type"].values
    structures = df["structure"].values

    for seq, loop, struct in zip(sequences, loops, structures):
        # Sequence encoding
        seq_enc = [SEQ_MAP.get(c, 0) for c in seq]
        seq_list.append(seq_enc)

        # Loop encoding
        loop_enc = [LOOP_MAP.get(c, 0) for c in loop]
        loop_list.append(loop_enc)

        # Structure distance
        dist_enc = parse_structure_to_distance(struct)
        dist_list.append(dist_enc)

    # Convert to numpy arrays
    X_seq = np.array(seq_list, dtype=np.int32)
    X_loop = np.array(loop_list, dtype=np.int32)
    X_dist = np.array(dist_list, dtype=np.int32)

    data_dict = {"ids": ids, "seq": X_seq, "loop": X_loop, "dist": X_dist}

    # 3. Process Targets (if available)
    if mode in ["train", "val"]:
        # Extract target columns defined in Config
        # Each cell in the dataframe column is a list/array of floats
        targets_list = []

        # We need to stack the specific target columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # Shape per sample: (68, 3)

        # Pre-fetch columns to avoid repetitive indexing
        target_cols_data = [df[col].values for col in Config.TARGET_COLS]

        num_samples = len(df)

        # Iterate and stack
        # Assuming all target lists are length 68 (Config.PRED_LEN)
        # We construct a (N, 68, 3) array

        # Initialize array
        y = np.zeros(
            (num_samples, Config.PRED_LEN, len(Config.TARGET_COLS)), dtype=np.float32
        )

        for i in range(num_samples):
            for j, col_data in enumerate(target_cols_data):
                # col_data[i] is the list of values for sample i
                # We take the first PRED_LEN values just in case, though metadata guarantees consistency
                vals = col_data[i]
                length = min(len(vals), Config.PRED_LEN)
                y[i, :length, j] = vals[:length]

        data_dict["targets"] = y

    return data_dict


def load_and_preprocess(mode="train", load_cached_data=True):
    """
    Loads data from Parquet, preprocesses it, and caches it as .npz.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Data dictionary with numpy arrays.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_data.npz")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            # np.load returns a NpzFile object, we convert to dict
            loaded = np.load(cache_path, allow_pickle=True)
            data_dict = {k: loaded[k] for k in loaded.files}

            # Sanity check on shapes
            if "seq" in data_dict:
                return data_dict
            else:
                print("Cache file corrupted or missing keys. Re-processing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Re-processing...")

    # 2. Process from scratch
    print(f"Processing {mode} data from raw files...")

    if mode == "train":
        path = Config.TRAIN_DATA_PATH
    elif mode == "val":
        path = Config.VAL_DATA_PATH
    else:
        path = Config.TEST_DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)
    data_dict = preprocess_dataframe(df, mode=mode)

    # 3. Save to cache
    print(f"Saving processed {mode} data to {cache_path}...")
    np.savez_compressed(cache_path, **data_dict)

    return data_dict


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.ids = data_dict["ids"]
        self.seq = data_dict["seq"]
        self.loop = data_dict["loop"]
        self.dist = data_dict["dist"]

        # Targets only exist for train/val
        self.targets = data_dict.get("targets", None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert to torch tensors
        seq_tensor = torch.tensor(self.seq[idx], dtype=torch.long)
        loop_tensor = torch.tensor(self.loop[idx], dtype=torch.long)
        dist_tensor = torch.tensor(
            self.dist[idx], dtype=torch.float32
        )  # Float for sinusoidal encoding later

        item = {
            "seq": seq_tensor,
            "loop": loop_tensor,
            "dist": dist_tensor,
            "id": self.ids[idx],
        }

        if self.targets is not None:
            # Targets shape: (68, 3)
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = target_tensor

        return item


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_data = load_and_preprocess("train", load_cached_data)
    val_data = load_and_preprocess("val", load_cached_data)
    test_data = load_and_preprocess("test", load_cached_data)

    # Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # Create DataLoaders
    # Train loader: Shuffle=True, Drop_last=True (usually good for batch norm/training stability)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val loader: Shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Test loader: Shuffle=False
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
