import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# =========================================================================
# Token Mappings
# =========================================================================
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_pair_indices(structure_str):
    """
    Parses a dot-bracket structure string to find base pairs.
    Returns an array of indices where arr[i] = j if i is paired with j.
    If i is unpaired, arr[i] = -1.
    """
    n = len(structure_str)
    pair_indices = np.full(n, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_indices[i] = j
                pair_indices[j] = i

    return pair_indices


def one_hot_encode(sequence, structure, loop_type):
    """
    Creates the (Seq_Len, 14) input tensor.
    Channels: 4 (Seq) + 3 (Struct) + 7 (Loop)
    """
    seq_len = len(sequence)
    encoding = np.zeros((seq_len, 14), dtype=np.float32)

    for i in range(seq_len):
        # Sequence (0-3)
        if sequence[i] in SEQ_MAP:
            encoding[i, SEQ_MAP[sequence[i]]] = 1.0

        # Structure (4-6)
        if structure[i] in STRUCT_MAP:
            encoding[i, 4 + STRUCT_MAP[structure[i]]] = 1.0

        # Loop Type (7-13)
        if loop_type[i] in LOOP_MAP:
            encoding[i, 7 + LOOP_MAP[loop_type[i]]] = 1.0

    return encoding


def preprocess_dataset(parquet_path, cache_path, mode="train", load_cached_data=True):
    """
    Loads data from Parquet, processes features, and caches them as .npz.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Verify keys exist
            required_keys = ["inputs", "pair_indices", "ids"]
            if mode != "test":
                required_keys.append("targets")

            if all(k in data for k in required_keys):
                # Reconstruct dictionary
                result = {k: data[k] for k in required_keys}
                return result
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Load Metadata
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # 3. Process Features
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    inputs = np.zeros((num_samples, seq_len, 14), dtype=np.float32)
    pair_indices = np.zeros((num_samples, seq_len), dtype=np.int64)
    ids = df["id"].values

    # Targets (only for train/val)
    targets = None
    if mode != "test":
        # Targets are lists of length 68 in the dataframe
        # We stack them into a (N, 68, 5) tensor
        targets = np.zeros((num_samples, Config.PRED_LEN, 5), dtype=np.float32)

    for idx, row in df.iterrows():
        # Input Features
        inputs[idx] = one_hot_encode(
            row["sequence"], row["structure"], row["predicted_loop_type"]
        )

        # Adjacency
        pair_indices[idx] = get_pair_indices(row["structure"])

        # Targets
        if mode != "test":
            # Extract target lists
            t_react = row["reactivity"]
            t_mg_ph10 = row["deg_Mg_pH10"]
            t_ph10 = row["deg_pH10"]
            t_mg_50c = row["deg_Mg_50C"]
            t_50c = row["deg_50C"]

            # Stack columns: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            # Ensure they are lists/arrays and take first 68
            # (Parquet should preserve list type, but safe to cast)
            sample_targets = np.column_stack(
                [t_react, t_mg_ph10, t_ph10, t_mg_50c, t_50c]
            )

            # Assign (ensure shape match)
            # Note: sample_targets should be (68, 5)
            targets[idx] = sample_targets[: Config.PRED_LEN, :]

    # 4. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    save_dict = {"inputs": inputs, "pair_indices": pair_indices, "ids": ids}
    if targets is not None:
        save_dict["targets"] = targets

    np.savez_compressed(cache_path, **save_dict)

    return save_dict


class RNADataset(Dataset):
    def __init__(self, data_dict, mode="train"):
        self.inputs = data_dict["inputs"]
        self.pair_indices = data_dict["pair_indices"]
        self.ids = data_dict["ids"]
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

        # Pair Indices: (107,)
        pairs = torch.tensor(self.pair_indices[idx], dtype=torch.long)

        item = {"inputs": x, "pair_indices": pairs, "id": self.ids[idx]}

        if self.targets is not None:
            # Targets: (68, 5)
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["targets"] = y

        return item


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    seed_everything(Config.SEED)

    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_cache.npz")
    val_cache = os.path.join(Config.WORKING_DIR, "val_cache.npz")
    test_cache = os.path.join(Config.WORKING_DIR, "test_cache.npz")

    # 1. Train Data
    print("Preparing Train Data...")
    train_data = preprocess_dataset(
        Config.TRAIN_PATH, train_cache, mode="train", load_cached_data=load_cached_data
    )
    train_dataset = RNADataset(train_data, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    # 2. Val Data
    print("Preparing Val Data...")
    val_data = preprocess_dataset(
        Config.VAL_PATH, val_cache, mode="val", load_cached_data=load_cached_data
    )
    val_dataset = RNADataset(val_data, mode="val")
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    # 3. Test Data
    print("Preparing Test Data...")
    test_data = preprocess_dataset(
        Config.TEST_PATH, test_cache, mode="test", load_cached_data=load_cached_data
    )
    test_dataset = RNADataset(test_data, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True
    )

    return train_loader, val_loader, test_loader
