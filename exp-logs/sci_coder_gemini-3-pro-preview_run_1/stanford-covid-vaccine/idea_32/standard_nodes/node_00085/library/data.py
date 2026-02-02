import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Token Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_to_distance(structure_str, seq_len):
    """
    Parses a dot-bracket structure string into a signed distance vector.
    For a pair (i, j), index i has value j-i, index j has value i-j.
    Unpaired bases have value 0.
    """
    # Initialize with zeros
    dist_vector = np.zeros(seq_len, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i is current (closing), j is popped (opening)
                # Distance for opening base j: i - j (positive)
                # Distance for closing base i: j - i (negative)
                dist_vector[j] = float(i - j)
                dist_vector[i] = float(j - i)

    return dist_vector


def process_dataframe(df, config, is_test=False):
    """
    Converts dataframe columns into numpy arrays for the model.
    """
    num_samples = len(df)
    seq_len = config.seq_len

    # Pre-allocate arrays
    seqs = np.zeros((num_samples, seq_len), dtype=np.int64)
    loops = np.zeros((num_samples, seq_len), dtype=np.int64)
    dists = np.zeros((num_samples, seq_len), dtype=np.float32)

    # Targets: (N, seq_len, 3)
    # We pad the 68-length targets to 107 with zeros.
    # Mask: (N, seq_len) -> 1 for scored positions, 0 otherwise.
    targets = np.zeros((num_samples, seq_len, 3), dtype=np.float32)
    masks = np.zeros((num_samples, seq_len), dtype=np.float32)

    ids = df["id"].values

    # Iterate and process
    for idx, row in df.iterrows():
        # 1. Sequence Tokenization
        seq_str = row["sequence"]
        seqs[idx] = np.array([SEQ_MAP.get(c, 0) for c in seq_str])

        # 2. Loop Type Tokenization
        loop_str = row["predicted_loop_type"]
        loops[idx] = np.array([LOOP_MAP.get(c, 0) for c in loop_str])

        # 3. Structure Distance Encoding
        struct_str = row["structure"]
        dists[idx] = parse_structure_to_distance(struct_str, seq_len)

        # 4. Targets & Mask
        # seq_scored is usually 68
        scored_len = row["seq_scored"]
        masks[idx, :scored_len] = 1.0

        if not is_test:
            # Extract the 3 specific target columns
            # Each column in the parquet is a list/array of floats
            for t_i, col_name in enumerate(config.target_cols):
                val_array = np.array(row[col_name])
                # Ensure we don't overflow if data is weird, though metadata checks passed
                curr_len = min(len(val_array), seq_len)
                targets[idx, :curr_len, t_i] = val_array[:curr_len]

    return {
        "seqs": seqs,
        "loops": loops,
        "dists": dists,
        "targets": targets,
        "masks": masks,
        "ids": ids,
    }


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.seqs = data_dict["seqs"]
        self.loops = data_dict["loops"]
        self.dists = data_dict["dists"]
        self.targets = data_dict["targets"]
        self.masks = data_dict["masks"]
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Convert to tensors
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        loop = torch.tensor(self.loops[idx], dtype=torch.long)
        dist = torch.tensor(self.dists[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        return seq, loop, dist, target, mask


def get_dataloaders(config, load_cached_data=True):
    """
    Main entry point to get DataLoaders. Handles caching.
    """
    seed_everything(config.seed)

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # Define cache paths
    cache_train = os.path.join(config.working_dir, "train_data.npz")
    cache_val = os.path.join(config.working_dir, "val_data.npz")
    cache_test = os.path.join(config.working_dir, "test_data.npz")

    data_cache = {"train": cache_train, "val": cache_val, "test": cache_test}

    loaded_data = {}

    # Check if we can load from cache
    all_cached = all(os.path.exists(p) for p in data_cache.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        for split, path in data_cache.items():
            # Allow_pickle=True is needed if object arrays (like strings) are saved,
            # though we tried to keep numericals. ids are strings.
            loaded = np.load(path, allow_pickle=True)
            loaded_data[split] = {k: loaded[k] for k in loaded.files}
    else:
        print("Processing data from source Parquet files...")

        # Load Parquet files
        df_train = pd.read_parquet(config.train_file)
        df_val = pd.read_parquet(config.val_file)
        df_test = pd.read_parquet(config.test_file)

        # Process
        train_dict = process_dataframe(df_train, config, is_test=False)
        val_dict = process_dataframe(df_val, config, is_test=False)
        test_dict = process_dataframe(df_test, config, is_test=True)

        # Save to cache
        np.savez_compressed(cache_train, **train_dict)
        np.savez_compressed(cache_val, **val_dict)
        np.savez_compressed(cache_test, **test_dict)

        loaded_data["train"] = train_dict
        loaded_data["val"] = val_dict
        loaded_data["test"] = test_dict

    # Create Datasets
    train_dataset = RNADataset(loaded_data["train"])
    val_dataset = RNADataset(loaded_data["val"])
    test_dataset = RNADataset(loaded_data["test"])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
