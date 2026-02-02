import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Tokenizer Mappings
NUCLEOTIDE_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def parse_structure_to_distance(structure_str):
    """
    Parses a dot-bracket structure string and calculates signed pairing distances.
    If index i is paired with j:
       distance[i] = j - i
    If unpaired, distance[i] = 0.
    """
    n = len(structure_str)
    distances = np.zeros(n, dtype=np.float32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Pair is (j, i) where j < i
                # For position j: pair is i, dist = i - j (positive)
                # For position i: pair is j, dist = j - i (negative)
                distances[j] = float(i - j)
                distances[i] = float(j - i)
    return distances


class RNADataset(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data["ids"])

    def __getitem__(self, idx):
        item = {
            "ids": self.data["ids"][idx],
            "sequences": self.data["sequences"][idx],
            "loop_types": self.data["loop_types"][idx],
            "pair_dists": self.data["pair_dists"][idx],
        }

        if not self.is_test:
            item["targets"] = self.data["targets"][idx]

        return item


def preprocess_dataframe(df, is_test=False):
    """
    Converts a pandas DataFrame into a dictionary of tensors suitable for the RNADataset.
    """
    # 1. IDs
    ids = df["id"].tolist()

    # 2. Sequences -> Indices
    sequences = []
    for seq in df["sequence"]:
        tokens = [NUCLEOTIDE_MAP.get(c, 0) for c in seq]
        sequences.append(tokens)
    sequences = torch.tensor(sequences, dtype=torch.long)

    # 3. Loop Types -> Indices
    loop_types = []
    for lt in df["predicted_loop_type"]:
        tokens = [LOOP_TYPE_MAP.get(c, 0) for c in lt]
        loop_types.append(tokens)
    loop_types = torch.tensor(loop_types, dtype=torch.long)

    # 4. Structure -> Signed Pairing Distances
    pair_dists = []
    for struct in df["structure"]:
        dists = parse_structure_to_distance(struct)
        pair_dists.append(dists)
    pair_dists = torch.tensor(np.array(pair_dists), dtype=torch.float32)

    # 5. Targets (if not test)
    targets = None
    if not is_test:
        # Extract the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C
        # These columns contain lists of length 68
        t_react = np.vstack(df["reactivity"].values)
        t_mg_ph10 = np.vstack(df["deg_Mg_pH10"].values)
        t_mg_50c = np.vstack(df["deg_Mg_50C"].values)

        # Stack into (N, 68, 3)
        targets_np = np.stack([t_react, t_mg_ph10, t_mg_50c], axis=2)
        targets = torch.tensor(targets_np, dtype=torch.float32)

    return {
        "ids": ids,
        "sequences": sequences,
        "loop_types": loop_types,
        "pair_dists": pair_dists,
        "targets": targets,
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug_subset=None,
):
    """
    Loads data, processes it (with caching), and returns DataLoaders.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers for DataLoader.
        load_cached_data (bool): Whether to try loading from cache.
        debug_subset (int, optional): If provided, limits dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """

    cache_path = os.path.join(Config.WORKING_DIR, "processed_data.pt")
    data_dict = None

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data_dict = torch.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            data_dict = None

    # 2. Process if needed
    if data_dict is None:
        print("Processing data from scratch...")

        # Load Parquet files from Metadata directory
        df_train = pd.read_parquet(Config.TRAIN_FILE)
        df_val = pd.read_parquet(Config.VAL_FILE)
        df_test = pd.read_parquet(Config.TEST_FILE)

        # Process into tensors
        train_data = preprocess_dataframe(df_train, is_test=False)
        val_data = preprocess_dataframe(df_val, is_test=False)
        test_data = preprocess_dataframe(df_test, is_test=True)

        data_dict = {"train": train_data, "val": val_data, "test": test_data}

        # Save Cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        torch.save(data_dict, cache_path)
        print(f"Data processed and saved to {cache_path}")

    # 3. Create Datasets
    train_data = data_dict["train"]
    val_data = data_dict["val"]
    test_data = data_dict["test"]

    # Apply Debug Subset if requested
    if debug_subset is not None:
        print(f"Debugging: Slicing datasets to {debug_subset} samples.")

        def slice_dict(d, n):
            d["ids"] = d["ids"][:n]
            d["sequences"] = d["sequences"][:n]
            d["loop_types"] = d["loop_types"][:n]
            d["pair_dists"] = d["pair_dists"][:n]
            if d["targets"] is not None:
                d["targets"] = d["targets"][:n]
            return d

        train_data = slice_dict(train_data, debug_subset)
        val_data = slice_dict(val_data, debug_subset)
        test_data = slice_dict(test_data, debug_subset)

    train_dataset = RNADataset(train_data, is_test=False)
    val_dataset = RNADataset(val_data, is_test=False)
    test_dataset = RNADataset(test_data, is_test=True)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
