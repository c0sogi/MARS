import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Tokenization Mappings
SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def get_structure_distances(structure):
    """
    Parses a dot-bracket structure string and calculates signed pairing distances.
    For a pair (i, j), distance at i is (j - i), distance at j is (i - j).
    Unpaired bases have distance 0.
    """
    length = len(structure)
    distances = np.zeros(length, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # i is current (closing), j is popped (opening)
                # Distance at opening (j): i - j (positive)
                # Distance at closing (i): j - i (negative)
                # Wait, prompt says "Signed Sinusoidal Pairing Distance".
                # Standard convention: dist = pair_index - current_index
                distances[j] = i - j
                distances[i] = j - i

    return distances


def process_data(parquet_path, is_test=False, debug=False):
    """
    Loads data from parquet, processes features and targets.
    Returns dictionary of numpy arrays.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Metadata file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Pre-allocate arrays
    # Inputs: (N, Seq_Len, 3) -> [Seq_Idx, Loop_Idx, Pair_Dist]
    input_array = np.zeros((num_samples, seq_len, 3), dtype=np.int32)

    # Targets: (N, Seq_Len, Num_Targets)
    # Initialize with zeros (padding for unscored positions)
    target_array = np.zeros(
        (num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32
    )

    ids = []

    # Process rows
    for idx, row in df.iterrows():
        # 1. Sequence Tokenization
        seq = row["sequence"]
        seq_encoded = [
            SEQ_MAP.get(c, 0) for c in seq
        ]  # Default to 0 (A) if unknown, though vocab is fixed

        # 2. Loop Type Tokenization
        loop = row["predicted_loop_type"]
        loop_encoded = [LOOP_MAP.get(c, 5) for c in loop]  # Default to 5 (E) if unknown

        # 3. Structure Distance
        struct = row["structure"]
        dists = get_structure_distances(struct)

        # Fill Input Array
        # Use integer indexing for the row in numpy array (idx might be non-sequential in dataframe)
        # So we use an enumerator or just append to list and stack later?
        # Pre-allocation is faster, let's use a counter.
        array_idx = (
            idx if isinstance(idx, int) and idx < num_samples else idx
        )  # DataFrame index might be preserved
        # Safer to iterate with enumerate if we reset index, but let's just use a separate counter

    # Re-implement loop to be safe with indices
    for i, (_, row) in enumerate(df.iterrows()):
        ids.append(row["id"])

        # Inputs
        seq_encoded = [SEQ_MAP.get(c, 0) for c in row["sequence"]]
        loop_encoded = [LOOP_MAP.get(c, 5) for c in row["predicted_loop_type"]]
        dists = get_structure_distances(row["structure"])

        input_array[i, :, 0] = seq_encoded
        input_array[i, :, 1] = loop_encoded
        input_array[i, :, 2] = dists

        # Targets
        if not is_test:
            # Targets are lists of length 68 (Config.SCORED_LEN)
            # We place them in the first 68 positions of the (107,) vector
            for t_idx, col in enumerate(Config.TARGET_COLS):
                val_list = row[col]
                # Ensure it's a list or array
                if hasattr(val_list, "tolist"):
                    val_list = val_list.tolist()

                # Copy available data
                length = min(len(val_list), seq_len)
                target_array[i, :length, t_idx] = val_list[:length]

    return {"inputs": input_array, "targets": target_array, "ids": np.array(ids)}


class RNADataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.from_numpy(data_dict["inputs"]).long()
        self.targets = torch.from_numpy(data_dict["targets"]).float()
        self.ids = data_dict["ids"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Return tuple (inputs, targets)
        # inputs: (Seq_Len, 3)
        # targets: (Seq_Len, Num_Targets)
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Main entry point to get DataLoaders.
    Handles caching logic using .npz files to avoid pickle.
    """

    # Define cache filenames based on Config paths (replacing extension with .npz)
    # We use the names from Config but enforce .npz extension for numpy storage
    train_cache = os.path.splitext(Config.CACHE_TRAIN_PATH)[0] + ".npz"
    val_cache = os.path.splitext(Config.CACHE_VAL_PATH)[0] + ".npz"
    test_cache = os.path.splitext(Config.CACHE_TEST_PATH)[0] + ".npz"

    datasets = {}

    # Helper to load or process
    def get_data(cache_path, metadata_path, is_test):
        data = None
        if load_cached_data and os.path.exists(cache_path):
            try:
                # Load from npz
                loaded = np.load(
                    cache_path, allow_pickle=True
                )  # allow_pickle=True needed for string arrays (ids)
                data = {
                    "inputs": loaded["inputs"],
                    "targets": loaded["targets"],
                    "ids": loaded["ids"],
                }
                print(f"Loaded cached data from {cache_path}")
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}")
                data = None

        if data is None:
            print(f"Processing data from {metadata_path}...")
            data = process_data(metadata_path, is_test=is_test, debug=debug)
            # Save to npz
            np.savez_compressed(
                cache_path,
                inputs=data["inputs"],
                targets=data["targets"],
                ids=data["ids"],
            )
            print(f"Saved processed data to {cache_path}")

        return RNADataset(data)

    # 1. Train Set
    train_ds = get_data(train_cache, Config.TRAIN_METADATA, is_test=False)

    # 2. Val Set
    val_ds = get_data(val_cache, Config.VAL_METADATA, is_test=False)

    # 3. Test Set
    test_ds = get_data(test_cache, Config.TEST_METADATA, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability in training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
