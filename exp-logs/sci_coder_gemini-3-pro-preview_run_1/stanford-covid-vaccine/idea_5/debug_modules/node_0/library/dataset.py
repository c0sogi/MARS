import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    SEQ_LENGTH,
    SEQ_SCORED,
    TARGET_COLS,
    TOKEN2INT_SEQ,
    TOKEN2INT_STRUCT,
    TOKEN2INT_LOOP,
    BATCH_SIZE,
    DEBUG,
    DEBUG_SUBSET_SIZE,
)


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, data):
        self.ids = data["ids"]
        self.sequences = torch.tensor(data["sequences"], dtype=torch.long)
        self.structures = torch.tensor(data["structures"], dtype=torch.long)
        self.loop_types = torch.tensor(data["loop_types"], dtype=torch.long)

        # Targets might not exist for test set, handle accordingly
        if "targets" in data:
            self.targets = torch.tensor(data["targets"], dtype=torch.float32)
            self.masks = torch.tensor(data["masks"], dtype=torch.float32)
            self.weights = torch.tensor(data["weights"], dtype=torch.float32)
        else:
            self.targets = None
            self.masks = None
            self.weights = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {
            "id": self.ids[idx],
            "sequence": self.sequences[idx],
            "structure": self.structures[idx],
            "predicted_loop_type": self.loop_types[idx],
        }

        if self.targets is not None:
            sample["targets"] = self.targets[idx]
            sample["mask"] = self.masks[idx]
            sample["weight"] = self.weights[idx]

        return sample


def tokenize_sequence(seq_series, token_map, max_len):
    """
    Tokenizes a pandas Series of strings into a numpy array of integers.
    """
    num_samples = len(seq_series)
    tokenized = np.zeros((num_samples, max_len), dtype=np.int32)

    for i, seq in enumerate(seq_series):
        # Truncate if necessary (though data should be fixed length)
        seq = seq[:max_len]
        # Map characters to integers
        indices = [token_map.get(char, 0) for char in seq]
        tokenized[i, : len(indices)] = indices

    return tokenized


def process_dataframe(df, is_test=False):
    """
    Process a DataFrame into a dictionary of numpy arrays.
    """
    # 1. Tokenize Inputs
    sequences = tokenize_sequence(df["sequence"], TOKEN2INT_SEQ, SEQ_LENGTH)
    structures = tokenize_sequence(df["structure"], TOKEN2INT_STRUCT, SEQ_LENGTH)
    loop_types = tokenize_sequence(
        df["predicted_loop_type"], TOKEN2INT_LOOP, SEQ_LENGTH
    )

    ids = df["id"].values

    data_dict = {
        "ids": ids,
        "sequences": sequences,
        "structures": structures,
        "loop_types": loop_types,
    }

    # 2. Process Targets (if not test)
    if not is_test:
        num_samples = len(df)
        num_targets = len(TARGET_COLS)

        # Initialize targets with zeros: (N, 107, 5)
        targets_array = np.zeros(
            (num_samples, SEQ_LENGTH, num_targets), dtype=np.float32
        )

        # Initialize mask: (N, 107) - 1 for scored positions, 0 otherwise
        masks_array = np.zeros((num_samples, SEQ_LENGTH), dtype=np.float32)
        masks_array[:, :SEQ_SCORED] = 1.0

        # Extract targets from list columns
        # Note: The parquet file stores these as arrays/lists
        for t_idx, col in enumerate(TARGET_COLS):
            # Stack the arrays from the series
            # Each row in df[col] is a list/array of length 68
            col_values = np.vstack(df[col].values)
            # Assign to the first 68 positions of the target array
            targets_array[:, :SEQ_SCORED, t_idx] = col_values

        # 3. Process Weights (Signal to Noise)
        # Use signal_to_noise column if available, else default to 1.0
        if "signal_to_noise" in df.columns:
            weights = df["signal_to_noise"].values.astype(np.float32)
            # Clip negative weights to 0 just in case
            weights = np.maximum(weights, 0.0)
        else:
            weights = np.ones(num_samples, dtype=np.float32)

        data_dict["targets"] = targets_array
        data_dict["masks"] = masks_array
        data_dict["weights"] = weights

    return data_dict


def get_dataloaders(load_cached_data=True, batch_size=BATCH_SIZE, num_workers=2):
    """
    Loads data, processes it (with caching), and returns DataLoaders.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    splits = [
        ("train", TRAIN_PATH, False),
        ("val", VAL_PATH, False),
        ("test", TEST_PATH, True),
    ]

    loaders = []

    for split_name, file_path, is_test in splits:
        cache_file = os.path.join(WORKING_DIR, f"{split_name}_data.npz")

        data_dict = None

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split_name} data from cache: {cache_file}")
            try:
                loaded = np.load(cache_file, allow_pickle=True)
                # Convert NpzFile to dict
                data_dict = {key: loaded[key] for key in loaded.files}
            except Exception as e:
                print(f"Failed to load cache for {split_name}: {e}")
                data_dict = None

        # 2. Process from Scratch if needed
        if data_dict is None:
            print(f"Processing {split_name} data from source: {file_path}")
            df = pd.read_parquet(file_path)

            if DEBUG:
                df = df.iloc[:DEBUG_SUBSET_SIZE].copy()
                print(f"DEBUG MODE: Reduced {split_name} size to {len(df)}")

            data_dict = process_dataframe(df, is_test=is_test)

            # Save to cache
            print(f"Saving {split_name} data to cache: {cache_file}")
            np.savez(cache_file, **data_dict)

        # 3. Create Dataset and Loader
        dataset = RNADataset(data_dict)

        shuffle = split_name == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)

    return loaders[0], loaders[1], loaders[2]
