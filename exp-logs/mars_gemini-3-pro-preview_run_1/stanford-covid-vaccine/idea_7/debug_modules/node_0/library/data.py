import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Tokenization Maps
SEQ_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
STRUCT_MAP = {"(": 0, ")": 1, ".": 2}
LOOP_MAP = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}

TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA Degradation Prediction.
    Wraps pre-processed numpy arrays.
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.ids = data_dict["ids"]
        self.seq = data_dict["seq"]
        self.struct = data_dict["struct"]
        self.loop = data_dict["loop"]

        if mode != "test":
            self.targets = data_dict["targets"]
            self.mask = data_dict["mask"]
        else:
            self.targets = None
            self.mask = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Convert numpy arrays to tensors
        item = {
            "seq": torch.tensor(self.seq[idx], dtype=torch.long),
            "struct": torch.tensor(self.struct[idx], dtype=torch.long),
            "loop": torch.tensor(self.loop[idx], dtype=torch.long),
            "id": self.ids[idx],
        }

        if self.mode != "test":
            # Targets: (107, 5)
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)
            # Mask: (107,) - 1 for scored positions, 0 for unscored
            item["mask"] = torch.tensor(self.mask[idx], dtype=torch.float32)

        return item


def tokenize_sequence(seq_series, token_map, max_len):
    """
    Tokenizes a pandas series of strings into a numpy integer array.
    """
    # Create a translation table or apply map
    # Using apply with map is straightforward
    tokenized = seq_series.apply(lambda x: [token_map.get(c, 0) for c in x])
    # Stack into 2D array
    return np.vstack(tokenized.values).astype(np.int32)


def process_data(parquet_path, mode="train"):
    """
    Reads parquet file, tokenizes inputs, and formats targets.
    Returns a dictionary of numpy arrays.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Tokenize Inputs
    ids = df["id"].values
    seq_arr = tokenize_sequence(df["sequence"], SEQ_MAP, Config.SEQ_LEN)
    struct_arr = tokenize_sequence(df["structure"], STRUCT_MAP, Config.SEQ_LEN)
    loop_arr = tokenize_sequence(df["predicted_loop_type"], LOOP_MAP, Config.SEQ_LEN)

    data_dict = {"ids": ids, "seq": seq_arr, "struct": struct_arr, "loop": loop_arr}

    if mode != "test":
        # Process Targets
        # Targets are lists of length 68 in the dataframe
        # We need to create a (N, 107, 5) tensor
        num_samples = len(df)
        targets_arr = np.zeros(
            (num_samples, Config.SEQ_LEN, Config.NUM_TARGETS), dtype=np.float32
        )
        mask_arr = np.zeros((num_samples, Config.SEQ_LEN), dtype=np.float32)

        for i, col in enumerate(TARGET_COLS):
            # Extract column data (list of lists)
            col_values = df[col].values
            # Stack them: result shape (N, 68)
            stacked_col = np.vstack(col_values)

            # Place into the (N, 107, 5) array
            # We fill the first SCORED_LEN positions
            targets_arr[:, : Config.SCORED_LEN, i] = stacked_col

        # Create mask
        mask_arr[:, : Config.SCORED_LEN] = 1.0

        data_dict["targets"] = targets_arr
        data_dict["mask"] = mask_arr

    return data_dict


def get_dataloaders(load_cached_data=True):
    """
    Main function to get DataLoaders.
    Handles caching and debug logic.
    """
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Helper to handle cache logic
    def load_or_process(cache_path, parquet_path, mode):
        if load_cached_data and os.path.exists(cache_path):
            # print(f"Loading cached data from {cache_path}")
            loaded = np.load(cache_path, allow_pickle=True)
            # Convert NpzFile to dict
            data = {k: loaded[k] for k in loaded.files}
            return data
        else:
            # print(f"Processing data from {parquet_path}")
            data = process_data(parquet_path, mode)
            np.savez(cache_path, **data)
            return data

    # 1. Load Data
    train_data = load_or_process(Config.CACHE_TRAIN, Config.TRAIN_PATH, "train")
    val_data = load_or_process(Config.CACHE_VAL, Config.VAL_PATH, "val")
    test_data = load_or_process(Config.CACHE_TEST, Config.TEST_PATH, "test")

    # 2. Handle Debug Mode (Subset data)
    if Config.DEBUG:
        debug_size = 100
        # print(f"DEBUG mode: Subsetting to {debug_size} samples")
        for d in [train_data, val_data]:
            for k in d.keys():
                d[k] = d[k][:debug_size]
        # Test data might be smaller, but subset anyway
        for k in test_data.keys():
            test_data[k] = test_data[k][:debug_size]

    # 3. Create Datasets
    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")
    test_dataset = RNADataset(test_data, mode="test")

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
