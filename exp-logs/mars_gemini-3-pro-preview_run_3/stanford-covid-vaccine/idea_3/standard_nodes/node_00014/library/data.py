import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA degradation prediction.
    """

    def __init__(self, inputs, targets=None, ids=None):
        """
        Args:
            inputs (np.ndarray): Shape (N, 107, 14). Input features.
            targets (np.ndarray, optional): Shape (N, 107, 5). Target values.
            ids (list or np.ndarray, optional): Sample IDs.
        """
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.ids = ids

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # inputs: (107, 14)
        x = self.inputs[idx]

        if self.targets is not None:
            # targets: (107, 5)
            y = self.targets[idx]
            return x, y
        else:
            return x


def get_one_hot_map(chars):
    return {c: i for i, c in enumerate(chars)}


SEQ_MAP = get_one_hot_map(["A", "G", "C", "U"])
STRUCT_MAP = get_one_hot_map([".", "(", ")"])
LOOP_MAP = get_one_hot_map(["S", "M", "I", "B", "H", "E", "X"])


def one_hot_encode(seq_str, map_dict, length):
    """
    One-hot encodes a string sequence based on a mapping dictionary.
    Returns shape (length, len(map_dict)).
    """
    # Create empty array
    encoding = np.zeros((length, len(map_dict)), dtype=np.float32)

    # Fill based on mapping
    for i, char in enumerate(seq_str):
        if i >= length:
            break
        if char in map_dict:
            encoding[i, map_dict[char]] = 1.0

    return encoding


def process_dataframe(df, is_test=False):
    """
    Processes a dataframe into input features and targets.

    Args:
        df (pd.DataFrame): Input dataframe.
        is_test (bool): If True, does not attempt to extract targets.

    Returns:
        dict: {'inputs': np.ndarray, 'targets': np.ndarray (or None), 'ids': list}
    """
    num_samples = len(df)
    seq_len = Config.SEQ_LEN

    # Initialize arrays
    # Channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    inputs = np.zeros((num_samples, seq_len, Config.INPUT_CHANNELS), dtype=np.float32)

    if not is_test:
        targets = np.zeros((num_samples, seq_len, Config.NUM_TARGETS), dtype=np.float32)
    else:
        targets = None

    ids = df["id"].tolist()

    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Iterate and process
    for idx, row in df.iterrows():
        # --- Inputs ---
        # 1. Sequence (4 channels)
        seq_enc = one_hot_encode(row["sequence"], SEQ_MAP, seq_len)

        # 2. Structure (3 channels)
        struct_enc = one_hot_encode(row["structure"], STRUCT_MAP, seq_len)

        # 3. Loop Type (7 channels)
        loop_enc = one_hot_encode(row["predicted_loop_type"], LOOP_MAP, seq_len)

        # Concatenate: (107, 4) + (107, 3) + (107, 7) -> (107, 14)
        inputs[idx] = np.concatenate([seq_enc, struct_enc, loop_enc], axis=1)

        # --- Targets ---
        if not is_test:
            # Targets are provided as lists of length `seq_scored` (68)
            # We pad them to `seq_len` (107) with zeros.
            # The loss function will handle the slicing or masking if needed,
            # but here we provide the full tensor structure.
            for t_i, col in enumerate(target_cols):
                val_list = row[col]
                # Check if it's a valid list/array
                if isinstance(val_list, (list, np.ndarray)):
                    length = len(val_list)
                    # Assign to the first `length` positions
                    targets[idx, :length, t_i] = val_list
                else:
                    # Fallback for missing/malformed data (should not happen in clean data)
                    pass

    return {"inputs": inputs, "targets": targets, "ids": ids}


def load_or_process_data(split, load_cached_data=True):
    """
    Loads data from cache or processes from parquet files.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Processed data dictionary.
    """
    # Determine paths
    if split == "train":
        parquet_path = Config.TRAIN_METADATA
        cache_path = Config.TRAIN_CACHE
    elif split == "val":
        parquet_path = Config.VAL_METADATA
        cache_path = Config.VAL_CACHE
    elif split == "test":
        parquet_path = Config.TEST_METADATA
        cache_path = Config.TEST_CACHE
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading {split} data from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {split} data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Apply Filter for Training Set
    if split == "train" and Config.FILTER_SN:
        initial_len = len(df)
        df = df[df["SN_filter"] == 1].reset_index(drop=True)
        print(f"Applied SN_filter=1 to train set. Rows: {initial_len} -> {len(df)}")

    is_test = split == "test"
    data = process_dataframe(df, is_test=is_test)

    # 3. Save Cache
    try:
        print(f"Saving {split} data to cache: {cache_path}")
        np.save(cache_path, data)
    except Exception as e:
        print(f"Warning: Could not save cache for {split}: {e}")

    return data


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main function to get PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.
        debug (bool): If True, reduces dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Data
    train_data = load_or_process_data("train", load_cached_data)
    val_data = load_or_process_data("val", load_cached_data)
    test_data = load_or_process_data("test", load_cached_data)

    # Debug Slicing
    if debug:
        subset = Config.DEBUG_SUBSET_SIZE
        print(f"DEBUG mode: Slicing datasets to {subset} samples.")

        train_data["inputs"] = train_data["inputs"][:subset]
        train_data["targets"] = train_data["targets"][:subset]
        train_data["ids"] = train_data["ids"][:subset]

        val_data["inputs"] = val_data["inputs"][:subset]
        val_data["targets"] = val_data["targets"][:subset]
        val_data["ids"] = val_data["ids"][:subset]

        test_data["inputs"] = test_data["inputs"][:subset]
        # Test targets are None
        test_data["ids"] = test_data["ids"][:subset]

    # Create Datasets
    train_dataset = RNADataset(
        inputs=train_data["inputs"],
        targets=train_data["targets"],
        ids=train_data["ids"],
    )

    val_dataset = RNADataset(
        inputs=val_data["inputs"], targets=val_data["targets"], ids=val_data["ids"]
    )

    test_dataset = RNADataset(
        inputs=test_data["inputs"],
        targets=None,  # Test set has no targets
        ids=test_data["ids"],
    )

    # Create DataLoaders
    # Note: Shuffle train, but not val/test
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
