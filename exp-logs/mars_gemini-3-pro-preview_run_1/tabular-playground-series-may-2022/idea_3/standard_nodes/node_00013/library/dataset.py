import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class Tokenizer:
    """
    Simple character-level tokenizer for the f_27 sequence feature.
    Maps characters A-Z to integers 1-26.
    """

    def __init__(self):
        self.char_to_idx = {}
        self.vocab_size = Config.VOCAB_SIZE
        # Initialize vocabulary with A-Z
        # 0 is reserved for padding/unknown
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx, char in enumerate(alphabet):
            self.char_to_idx[char] = idx + 1

    def fit(self, texts):
        """
        No-op for this specific task as vocabulary is fixed (A-Z),
        but kept for API consistency.
        """
        pass

    def transform(self, texts):
        """
        Converts a list/series of strings into a numpy array of shape (N, max_len).
        """
        batch_size = len(texts)
        max_len = Config.MAX_SEQ_LEN

        # Initialize with zeros (padding index)
        output = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, text in enumerate(texts):
            # Truncate if longer than max_len
            chars = list(str(text))[:max_len]
            for j, char in enumerate(chars):
                if char in self.char_to_idx:
                    output[i, j] = self.char_to_idx[char]
                else:
                    output[i, j] = 0  # Unknown

        return output


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    Returns a dictionary with sequence, numerical features, and target.
    """

    def __init__(self, sequence_data, numerical_data, targets=None, ids=None):
        self.sequence_data = torch.tensor(sequence_data, dtype=torch.long)
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float32)

        self.targets = None
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)

        self.ids = ids

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        item = {
            "sequence": self.sequence_data[idx],
            "numerical": self.numerical_data[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def _get_cache_paths(base_dir):
    """Returns a dictionary of file paths for cached data."""
    return {
        "train_seq": os.path.join(base_dir, "X_train_seq.npy"),
        "train_num": os.path.join(base_dir, "X_train_num.npy"),
        "train_target": os.path.join(base_dir, "y_train.npy"),
        "val_seq": os.path.join(base_dir, "X_val_seq.npy"),
        "val_num": os.path.join(base_dir, "X_val_num.npy"),
        "val_target": os.path.join(base_dir, "y_val.npy"),
        "test_seq": os.path.join(base_dir, "X_test_seq.npy"),
        "test_num": os.path.join(base_dir, "X_test_num.npy"),
        "test_ids": os.path.join(base_dir, "ids_test.npy"),
    }


def process_data(load_cached_data=True):
    """
    Loads raw data, preprocesses it (tokenization, scaling), and caches the result.
    If cached data exists and load_cached_data is True, loads from disk.
    """
    cache_paths = _get_cache_paths(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and all_cached:
        print("Loading cached data from", Config.WORKING_DIR)
        X_train_seq = np.load(cache_paths["train_seq"])
        X_train_num = np.load(cache_paths["train_num"])
        y_train = np.load(cache_paths["train_target"])

        X_val_seq = np.load(cache_paths["val_seq"])
        X_val_num = np.load(cache_paths["val_num"])
        y_val = np.load(cache_paths["val_target"])

        X_test_seq = np.load(cache_paths["test_seq"])
        X_test_num = np.load(cache_paths["test_num"])
        ids_test = np.load(cache_paths["test_ids"])

        return (
            (X_train_seq, X_train_num, y_train),
            (X_val_seq, X_val_num, y_val),
            (X_test_seq, X_test_num, ids_test),
        )

    print("Processing data from scratch...")

    # Load Metadata CSVs
    print(f"Reading {Config.TRAIN_DATA_PATH}...")
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    print(f"Reading {Config.VAL_DATA_PATH}...")
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    print(f"Reading {Config.TEST_DATA_PATH}...")
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Identify columns
    # Numerical columns are all columns except id, target, f_27, and source_path
    exclude_cols = [Config.ID_COL, Config.TARGET_COL, Config.SEQ_COL, "source_path"]
    num_cols = [c for c in df_train.columns if c not in exclude_cols]

    print(f"Identified {len(num_cols)} numerical features.")

    # 1. Process Numerical Features
    scaler = StandardScaler()
    print("Fitting StandardScaler on training data...")
    X_train_num = scaler.fit_transform(df_train[num_cols].values.astype(np.float32))
    X_val_num = scaler.transform(df_val[num_cols].values.astype(np.float32))
    X_test_num = scaler.transform(df_test[num_cols].values.astype(np.float32))

    # 2. Process Sequence Features
    tokenizer = Tokenizer()
    print("Tokenizing sequence data...")
    X_train_seq = tokenizer.transform(df_train[Config.SEQ_COL].values)
    X_val_seq = tokenizer.transform(df_val[Config.SEQ_COL].values)
    X_test_seq = tokenizer.transform(df_test[Config.SEQ_COL].values)

    # 3. Extract Targets and IDs
    y_train = df_train[Config.TARGET_COL].values.astype(np.float32)
    y_val = df_val[Config.TARGET_COL].values.astype(np.float32)
    ids_test = df_test[Config.ID_COL].values

    # 4. Cache Data
    print("Caching processed data...")
    np.save(cache_paths["train_seq"], X_train_seq)
    np.save(cache_paths["train_num"], X_train_num)
    np.save(cache_paths["train_target"], y_train)

    np.save(cache_paths["val_seq"], X_val_seq)
    np.save(cache_paths["val_num"], X_val_num)
    np.save(cache_paths["val_target"], y_val)

    np.save(cache_paths["test_seq"], X_test_seq)
    np.save(cache_paths["test_num"], X_test_num)
    np.save(cache_paths["test_ids"], ids_test)

    return (
        (X_train_seq, X_train_num, y_train),
        (X_val_seq, X_val_num, y_val),
        (X_test_seq, X_test_num, ids_test),
    )


def get_dataloaders(debug=False, batch_size=Config.BATCH_SIZE):
    """
    Returns train, validation, and test DataLoaders.

    Args:
        debug (bool): If True, subsets the data for quick debugging.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load processed data (either from cache or processed fresh)
    # We always try to load cached data unless explicitly disabled in Config,
    # but the function argument here allows overriding if needed in future.
    train_data, val_data, test_data = process_data(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    X_train_seq, X_train_num, y_train = train_data
    X_val_seq, X_val_num, y_val = val_data
    X_test_seq, X_test_num, ids_test = test_data

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled. Subsetting to {Config.DEBUG_SAMPLE_SIZE} samples.")
        limit = min(len(y_train), Config.DEBUG_SAMPLE_SIZE)
        X_train_seq = X_train_seq[:limit]
        X_train_num = X_train_num[:limit]
        y_train = y_train[:limit]

        limit_val = min(len(y_val), Config.DEBUG_SAMPLE_SIZE)
        X_val_seq = X_val_seq[:limit_val]
        X_val_num = X_val_num[:limit_val]
        y_val = y_val[:limit_val]

        limit_test = min(len(ids_test), Config.DEBUG_SAMPLE_SIZE)
        X_test_seq = X_test_seq[:limit_test]
        X_test_num = X_test_num[:limit_test]
        ids_test = ids_test[:limit_test]

    # Create Datasets
    train_dataset = ManufacturingDataset(X_train_seq, X_train_num, targets=y_train)
    val_dataset = ManufacturingDataset(X_val_seq, X_val_num, targets=y_val)
    test_dataset = ManufacturingDataset(X_test_seq, X_test_num, ids=ids_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
