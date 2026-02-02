import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config


class CharTokenizer:
    """
    Simple character-level tokenizer that maps characters to integers.
    """

    def __init__(self):
        self.char_to_idx = {}
        self.vocab_size = 0

    def fit(self, texts):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Start indices at 1 (0 reserved for padding/unknown if needed)
        self.char_to_idx = {c: i + 1 for i, c in enumerate(sorted_chars)}
        self.vocab_size = len(self.char_to_idx) + 1

    def transform(self, texts, max_len=None):
        """
        Converts a list of strings to a numpy array of integers.
        """
        encoded_batch = []
        for text in texts:
            encoded = [self.char_to_idx.get(c, 0) for c in str(text)]
            encoded_batch.append(encoded)

        if max_len is None:
            max_len = max(len(seq) for seq in encoded_batch)

        # Create array padded with 0
        batch_array = np.zeros((len(encoded_batch), max_len), dtype=np.int64)
        for i, seq in enumerate(encoded_batch):
            batch_array[i, : len(seq)] = seq

        return batch_array


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing data.
    """

    def __init__(self, X_num, X_seq, y=None):
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.X_seq = torch.as_tensor(X_seq, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        sample = {
            "numerical_features": self.X_num[idx],
            "sequence_features": self.X_seq[idx],
        }
        if self.y is not None:
            sample["target"] = self.y[idx]
        return sample


def _process_data(load_cached_data=True):
    """
    Internal function to load, process, and cache data.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "X_num_train": os.path.join(cache_dir, "X_num_train.npy"),
        "X_seq_train": os.path.join(cache_dir, "X_seq_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_num_val": os.path.join(cache_dir, "X_num_val.npy"),
        "X_seq_val": os.path.join(cache_dir, "X_seq_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_num_test": os.path.join(cache_dir, "X_num_test.npy"),
        "X_seq_test": os.path.join(cache_dir, "X_seq_test.npy"),
        "ids_test": os.path.join(cache_dir, "ids_test.npy"),
        "vocab_size": os.path.join(cache_dir, "vocab_size.npy"),
    }

    # Attempt to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in files.values()):
            print("Loading preprocessed data from cache...")
            data = {k: np.load(v) for k, v in files.items() if k != "vocab_size"}
            # Load scalar value separately
            vocab_size = np.load(files["vocab_size"])[0]
            return data, int(vocab_size)
        else:
            print("Cache missing or incomplete. Reprocessing data...")

    # Load raw metadata
    print("Loading raw data from metadata...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Feature Engineering: Unique Characters Count
    print("Performing feature engineering...")
    for df in [train_df, val_df, test_df]:
        df["unique_characters"] = df[Config.SEQ_COL].apply(lambda x: len(set(str(x))))

    # Identify Numerical Columns
    # Exclude ID, Target, Source Path, and the Sequence Column itself
    exclude_cols = [Config.ID_COL, Config.TARGET_COL, "source_path", Config.SEQ_COL]
    # Dynamically select numeric columns (this will include 'unique_characters' and original 'f_00'...'f_30')
    num_cols = [
        c
        for c in train_df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    print(f"Identified {len(num_cols)} numerical features.")

    # Standardization
    print("Standardizing numerical features...")
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train_df[num_cols].values.astype(np.float32))
    X_num_val = scaler.transform(val_df[num_cols].values.astype(np.float32))
    X_num_test = scaler.transform(test_df[num_cols].values.astype(np.float32))

    # Tokenization
    print("Tokenizing sequence features...")
    tokenizer = CharTokenizer()
    tokenizer.fit(train_df[Config.SEQ_COL].astype(str).tolist())

    X_seq_train = tokenizer.transform(train_df[Config.SEQ_COL].astype(str).tolist())
    X_seq_val = tokenizer.transform(val_df[Config.SEQ_COL].astype(str).tolist())
    X_seq_test = tokenizer.transform(test_df[Config.SEQ_COL].astype(str).tolist())

    vocab_size = tokenizer.vocab_size

    # Extract Targets and IDs
    y_train = train_df[Config.TARGET_COL].values.astype(np.float32)
    y_val = val_df[Config.TARGET_COL].values.astype(np.float32)
    ids_test = test_df[Config.ID_COL].values

    # Save to cache
    print("Saving processed data to cache...")
    np.save(files["X_num_train"], X_num_train)
    np.save(files["X_seq_train"], X_seq_train)
    np.save(files["y_train"], y_train)
    np.save(files["X_num_val"], X_num_val)
    np.save(files["X_seq_val"], X_seq_val)
    np.save(files["y_val"], y_val)
    np.save(files["X_num_test"], X_num_test)
    np.save(files["X_seq_test"], X_seq_test)
    np.save(files["ids_test"], ids_test)
    np.save(files["vocab_size"], np.array([vocab_size]))

    data = {
        "X_num_train": X_num_train,
        "X_seq_train": X_seq_train,
        "y_train": y_train,
        "X_num_val": X_num_val,
        "X_seq_val": X_seq_val,
        "y_val": y_val,
        "X_num_test": X_num_test,
        "X_seq_test": X_seq_test,
        "ids_test": ids_test,
    }

    return data, vocab_size


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to try loading from cache.
        debug (bool): If True, uses a small subset of data.

    Returns:
        train_loader, val_loader, test_loader, vocab_size, ids_test
    """
    data, vocab_size = _process_data(load_cached_data=load_cached_data)

    # Unpack data
    X_num_train, X_seq_train, y_train = (
        data["X_num_train"],
        data["X_seq_train"],
        data["y_train"],
    )
    X_num_val, X_seq_val, y_val = data["X_num_val"], data["X_seq_val"], data["y_val"]
    X_num_test, X_seq_test, ids_test = (
        data["X_num_test"],
        data["X_seq_test"],
        data["ids_test"],
    )

    # Debug mode: slice data
    if debug:
        print("DEBUG MODE: Using subset of 1000 samples.")
        limit = 1000
        X_num_train, X_seq_train, y_train = (
            X_num_train[:limit],
            X_seq_train[:limit],
            y_train[:limit],
        )
        X_num_val, X_seq_val, y_val = (
            X_num_val[:limit],
            X_seq_val[:limit],
            y_val[:limit],
        )
        X_num_test, X_seq_test = X_num_test[:limit], X_seq_test[:limit]
        ids_test = ids_test[:limit]

    # Create Datasets
    train_ds = ManufacturingDataset(X_num_train, X_seq_train, y_train)
    val_ds = ManufacturingDataset(X_num_val, X_seq_val, y_val)
    test_ds = ManufacturingDataset(X_num_test, X_seq_test, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab_size, ids_test
