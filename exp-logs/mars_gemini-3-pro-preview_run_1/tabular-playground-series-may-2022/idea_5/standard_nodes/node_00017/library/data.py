import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class CharTokenizer:
    """
    Character-level tokenizer for the sequence feature f_27.
    Maps uppercase letters A-Z to indices 1-26. Index 0 is reserved for padding.
    """

    def __init__(self, max_len=15):
        self.max_len = max_len
        # Create mapping A=1, B=2, ..., Z=26
        self.char_map = {chr(i + 65): i + 1 for i in range(26)}
        self.vocab_size = 27  # 0 (pad) + 26 letters

    def transform(self, sequences):
        """
        Converts a list or Series of strings into a padded integer numpy array.

        Args:
            sequences: List or Series of strings.

        Returns:
            np.ndarray: Shape (N, max_len) containing integer tokens.
        """
        batch_size = len(sequences)
        tokenized = np.zeros((batch_size, self.max_len), dtype=np.int64)

        for idx, seq in enumerate(sequences):
            # Ensure seq is a string (handle potential NaNs if any, though data analysis said none)
            if not isinstance(seq, str):
                continue

            # Truncate to max_len
            chars = list(seq)[: self.max_len]
            # Map characters to integers
            indices = [self.char_map.get(c, 0) for c in chars]
            # Fill the array (padding is already 0)
            tokenized[idx, : len(indices)] = indices

        return tokenized


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing data.
    Returns a dictionary containing numerical features, sequence tokens, and targets.
    """

    def __init__(self, numerical_data, sequence_data, targets=None, ids=None):
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
        self.sequence_data = torch.tensor(sequence_data, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )
        self.ids = ids  # IDs are kept as numpy array/list for submission generation

    def __len__(self):
        return len(self.numerical_data)

    def __getitem__(self, idx):
        item = {
            "numerical": self.numerical_data[idx],
            "sequence": self.sequence_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.ids is not None:
            item["id"] = self.ids[idx]
        return item


def process_data(load_cached_data=True):
    """
    Loads raw metadata, preprocesses features (Scaling, Tokenization),
    and caches the processed arrays to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files first.

    Returns:
        dict: Dictionary containing processed numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define paths for cached files
    files = {
        "train_num": os.path.join(cache_dir, "X_num_train.npy"),
        "train_seq": os.path.join(cache_dir, "X_seq_train.npy"),
        "train_target": os.path.join(cache_dir, "y_train.npy"),
        "val_num": os.path.join(cache_dir, "X_num_val.npy"),
        "val_seq": os.path.join(cache_dir, "X_seq_val.npy"),
        "val_target": os.path.join(cache_dir, "y_val.npy"),
        "test_num": os.path.join(cache_dir, "X_num_test.npy"),
        "test_seq": os.path.join(cache_dir, "X_seq_test.npy"),
        "test_ids": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Attempt to load from cache
    if load_cached_data and all(os.path.exists(f) for f in files.values()):
        print(f"Loading cached data from {cache_dir}...")
        data = {k: np.load(v) for k, v in files.items()}
        return data

    print("Processing data from scratch...")

    # Load Dataframes from Metadata
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Debug Sampling
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: min(len(df_val), Config.DEBUG_SAMPLE_SIZE)]
        # We usually keep test intact or sample it too if purely debugging pipeline flow
        # df_test = df_test.iloc[:Config.DEBUG_SAMPLE_SIZE]

    # 1. Numerical Feature Processing
    num_cols = Config.NUMERICAL_FEATURES

    # Extract
    X_num_train = df_train[num_cols].values.astype(np.float32)
    X_num_val = df_val[num_cols].values.astype(np.float32)
    X_num_test = df_test[num_cols].values.astype(np.float32)

    # Standardize (Fit on Train, Transform all)
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_val = scaler.transform(X_num_val)
    X_num_test = scaler.transform(X_num_test)

    # 2. Sequence Feature Processing
    tokenizer = CharTokenizer(max_len=Config.MAX_SEQ_LEN)
    seq_col = Config.SEQUENCE_FEATURE

    X_seq_train = tokenizer.transform(df_train[seq_col])
    X_seq_val = tokenizer.transform(df_val[seq_col])
    X_seq_test = tokenizer.transform(df_test[seq_col])

    # 3. Targets and IDs
    y_train = df_train[Config.TARGET_COL].values.astype(np.float32)
    y_val = df_val[Config.TARGET_COL].values.astype(np.float32)
    ids_test = df_test[Config.ID_COL].values

    # Save to Cache
    np.save(files["train_num"], X_num_train)
    np.save(files["train_seq"], X_seq_train)
    np.save(files["train_target"], y_train)
    np.save(files["val_num"], X_num_val)
    np.save(files["val_seq"], X_seq_val)
    np.save(files["val_target"], y_val)
    np.save(files["test_num"], X_num_test)
    np.save(files["test_seq"], X_seq_test)
    np.save(files["test_ids"], ids_test)

    print("Data processing complete and cached.")

    return {
        "train_num": X_num_train,
        "train_seq": X_seq_train,
        "train_target": y_train,
        "val_num": X_num_val,
        "val_seq": X_seq_val,
        "val_target": y_val,
        "test_num": X_num_test,
        "test_seq": X_seq_test,
        "test_ids": ids_test,
    }


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True, num_workers=4):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to use cached numpy files.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data = process_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ManufacturingDataset(
        data["train_num"], data["train_seq"], data["train_target"]
    )
    val_dataset = ManufacturingDataset(
        data["val_num"], data["val_seq"], data["val_target"]
    )
    test_dataset = ManufacturingDataset(
        data["test_num"], data["test_seq"], ids=data["test_ids"]
    )

    # Create DataLoaders
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
