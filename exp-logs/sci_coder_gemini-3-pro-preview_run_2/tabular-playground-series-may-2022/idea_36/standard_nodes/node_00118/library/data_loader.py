import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import set_seed


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing data.
    Handles dual-stream input: continuous features and integer-encoded sequences.
    """

    def __init__(self, continuous_data, sequence_data, targets=None):
        """
        Args:
            continuous_data (np.ndarray): Normalized continuous features (N, 30).
            sequence_data (np.ndarray): Integer-encoded sequence features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
        self.sequence_data = torch.tensor(sequence_data, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        else:
            self.targets = None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        sample = {
            "continuous": self.continuous_data[idx],
            "sequence": self.sequence_data[idx],
        }
        if self.targets is not None:
            sample["target"] = self.targets[idx]
        return sample


def _process_sequence_feature(series):
    """
    Converts a pandas Series of strings (length 10) into a numpy array of integers.
    Mapping: 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.
    """
    # Convert series to list of strings
    strings = series.values.astype(str)

    # Create a buffer for the result
    n_samples = len(strings)
    seq_len = Config.SEQUENCE_LENGTH

    # Vectorized approach using view (assuming fixed length ascii/utf-8)
    # However, a safer robust way for arbitrary environments is list comprehension or frombuffer
    # Given the constraints and simplicity:

    # Map characters to 1-26
    # We can perform this by iterating or using pandas apply.
    # Since dataset size is ~1M, a compiled list comprehension is fast enough.

    # Ord('A') is 65. We want 'A' -> 1. So ord(c) - 64.
    processed = np.array([[ord(c) - 64 for c in s] for s in strings], dtype=np.int64)

    return processed


def load_and_preprocess_data(load_cached_data=True, batch_size=Config.BATCH_SIZE):
    """
    Loads data, processes features (scaling, tokenization), and returns DataLoaders.
    Implements caching using .npz files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        train_loader, val_loader, test_loader (DataLoader)
    """
    set_seed(Config.SEED)

    cache_file = Config.CACHE_PATH

    # --------------------------------------------------------------------------
    # 1. Try Loading from Cache
    # --------------------------------------------------------------------------
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading processed data from cache: {cache_file}")
        data = np.load(cache_file)

        X_cont_train = data["X_cont_train"]
        X_seq_train = data["X_seq_train"]
        y_train = data["y_train"]

        X_cont_val = data["X_cont_val"]
        X_seq_val = data["X_seq_val"]
        y_val = data["y_val"]

        X_cont_test = data["X_cont_test"]
        X_seq_test = data["X_seq_test"]
        # Test IDs might be needed for submission construction outside,
        # but usually the loader just yields predictions.
        # The submission script will read the sample submission or test file for IDs.

    else:
        print("Processing data from scratch...")

        # ----------------------------------------------------------------------
        # 2. Load Raw Data and Metadata
        # ----------------------------------------------------------------------
        # Load Raw Data
        df_train_raw = pd.read_csv(Config.TRAIN_RAW_PATH)
        df_test_raw = pd.read_csv(Config.TEST_RAW_PATH)

        # Index by ID for fast lookup
        df_train_raw.set_index("id", inplace=True)
        df_test_raw.set_index("id", inplace=True)

        # Load Metadata
        meta_train = pd.read_csv(Config.TRAIN_META_PATH)
        meta_val = pd.read_csv(Config.VAL_META_PATH)
        meta_test = pd.read_csv(Config.TEST_META_PATH)

        # ----------------------------------------------------------------------
        # 3. Split and Align Data
        # ----------------------------------------------------------------------
        # Select rows based on metadata IDs
        df_train = df_train_raw.loc[meta_train["id"]].copy()
        df_val = df_train_raw.loc[meta_val["id"]].copy()
        df_test = df_test_raw.loc[meta_test["id"]].copy()

        # Extract Targets
        y_train = df_train["target"].values.astype(np.float32)
        y_val = df_val["target"].values.astype(np.float32)

        # ----------------------------------------------------------------------
        # 4. Feature Engineering
        # ----------------------------------------------------------------------
        # Identify feature columns
        # Continuous: f_00 to f_30 excluding f_27
        cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
        seq_col = "f_27"

        # A. Process Sequence Data
        print("Tokenizing sequence features...")
        X_seq_train = _process_sequence_feature(df_train[seq_col])
        X_seq_val = _process_sequence_feature(df_val[seq_col])
        X_seq_test = _process_sequence_feature(df_test[seq_col])

        # B. Process Continuous Data
        print("Scaling continuous features...")
        scaler = StandardScaler()

        # Fit ONLY on training data
        X_cont_train = scaler.fit_transform(df_train[cont_cols].values)

        # Transform Val and Test
        X_cont_val = scaler.transform(df_val[cont_cols].values)
        X_cont_test = scaler.transform(df_test[cont_cols].values)

        # ----------------------------------------------------------------------
        # 5. Save to Cache
        # ----------------------------------------------------------------------
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(
            cache_file,
            X_cont_train=X_cont_train,
            X_seq_train=X_seq_train,
            y_train=y_train,
            X_cont_val=X_cont_val,
            X_seq_val=X_seq_val,
            y_val=y_val,
            X_cont_test=X_cont_test,
            X_seq_test=X_seq_test,
        )
        print(f"Data processed and saved to {cache_file}")

    # --------------------------------------------------------------------------
    # 6. Create DataLoaders
    # --------------------------------------------------------------------------
    train_dataset = ManufacturingDataset(X_cont_train, X_seq_train, y_train)
    val_dataset = ManufacturingDataset(X_cont_val, X_seq_val, y_val)
    test_dataset = ManufacturingDataset(X_cont_test, X_seq_test, targets=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
