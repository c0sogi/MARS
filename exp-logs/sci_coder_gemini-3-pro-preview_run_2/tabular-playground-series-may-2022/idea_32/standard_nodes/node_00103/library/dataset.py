import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, continuous_data, sequence_data, targets=None):
        """
        Args:
            continuous_data (np.ndarray): Normalized continuous features (N, 30).
            sequence_data (np.ndarray): Integer-encoded sequence features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
        self.sequence_data = torch.tensor(sequence_data, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "sequence": self.sequence_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _tokenize_f27(series):
    """
    Converts a pandas Series of 10-character strings into a (N, 10) numpy array of integers.
    Mapping: 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.
    """
    # Convert series to list of strings
    strings = series.tolist()
    # Vectorized conversion using list comprehension
    # ord('A') is 65. We want A=1. So ord(c) - 64.
    tokenized = [[ord(c) - 64 for c in s] for s in strings]
    return np.array(tokenized, dtype=np.int32)


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, merges with metadata, preprocesses (scaling + tokenization),
    and caches the result.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(Config.CACHE_FILE_PATH):
        try:
            data = np.load(Config.CACHE_FILE_PATH)
            return (
                data["X_cont_train"],
                data["X_seq_train"],
                data["y_train"],
                data["X_cont_val"],
                data["X_seq_val"],
                data["y_val"],
                data["X_cont_test"],
                data["X_seq_test"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Load Raw Data
    # We read the full files once
    raw_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    raw_test = pd.read_csv(Config.TEST_DATA_PATH)

    # 3. Merge to create splits strictly based on metadata
    # This ensures order and correct stratification
    # Cite debug_lesson_1: Drop 'target' from raw data to prevent column collision with metadata
    if "target" in raw_train.columns:
        raw_train = raw_train.drop(columns=["target"])

    df_train = train_meta.merge(raw_train, on="id", how="left")
    df_val = val_meta.merge(raw_train, on="id", how="left")
    df_test = test_meta.merge(raw_test, on="id", how="left")

    # 4. Feature Extraction
    # Continuous columns: f_00 to f_30 (excluding f_27 which is categorical)
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    X_cont_train_raw = df_train[cont_cols].values
    X_cont_val_raw = df_val[cont_cols].values
    X_cont_test_raw = df_test[cont_cols].values

    # 5. Normalization
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_cont_train = scaler.fit_transform(X_cont_train_raw)
    X_cont_val = scaler.transform(X_cont_val_raw)
    X_cont_test = scaler.transform(X_cont_test_raw)

    # 6. Tokenization of f_27
    X_seq_train = _tokenize_f27(df_train["f_27"])
    X_seq_val = _tokenize_f27(df_val["f_27"])
    X_seq_test = _tokenize_f27(df_test["f_27"])

    # 7. Targets and IDs
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # 8. Save to Cache
    np.savez_compressed(
        Config.CACHE_FILE_PATH,
        X_cont_train=X_cont_train,
        X_seq_train=X_seq_train,
        y_train=y_train,
        X_cont_val=X_cont_val,
        X_seq_val=X_seq_val,
        y_val=y_val,
        X_cont_test=X_cont_test,
        X_seq_test=X_seq_test,
        test_ids=test_ids,
    )

    return (
        X_cont_train,
        X_seq_train,
        y_train,
        X_cont_val,
        X_seq_val,
        y_val,
        X_cont_test,
        X_seq_test,
        test_ids,
    )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    Also returns the test_ids for submission creation.
    """
    # Load processed data
    (
        X_cont_train,
        X_seq_train,
        y_train,
        X_cont_val,
        X_seq_val,
        y_val,
        X_cont_test,
        X_seq_test,
        test_ids,
    ) = process_and_cache_data(load_cached_data=load_cached_data)

    # Handle Debug Mode
    if Config.DEBUG:
        limit = Config.DEBUG_SAMPLES
        X_cont_train = X_cont_train[:limit]
        X_seq_train = X_seq_train[:limit]
        y_train = y_train[:limit]

        X_cont_val = X_cont_val[:limit]
        X_seq_val = X_seq_val[:limit]
        y_val = y_val[:limit]

        X_cont_test = X_cont_test[:limit]
        X_seq_test = X_seq_test[:limit]
        test_ids = test_ids[:limit]

    # Create Datasets
    train_dataset = ManufacturingDataset(X_cont_train, X_seq_train, y_train)
    val_dataset = ManufacturingDataset(X_cont_val, X_seq_val, y_val)
    test_dataset = ManufacturingDataset(X_cont_test, X_seq_test, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
