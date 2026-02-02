import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything()


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Serves continuous features, tokenized sequences, and targets.
    """

    def __init__(self, continuous_data, sequence_data, targets=None, ids=None):
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.sequence_data = torch.LongTensor(sequence_data)

        if targets is not None:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None

        if ids is not None:
            self.ids = ids
        else:
            self.ids = None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "sequence": self.sequence_data[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def _tokenize_f27(series):
    """
    Decomposes the f_27 string column into a numpy array of integer tokens.
    Maps 'A'->0, 'B'->1, ..., 'Z'->25.
    """
    # Convert series to list of strings
    strings = series.values
    # Create a buffer for the result: (N, 10)
    # We assume fixed length of 10 based on Config.CHAR_SEQ_LEN
    n_samples = len(strings)
    seq_len = Config.CHAR_SEQ_LEN

    # Efficient vectorization using list comprehension and ord
    # This is faster than applying a function row-wise for large datasets
    # ord('A') is 65
    tokenized = np.array(
        [[ord(c) - 65 for c in s[:seq_len]] for s in strings], dtype=np.int32
    )

    return tokenized


def preprocess_data(load_cached_data=True):
    """
    Loads raw data, performs feature engineering, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: Dictionary containing numpy arrays for train/val/test splits.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            # Reconstruct dictionary
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # 3. Load Raw Data
    # We load the full train and test files once
    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_test_full = pd.read_csv(Config.TEST_CSV)

    # 4. Merge to align with Metadata splits
    # This ensures we get exactly the rows defined in metadata in the correct order
    df_train = train_meta.merge(df_train_full, on="id", how="left")
    df_val = val_meta.merge(df_train_full, on="id", how="left")
    df_test = test_meta.merge(df_test_full, on="id", how="left")

    # Verify merge integrity (target in metadata should match target in raw, though raw is authoritative)
    # We drop the 'target_x' from metadata and keep 'target_y' from raw or vice versa.
    # Metadata has 'target', raw has 'target'. Merge creates target_x, target_y.
    # We'll use the target from the dataframe logic.

    # Clean up merge columns if necessary.
    # train_meta has 'target', df_train_full has 'target'.
    # Result has target_x (meta) and target_y (raw). They should be identical.
    if "target_y" in df_train.columns:
        df_train["target"] = df_train["target_y"]
        df_val["target"] = df_val["target_y"]

    # 5. Feature Selection
    # Continuous features: f_00 to f_30, excluding f_27
    feature_cols = [f"f_{i:02d}" for i in range(31)]
    cont_cols = [c for c in feature_cols if c != "f_27"]

    # 6. Normalization (Z-Score)
    # Fit ONLY on Training set
    train_cont = df_train[cont_cols].values.astype(np.float32)
    val_cont = df_val[cont_cols].values.astype(np.float32)
    test_cont = df_test[cont_cols].values.astype(np.float32)

    mean = np.mean(train_cont, axis=0)
    std = np.std(train_cont, axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    train_cont = (train_cont - mean) / std
    val_cont = (val_cont - mean) / std
    test_cont = (test_cont - mean) / std

    # 7. Sequence Tokenization (f_27)
    train_seq = _tokenize_f27(df_train["f_27"])
    val_seq = _tokenize_f27(df_val["f_27"])
    test_seq = _tokenize_f27(df_test["f_27"])

    # 8. Targets and IDs
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)

    ids_train = df_train["id"].values.astype(np.int64)
    ids_val = df_val["id"].values.astype(np.int64)
    ids_test = df_test["id"].values.astype(np.int64)

    # 9. Save to Cache
    data_dict = {
        "train_cont": train_cont,
        "train_seq": train_seq,
        "y_train": y_train,
        "ids_train": ids_train,
        "val_cont": val_cont,
        "val_seq": val_seq,
        "y_val": y_val,
        "ids_val": ids_val,
        "test_cont": test_cont,
        "test_seq": test_seq,
        "ids_test": ids_test,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **data_dict)
    print(f"Data processed and saved to {cache_path}")

    return data_dict


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data = preprocess_data(load_cached_data=load_cached_data)

    # Extract arrays
    train_cont = data["train_cont"]
    train_seq = data["train_seq"]
    y_train = data["y_train"]
    ids_train = data["ids_train"]

    val_cont = data["val_cont"]
    val_seq = data["val_seq"]
    y_val = data["y_val"]
    ids_val = data["ids_val"]

    test_cont = data["test_cont"]
    test_seq = data["test_seq"]
    ids_test = data["ids_test"]

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating data to {Config.DEBUG_SAMPLES} samples.")
        limit = Config.DEBUG_SAMPLES
        train_cont = train_cont[:limit]
        train_seq = train_seq[:limit]
        y_train = y_train[:limit]
        ids_train = ids_train[:limit]

        val_cont = val_cont[:limit]
        val_seq = val_seq[:limit]
        y_val = y_val[:limit]
        ids_val = ids_val[:limit]

        test_cont = test_cont[:limit]
        test_seq = test_seq[:limit]
        ids_test = ids_test[:limit]

    # Create Datasets
    train_dataset = ManufacturingDataset(
        train_cont, train_seq, targets=y_train, ids=ids_train
    )
    val_dataset = ManufacturingDataset(val_cont, val_seq, targets=y_val, ids=ids_val)
    test_dataset = ManufacturingDataset(test_cont, test_seq, targets=None, ids=ids_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader
