import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        """
        PyTorch Dataset for the Manufacturing task.

        Args:
            X_num (np.ndarray): Standardized numerical features.
            X_cat (np.ndarray): Tokenized categorical features (f_27).
            y (np.ndarray, optional): Target labels.
        """
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]


def process_data(load_cached_data=True):
    """
    Processes raw data into training, validation, and test sets with caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: Contains processed arrays (X_num_train, X_cat_train, y_train,
               X_num_val, X_cat_val, y_val, X_num_test, X_cat_test, test_ids, vocab_size)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return (
                data["X_num_train"],
                data["X_cat_train"],
                data["y_train"],
                data["X_num_val"],
                data["X_cat_val"],
                data["y_val"],
                data["X_num_test"],
                data["X_cat_test"],
                data["test_ids"],
                int(data["vocab_size"]),
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # 3. Load Raw Data
    # We load the full files and index them by 'id' for efficient lookup
    df_train_full = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"))
    df_test_full = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))

    df_train_full.set_index("id", inplace=True)
    df_test_full.set_index("id", inplace=True)

    # 4. Align Data with Metadata
    # Select rows corresponding to the metadata IDs
    df_train = df_train_full.loc[train_meta["id"]].copy()
    y_train = train_meta["target"].values.astype(np.float32)

    df_val = df_train_full.loc[val_meta["id"]].copy()
    y_val = val_meta["target"].values.astype(np.float32)

    df_test = df_test_full.loc[test_meta["id"]].copy()
    test_ids = test_meta["id"].values

    # 5. Feature Engineering

    # A. Categorical Feature (f_27) Tokenization
    # Build vocabulary from training set only to prevent leakage
    all_chars = sorted(list(set("".join(df_train["f_27"].unique()))))
    char_map = {
        c: i + 1 for i, c in enumerate(all_chars)
    }  # 1-based index, 0 is padding
    vocab_size = len(char_map) + 1

    def encode_f27(series):
        # Map characters to integers. Unknown chars (if any) map to 0.
        return np.array(
            [[char_map.get(c, 0) for c in s] for s in series], dtype=np.int64
        )

    X_cat_train = encode_f27(df_train["f_27"])
    X_cat_val = encode_f27(df_val["f_27"])
    X_cat_test = encode_f27(df_test["f_27"])

    # B. Numerical Features (f_00 - f_30, excluding f_27)
    # Identify numerical columns (exclude target and f_27)
    num_cols = [c for c in df_train.columns if c != "f_27" and c != "target"]

    scaler = StandardScaler()
    # Fit on training data only
    X_num_train = scaler.fit_transform(df_train[num_cols]).astype(np.float32)
    # Transform validation and test
    X_num_val = scaler.transform(df_val[num_cols]).astype(np.float32)
    X_num_test = scaler.transform(df_test[num_cols]).astype(np.float32)

    # 6. Cache Results
    np.savez(
        cache_path,
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_num_val=X_num_val,
        X_cat_val=X_cat_val,
        y_val=y_val,
        X_num_test=X_num_test,
        X_cat_test=X_cat_test,
        test_ids=test_ids,
        vocab_size=vocab_size,
    )
    print(f"Data processed and saved to {cache_path}")

    return (
        X_num_train,
        X_cat_train,
        y_train,
        X_num_val,
        X_cat_val,
        y_val,
        X_num_test,
        X_cat_test,
        test_ids,
        vocab_size,
    )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, vocab_size)
    """
    # Get processed data
    data = process_data(load_cached_data=load_cached_data)
    (
        X_num_train,
        X_cat_train,
        y_train,
        X_num_val,
        X_cat_val,
        y_val,
        X_num_test,
        X_cat_test,
        _,
        vocab_size,
    ) = data

    # Create Datasets
    train_dataset = ManufacturingDataset(X_num_train, X_cat_train, y_train)
    val_dataset = ManufacturingDataset(X_num_val, X_cat_val, y_val)
    test_dataset = ManufacturingDataset(X_num_test, X_cat_test)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, vocab_size
