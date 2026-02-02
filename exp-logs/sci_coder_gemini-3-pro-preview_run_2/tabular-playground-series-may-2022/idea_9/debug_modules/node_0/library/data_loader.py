import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Serves continuous features, sequence-encoded categorical features, and targets.
    """

    def __init__(self, cont_features, cat_sequence, targets=None):
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.cat_sequence = torch.tensor(cat_sequence, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {
            "cont_features": self.cont_features[idx],
            "cat_sequence": self.cat_sequence[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def preprocess_f27(series):
    """
    Converts a Series of strings (length 10) into a numpy array of shape (N, 10)
    where characters A-Z are mapped to integers 0-25.
    """
    # Convert series to list of strings, then to bytearray for fast processing
    # Assuming all strings are length 10 and contain A-Z.
    # We can use a vectorized approach by viewing the string buffer as bytes.
    # However, pandas strings are objects. A list comprehension is robust and reasonably fast for 1M rows.

    # Map 'A' (65) to 0, 'B' (66) to 1, etc.
    # We create a mapping table

    # Fast vectorized approach using list of lists
    # This takes ~1-2 seconds for 1M rows
    return np.array([[ord(c) - 65 for c in s] for s in series], dtype=np.int64)


def process_data(load_cached_data=True):
    """
    Loads raw data, aligns with metadata, preprocesses features, and caches the result.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading cached data from {Config.PROCESSED_DATA_PATH}...")
        data = np.load(Config.PROCESSED_DATA_PATH)
        return (
            data["X_train_cont"],
            data["X_train_cat"],
            data["y_train"],
            data["X_val_cont"],
            data["X_val_cat"],
            data["y_val"],
            data["X_test_cont"],
            data["X_test_cat"],
            data["test_ids"],
        )

    print("Cache not found or ignored. Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 3. Load Raw Data
    # We load the full train and test files once
    print("Loading raw CSV files...")
    df_train_full = pd.read_csv(Config.TRAIN_PATH)
    df_test_full = pd.read_csv(Config.TEST_PATH)

    # Index by ID for fast lookup
    df_train_full.set_index("id", inplace=True)
    df_test_full.set_index("id", inplace=True)

    # 4. Align Data using Metadata IDs
    print("Aligning data with metadata splits...")
    # Extract features for Train split
    train_ids = train_meta["id"].values
    df_train = df_train_full.loc[train_ids]
    y_train = train_meta["target"].values

    # Extract features for Val split
    val_ids = val_meta["id"].values
    df_val = df_train_full.loc[val_ids]
    y_val = val_meta["target"].values

    # Extract features for Test split
    test_ids = test_meta["id"].values
    df_test = df_test_full.loc[test_ids]

    # 5. Feature Engineering
    print("Preprocessing features...")

    # Identify Continuous Columns (f_00 to f_30, excluding f_27)
    # We can rely on column names or the Config
    all_cols = df_train.columns.tolist()
    cont_cols = [c for c in all_cols if c != "f_27" and c != "target" and c != "id"]
    # Ensure sorted order for consistency
    cont_cols.sort()

    # Extract Continuous Features
    X_train_cont = df_train[cont_cols].values.astype(np.float32)
    X_val_cont = df_val[cont_cols].values.astype(np.float32)
    X_test_cont = df_test[cont_cols].values.astype(np.float32)

    # Extract and Process Categorical Feature (f_27)
    X_train_cat = preprocess_f27(df_train["f_27"])
    X_val_cat = preprocess_f27(df_val["f_27"])
    X_test_cat = preprocess_f27(df_test["f_27"])

    # 6. Normalization
    print("Normalizing continuous features...")
    scaler = StandardScaler()
    # Fit ONLY on training data
    X_train_cont = scaler.fit_transform(X_train_cont)
    # Transform Val and Test
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # 7. Save to Cache
    print(f"Saving processed data to {Config.PROCESSED_DATA_PATH}...")
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
    np.savez(
        Config.PROCESSED_DATA_PATH,
        X_train_cont=X_train_cont,
        X_train_cat=X_train_cat,
        y_train=y_train,
        X_val_cont=X_val_cont,
        X_val_cat=X_val_cat,
        y_val=y_val,
        X_test_cont=X_test_cont,
        X_test_cat=X_test_cat,
        test_ids=test_ids,
    )

    return (
        X_train_cont,
        X_train_cat,
        y_train,
        X_val_cont,
        X_val_cat,
        y_val,
        X_test_cont,
        X_test_cat,
        test_ids,
    )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Load processed arrays
    (
        X_train_cont,
        X_train_cat,
        y_train,
        X_val_cont,
        X_val_cat,
        y_val,
        X_test_cont,
        X_test_cat,
        test_ids,
    ) = process_data(load_cached_data=load_cached_data)

    # Create Dataset objects
    train_dataset = ManufacturingDataset(X_train_cont, X_train_cat, y_train)
    val_dataset = ManufacturingDataset(X_val_cont, X_val_cat, y_val)
    test_dataset = ManufacturingDataset(X_test_cont, X_test_cat, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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

    print(
        f"DataLoaders ready. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
