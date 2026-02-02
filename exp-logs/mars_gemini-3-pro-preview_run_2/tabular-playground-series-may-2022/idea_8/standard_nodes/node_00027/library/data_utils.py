import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles numerical features, tokenized categorical features, and binary targets.
    """

    def __init__(self, numeric_data, cat_data, targets=None):
        self.numeric_data = torch.FloatTensor(numeric_data)
        self.cat_data = torch.LongTensor(cat_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.numeric_data)

    def __getitem__(self, idx):
        item = {"numeric": self.numeric_data[idx], "categorical": self.cat_data[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def decompose_f27(series):
    """
    Decomposes the 10-character string in 'f_27' into 10 integer tokens.
    Maps 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.
    """
    # Convert series to list of strings for processing
    # We assume all strings are length 10 and contain A-Z
    # ord('A') is 65. So ord(c) - 64 gives 1-based index.

    # Vectorized approach using list comprehension is generally efficient enough for <1M rows
    # and avoids complex pandas string manipulation overhead
    tokens = [[ord(c) - 64 for c in s] for s in series]
    return np.array(tokens, dtype=np.int32)


def get_data(load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.

    Args:
        load_cached_data (bool): If True, attempts to load from Config.PROCESSED_DATA_PATH.

    Returns:
        tuple: (train_num, train_cat, train_target,
                val_num, val_cat, val_target,
                test_num, test_cat, test_ids)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading processed data from {Config.PROCESSED_DATA_PATH}...")
        try:
            data = np.load(Config.PROCESSED_DATA_PATH)
            train_num = data["train_num"]
            train_cat = data["train_cat"]
            train_target = data["train_target"]
            val_num = data["val_num"]
            val_cat = data["val_cat"]
            val_target = data["val_target"]
            test_num = data["test_num"]
            test_cat = data["test_cat"]
            test_ids = data["test_ids"]

            # Handle Debug Slicing on cached data
            if Config.DEBUG:
                print(f"DEBUG mode: Slicing data to {Config.DEBUG_SAMPLES} samples.")
                train_num = train_num[: Config.DEBUG_SAMPLES]
                train_cat = train_cat[: Config.DEBUG_SAMPLES]
                train_target = train_target[: Config.DEBUG_SAMPLES]
                val_num = val_num[: Config.DEBUG_SAMPLES]
                val_cat = val_cat[: Config.DEBUG_SAMPLES]
                val_target = val_target[: Config.DEBUG_SAMPLES]
                # Keep test full or slice? Usually keep test full for submission check,
                # but for debug speed we can slice. Let's slice test too.
                test_num = test_num[: Config.DEBUG_SAMPLES]
                test_cat = test_cat[: Config.DEBUG_SAMPLES]
                test_ids = test_ids[: Config.DEBUG_SAMPLES]

            return (
                train_num,
                train_cat,
                train_target,
                val_num,
                val_cat,
                val_target,
                test_num,
                test_cat,
                test_ids,
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw Data
    # We read the full files and then merge/filter using metadata
    print("Reading raw CSV files...")
    df_train_raw = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_test_raw = pd.read_csv(Config.TEST_DATA_PATH)

    # Merge to create splits
    # We perform a left join on the metadata to preserve the specific split order and IDs
    print("Creating splits based on metadata...")
    df_train = pd.merge(train_meta[["id"]], df_train_raw, on="id", how="left")
    df_val = pd.merge(val_meta[["id"]], df_train_raw, on="id", how="left")
    df_test = pd.merge(test_meta[["id"]], df_test_raw, on="id", how="left")

    # Identify Columns
    # Numeric cols are f_00 to f_30, excluding f_27
    # We can select them dynamically
    all_cols = df_train.columns.tolist()
    numeric_cols = [c for c in all_cols if c.startswith("f_") and c != "f_27"]

    print(f"Identified {len(numeric_cols)} numerical columns.")

    # Extract Features
    print("Extracting and standardizing numerical features...")
    X_train_num = df_train[numeric_cols].values.astype(np.float32)
    X_val_num = df_val[numeric_cols].values.astype(np.float32)
    X_test_num = df_test[numeric_cols].values.astype(np.float32)

    # Standardization
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = scaler.transform(X_val_num)
    X_test_num = scaler.transform(X_test_num)

    # Process Categorical f_27
    print("Decomposing f_27...")
    X_train_cat = decompose_f27(df_train["f_27"])
    X_val_cat = decompose_f27(df_val["f_27"])
    X_test_cat = decompose_f27(df_test["f_27"])

    # Extract Targets
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # Save to Cache
    print(f"Saving processed data to {Config.PROCESSED_DATA_PATH}...")
    np.savez(
        Config.PROCESSED_DATA_PATH,
        train_num=X_train_num,
        train_cat=X_train_cat,
        train_target=y_train,
        val_num=X_val_num,
        val_cat=X_val_cat,
        val_target=y_val,
        test_num=X_test_num,
        test_cat=X_test_cat,
        test_ids=test_ids,
    )

    # Handle Debug Slicing (Post-processing)
    if Config.DEBUG:
        print(f"DEBUG mode: Slicing data to {Config.DEBUG_SAMPLES} samples.")
        X_train_num = X_train_num[: Config.DEBUG_SAMPLES]
        X_train_cat = X_train_cat[: Config.DEBUG_SAMPLES]
        y_train = y_train[: Config.DEBUG_SAMPLES]
        X_val_num = X_val_num[: Config.DEBUG_SAMPLES]
        X_val_cat = X_val_cat[: Config.DEBUG_SAMPLES]
        y_val = y_val[: Config.DEBUG_SAMPLES]
        X_test_num = X_test_num[: Config.DEBUG_SAMPLES]
        X_test_cat = X_test_cat[: Config.DEBUG_SAMPLES]
        test_ids = test_ids[: Config.DEBUG_SAMPLES]

    return (
        X_train_num,
        X_train_cat,
        y_train,
        X_val_num,
        X_val_cat,
        y_val,
        X_test_num,
        X_test_cat,
        test_ids,
    )


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Get data
    (
        train_num,
        train_cat,
        train_target,
        val_num,
        val_cat,
        val_target,
        test_num,
        test_cat,
        test_ids,
    ) = get_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ManufacturingDataset(train_num, train_cat, train_target)
    val_dataset = ManufacturingDataset(val_num, val_cat, val_target)
    test_dataset = ManufacturingDataset(test_num, test_cat, targets=None)

    # Create DataLoaders
    # Train loader needs shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # Val/Test loaders shuffle=False
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

    return train_loader, val_loader, test_loader, test_ids
