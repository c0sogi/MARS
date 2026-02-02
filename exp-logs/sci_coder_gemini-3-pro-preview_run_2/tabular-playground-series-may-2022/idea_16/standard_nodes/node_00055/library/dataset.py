import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Handles continuous features, tokenized categorical sequences, and binary targets.
    """

    def __init__(self, cont_features, cat_features, targets=None):
        """
        Args:
            cont_features (np.ndarray): Normalized continuous features (N, 30).
            cat_features (np.ndarray): Tokenized string features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.cont_features = torch.FloatTensor(cont_features)
        self.cat_features = torch.LongTensor(cat_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        item = {
            "cont_features": self.cont_features[idx],
            "cat_features": self.cat_features[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (tokenization, normalization),
    and caches the result to disk. Strictly follows metadata splits.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: Dictionary containing numpy arrays for train/val/test splits.
    """
    processed_path = Config.PROCESSED_DATA

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(processed_path):
        print(f"Loading cached data from {processed_path}...")
        try:
            data = np.load(processed_path)
            return {
                "train_cont": data["train_cont"],
                "train_cat": data["train_cat"],
                "train_y": data["train_y"],
                "val_cont": data["val_cont"],
                "val_cat": data["val_cat"],
                "val_y": data["val_y"],
                "test_cont": data["test_cont"],
                "test_cat": data["test_cat"],
                "test_ids": data["test_ids"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load Metadata and Raw Data
    print("Reading raw data and metadata...")
    train_meta = pd.read_csv(Config.TRAIN_META)
    val_meta = pd.read_csv(Config.VAL_META)
    test_meta = pd.read_csv(Config.TEST_META)

    # Load raw CSVs
    # Note: train.csv contains both train and val samples as per original file structure
    raw_train_full = pd.read_csv(Config.TRAIN_CSV)
    raw_test = pd.read_csv(Config.TEST_CSV)

    # 3. Merge to create splits based on Metadata
    # We use inner join on 'id' to filter and sort exactly as metadata specifies
    print("Splitting data based on metadata...")
    train_df = train_meta.merge(raw_train_full, on="id", suffixes=("_meta", ""))
    val_df = val_meta.merge(raw_train_full, on="id", suffixes=("_meta", ""))
    test_df = test_meta.merge(raw_test, on="id", suffixes=("_meta", ""))

    # Handle potential duplicate target columns from merge if metadata had target
    if "target_meta" in train_df.columns:
        train_df = train_df.drop(columns=["target_meta"])
    if "target_meta" in val_df.columns:
        val_df = val_df.drop(columns=["target_meta"])

    # 4. Feature Engineering
    print("Preprocessing features...")

    # Identify columns
    # Continuous features are f_00 to f_30, excluding f_27
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    cat_col = "f_27"

    # 4.1 Continuous Normalization
    scaler = StandardScaler()
    # Fit only on TRAIN
    scaler.fit(train_df[cont_cols].values)

    train_cont = scaler.transform(train_df[cont_cols].values).astype(np.float32)
    val_cont = scaler.transform(val_df[cont_cols].values).astype(np.float32)
    test_cont = scaler.transform(test_df[cont_cols].values).astype(np.float32)

    # 4.2 Categorical Tokenization (f_27)
    # Map A-Z to 1-26. 0 is reserved (though not used here as length is fixed).
    def tokenize_string(series):
        # Convert series of strings to list of lists of ascii values shifted
        # 'A' is 65. We want 'A' -> 1. So ord(c) - 64.
        return np.array([[ord(c) - 64 for c in s] for s in series], dtype=np.int64)

    train_cat = tokenize_string(train_df[cat_col])
    val_cat = tokenize_string(val_df[cat_col])
    test_cat = tokenize_string(test_df[cat_col])

    # 4.3 Targets and IDs
    train_y = train_df["target"].values.astype(np.float32)
    val_y = val_df["target"].values.astype(np.float32)
    test_ids = test_df["id"].values.astype(np.int64)

    # 5. Cache Results
    print(f"Saving processed data to {processed_path}...")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    np.savez_compressed(
        processed_path,
        train_cont=train_cont,
        train_cat=train_cat,
        train_y=train_y,
        val_cont=val_cont,
        val_cat=val_cat,
        val_y=val_y,
        test_cont=test_cont,
        test_cat=test_cat,
        test_ids=test_ids,
    )

    return {
        "train_cont": train_cont,
        "train_cat": train_cat,
        "train_y": train_y,
        "val_cont": val_cont,
        "val_cat": val_cat,
        "val_y": val_y,
        "test_cont": test_cont,
        "test_cat": test_cat,
        "test_ids": test_ids,
    }


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    seed_everything(Config.SEED)

    data = process_and_cache_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ManufacturingDataset(
        data["train_cont"], data["train_cat"], data["train_y"]
    )

    val_dataset = ManufacturingDataset(data["val_cont"], data["val_cat"], data["val_y"])

    test_dataset = ManufacturingDataset(
        data["test_cont"], data["test_cat"], targets=None
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, data["test_ids"]
