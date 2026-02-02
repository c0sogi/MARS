import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class Tokenizer:
    """
    Tokenizer for feature f_27.
    Maps uppercase English letters (A-Z) to integers (0-25).
    """

    def transform(self, series: pd.Series) -> np.ndarray:
        """
        Transforms a pandas Series of strings into a numpy array of integers.
        Assumes all strings are length 10 and contain only A-Z.
        """
        # Convert series to list of strings
        strings = series.values.tolist()

        # Vectorized conversion using list comprehension and ASCII mapping
        # ord('A') is 65. So 'A' -> 0, 'B' -> 1, etc.
        tokenized = [[ord(char) - 65 for char in s] for s in strings]

        return np.array(tokenized, dtype=np.int64)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    """

    def __init__(self, x_cat, x_cont, y=None):
        """
        Args:
            x_cat (np.ndarray or torch.Tensor): Categorical features (f_27 tokenized).
            x_cont (np.ndarray or torch.Tensor): Continuous features (normalized).
            y (np.ndarray or torch.Tensor, optional): Target labels.
        """
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.as_tensor(x_cont, dtype=torch.float32)
        self.y = (
            torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
            if y is not None
            else None
        )

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        else:
            return self.x_cat[idx], self.x_cont[idx]


def process_data(load_cached_data=True, debug=False):
    """
    Loads, processes, and caches data.

    Logic:
    1. If cached .npz exists and load_cached_data is True, load and return.
    2. Else, load raw CSVs and Metadata.
    3. Merge to align with stratified splits.
    4. Tokenize f_27.
    5. Standardize numerical features (fit on train, transform all).
    6. Save to .npz.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path)
        return (
            data["train_cat"],
            data["train_cont"],
            data["train_target"],
            data["val_cat"],
            data["val_cont"],
            data["val_target"],
            data["test_cat"],
            data["test_cont"],
        )

    print("Cache not found or reload requested. Processing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Debugging: Subsample if requested
    if debug:
        print(f"Debug mode: Subsampling {Config.DEBUG_SAMPLES} samples.")
        train_meta = train_meta.iloc[: Config.DEBUG_SAMPLES]
        val_meta = val_meta.iloc[: Config.DEBUG_SAMPLES]
        test_meta = test_meta.iloc[: Config.DEBUG_SAMPLES]

    # 3. Load Raw Data
    # We load the full train and test files once
    df_train_raw = pd.read_csv(Config.TRAIN_PATH)
    df_test_raw = pd.read_csv(Config.TEST_PATH)

    # 4. Merge Metadata with Raw Data
    # This ensures we get the exact rows corresponding to the stratified split
    # and preserves the order defined in metadata.
    train_df = train_meta.merge(df_train_raw, on="id", how="left")
    val_df = val_meta.merge(df_train_raw, on="id", how="left")
    test_df = test_meta.merge(df_test_raw, on="id", how="left")

    # 5. Feature Selection
    # Identify numerical columns: f_00 to f_30, excluding f_27
    # f_27 is the categorical string column
    all_cols = [f"f_{i:02d}" for i in range(31)]
    cat_col = "f_27"
    num_cols = [c for c in all_cols if c != cat_col]

    # 6. Processing
    tokenizer = Tokenizer()
    scaler = StandardScaler()

    # A. Tokenize Categorical (Stream 1)
    train_cat = tokenizer.transform(train_df[cat_col])
    val_cat = tokenizer.transform(val_df[cat_col])
    test_cat = tokenizer.transform(test_df[cat_col])

    # B. Normalize Continuous (Stream 2)
    # Fit scaler ONLY on training data
    train_cont = scaler.fit_transform(train_df[num_cols].values)
    val_cont = scaler.transform(val_df[num_cols].values)
    test_cont = scaler.transform(test_df[num_cols].values)

    # C. Targets
    train_target = train_df["target"].values.astype(np.float32)
    val_target = val_df["target"].values.astype(np.float32)
    # Test set has no target

    # 7. Save to Cache
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_cat=train_cat,
        train_cont=train_cont,
        train_target=train_target,
        val_cat=val_cat,
        val_cont=val_cont,
        val_target=val_target,
        test_cat=test_cat,
        test_cont=test_cont,
    )
    print(f"Data processed and saved to {cache_path}")

    return (
        train_cat,
        train_cont,
        train_target,
        val_cat,
        val_cont,
        val_target,
        test_cat,
        test_cont,
    )


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Returns DataLoaders for train, validation, and test sets.
    """
    # Get processed numpy arrays
    (
        train_cat,
        train_cont,
        train_target,
        val_cat,
        val_cont,
        val_target,
        test_cat,
        test_cont,
    ) = process_data(load_cached_data=load_cached_data, debug=debug)

    # Create Datasets
    train_dataset = ManufacturingDataset(train_cat, train_cont, train_target)
    val_dataset = ManufacturingDataset(val_cat, val_cont, val_target)
    test_dataset = ManufacturingDataset(test_cat, test_cont, y=None)

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

    return train_loader, val_loader, test_loader
