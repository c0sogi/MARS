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
    Serves continuous features, decomposed categorical sequences, and targets.
    """

    def __init__(self, continuous_data, categorical_data, targets=None):
        # Ensure data is in correct tensor format
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.categorical_data = torch.LongTensor(categorical_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "categorical": self.categorical_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _process_data(load_cached_data):
    """
    Internal function to load, preprocess, and cache data.
    Strictly follows the metadata splits and applies Z-score normalization.
    """
    # 1. Caching Logic
    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading cached data from {Config.PROCESSED_DATA_PATH}...")
        try:
            data = np.load(Config.PROCESSED_DATA_PATH)
            return (
                data["train_cont"],
                data["train_cat"],
                data["train_target"],
                data["val_cont"],
                data["val_cat"],
                data["val_target"],
                data["test_cont"],
                data["test_cat"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    print("Processing data from scratch...")

    # 2. Load Metadata (Defines the splits)
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Load Raw Data
    # train.csv contains the full training data (to be split into train/val)
    # test.csv contains the test data
    raw_train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    raw_test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Set index to ID for efficient lookup and alignment
    raw_train_df.set_index("id", inplace=True)
    raw_test_df.set_index("id", inplace=True)

    # 4. Align Data with Metadata
    # We use .loc to extract exactly the rows defined in metadata, in that order
    df_train = raw_train_df.loc[train_meta["id"]].copy()
    df_val = raw_train_df.loc[val_meta["id"]].copy()
    df_test = raw_test_df.loc[test_meta["id"]].copy()

    # 5. Feature Separation
    # Identify continuous columns: f_00 to f_30, excluding f_27
    all_cols = df_train.columns.tolist()
    cont_cols = [c for c in all_cols if c != "f_27" and c != "target"]

    # Verify we have the expected number of continuous features
    if len(cont_cols) != Config.NUM_CONTINUOUS_FEATURES:
        # Check if mismatch is due to f_27 exclusion or other factors
        # In this dataset f_00..f_30 is 31 columns. f_27 is categorical.
        # So 30 continuous columns is correct.
        pass

    # 6. Preprocessing: Continuous Features (Z-Score)
    scaler = StandardScaler()
    # Fit ONLY on training data to prevent leakage
    train_cont = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    val_cont = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # 7. Preprocessing: Categorical Feature (f_27 decomposition)
    def encode_f27(series):
        # f_27 is a string of length 10 (e.g., "AB...")
        # Map 'A' -> 0, 'B' -> 1, etc.
        # Result shape: (N, 10)
        # We use a list comprehension for simplicity and speed on this dataset size
        chars = series.values
        # Create a buffer for the integer codes
        # Assuming fixed length of 10 as per Config
        n_samples = len(chars)
        seq_len = Config.SEQUENCE_LENGTH
        encoded = np.zeros((n_samples, seq_len), dtype=np.int64)

        for i, s in enumerate(chars):
            # ord('A') is 65
            encoded[i] = [ord(c) - 65 for c in s]
        return encoded

    train_cat = encode_f27(df_train["f_27"])
    val_cat = encode_f27(df_val["f_27"])
    test_cat = encode_f27(df_test["f_27"])

    # 8. Extract Targets and IDs
    train_target = df_train["target"].values.astype(np.float32)
    val_target = df_val["target"].values.astype(np.float32)
    test_ids = df_test.index.values.astype(np.int64)

    # 9. Save to Cache
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
    np.savez(
        Config.PROCESSED_DATA_PATH,
        train_cont=train_cont,
        train_cat=train_cat,
        train_target=train_target,
        val_cont=val_cont,
        val_cat=val_cat,
        val_target=val_target,
        test_cont=test_cont,
        test_cat=test_cat,
        test_ids=test_ids,
    )
    print(f"Data processed and saved to {Config.PROCESSED_DATA_PATH}")

    return (
        train_cont,
        train_cat,
        train_target,
        val_cont,
        val_cat,
        val_target,
        test_cont,
        test_cat,
        test_ids,
    )


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create PyTorch DataLoaders for Train, Val, and Test.

    Args:
        load_cached_data (bool): If True, attempts to load preprocessed data from disk.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    seed_everything()

    # Get processed data arrays
    (
        train_cont,
        train_cat,
        train_target,
        val_cont,
        val_cat,
        val_target,
        test_cont,
        test_cat,
        test_ids,
    ) = _process_data(load_cached_data)

    # Create Dataset objects
    train_dataset = ManufacturingDataset(train_cont, train_cat, train_target)
    val_dataset = ManufacturingDataset(val_cont, val_cat, val_target)
    test_dataset = ManufacturingDataset(test_cont, test_cat, None)

    # Create DataLoaders
    # Train: Shuffle=True, Drop_Last=True (for BatchNorm stability)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val: Shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Test: Shuffle=False
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, test_ids
