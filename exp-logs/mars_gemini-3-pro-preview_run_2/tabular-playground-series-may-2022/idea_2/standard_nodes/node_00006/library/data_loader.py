import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    PROCESSED_DATA_PATH,
    IDEA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    seed_everything,
)


# ------------------------------------------------------------------------------
# Custom Dataset
# ------------------------------------------------------------------------------
class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Returns a dictionary containing:
        - 'cat': LongTensor of shape (10,) representing character indices of f_27.
        - 'cont': FloatTensor of shape (30,) representing standardized sensor data.
        - 'target': FloatTensor of shape (1,) (only for train/val).
    """

    def __init__(self, cat_features, cont_features, targets=None):
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        item = {"cat": self.cat_features[idx], "cont": self.cont_features[idx]}
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


# ------------------------------------------------------------------------------
# Preprocessing Logic
# ------------------------------------------------------------------------------
def _encode_f27(series):
    """
    Splits the 10-character string in f_27 into 10 separate integer features.
    Mapping: 'A' -> 0, 'B' -> 1, ...
    """
    # Convert series to list of strings
    strings = series.astype(str).tolist()
    # Convert list of strings to list of lists of integers
    # ord('A') is 65. So 'A' becomes 0, 'B' becomes 1.
    encoded = [[ord(c) - 65 for c in s] for s in strings]
    return np.array(encoded, dtype=np.int16)


def preprocess_data(load_cached_data=True):
    """
    Loads raw data, aligns it with metadata, performs feature engineering
    (scaling continuous, encoding categorical), and caches the result.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(PROCESSED_DATA_PATH):
        print(f"Loading cached data from {PROCESSED_DATA_PATH}...")
        data = np.load(PROCESSED_DATA_PATH)
        return (
            data["X_cat_train"],
            data["X_cont_train"],
            data["y_train"],
            data["X_cat_val"],
            data["X_cont_val"],
            data["y_val"],
            data["X_cat_test"],
            data["X_cont_test"],
            data["test_ids"],
        )

    print("Preprocessing data from scratch...")

    # 2. Load Metadata
    train_meta = pd.read_csv(TRAIN_META_PATH)
    val_meta = pd.read_csv(VAL_META_PATH)
    test_meta = pd.read_csv(TEST_META_PATH)

    # 3. Load Raw Data
    # We index by 'id' to facilitate fast lookup/alignment with metadata
    raw_train = pd.read_csv(TRAIN_DATA_PATH).set_index("id")
    raw_test = pd.read_csv(TEST_DATA_PATH).set_index("id")

    # 4. Align Data with Metadata
    # Using .loc ensures we get exactly the rows defined in metadata in the correct order
    df_train = raw_train.loc[train_meta["id"]]
    df_val = raw_train.loc[val_meta["id"]]
    df_test = raw_test.loc[test_meta["id"]]

    # 5. Extract Targets
    y_train = df_train["target"].values
    y_val = df_val["target"].values
    test_ids = test_meta["id"].values

    # 6. Feature Selection
    # Continuous features are f_00 to f_30, excluding f_27
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    cat_col = "f_27"

    # 7. Process Continuous Features (StandardScaler)
    scaler = StandardScaler()

    # Fit only on training data
    X_cont_train = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    # Transform val and test
    X_cont_val = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    X_cont_test = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # 8. Process Categorical Feature (f_27)
    X_cat_train = _encode_f27(df_train[cat_col])
    X_cat_val = _encode_f27(df_val[cat_col])
    X_cat_test = _encode_f27(df_test[cat_col])

    # 9. Cache Results
    os.makedirs(IDEA_DIR, exist_ok=True)
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        X_cat_train=X_cat_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_cat_val=X_cat_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_cat_test=X_cat_test,
        X_cont_test=X_cont_test,
        test_ids=test_ids,
    )
    print(f"Data processed and saved to {PROCESSED_DATA_PATH}")

    return (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    )


# ------------------------------------------------------------------------------
# DataLoader Factory
# ------------------------------------------------------------------------------
def get_dataloaders(load_cached_data=True):
    """
    Orchestrates data loading and returns PyTorch DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything()

    # Get processed numpy arrays
    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    ) = preprocess_data(load_cached_data=load_cached_data)

    # Create Datasets
    train_dataset = ManufacturingDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = ManufacturingDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = ManufacturingDataset(X_cat_test, X_cont_test, targets=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for stability in training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
