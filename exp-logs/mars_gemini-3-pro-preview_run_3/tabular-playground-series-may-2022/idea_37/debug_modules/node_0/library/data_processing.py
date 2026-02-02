import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config, get_unique_char_count


class ManufacturingDataset(Dataset):
    def __init__(self, x_cont, x_cat, y=None):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = (
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        )

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cont[idx], self.x_cat[idx], self.y[idx]
        return self.x_cont[idx], self.x_cat[idx]


def preprocess_data(load_cached_data=True, debug=False):
    """
    Loads data, performs feature engineering, scaling, and encoding.
    Uses caching to speed up subsequent runs.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if debug:
        print("Debug mode: Using subset of data.")
        train_df = train_df.head(5000)
        val_df = val_df.head(1000)
        test_df = test_df.head(1000)

    # Store IDs for submission
    test_ids = test_df["id"].values

    # Mark splits
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Concatenate for transductive processing
    # We align vocabulary and scaling across all splits
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # Feature Engineering
    # A. Set Cardinality for f_27
    full_df["f_27_unique_count"] = full_df["f_27"].apply(get_unique_char_count)

    # B. String Decomposition for f_27 (10 positions)
    for i in range(10):
        full_df[f"f_27_char_{i}"] = full_df["f_27"].str[i]

    # Column Identification
    # Continuous: f_00 to f_28 (excluding f_27) + f_27_unique_count
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_cols.append("f_27_unique_count")

    # Categorical: f_29, f_30 + decomposed f_27 characters
    cat_cols = ["f_29", "f_30"] + [f"f_27_char_{i}" for i in range(10)]

    # Scaling
    # Fit on TRAIN only, transform ALL to prevent data leakage in scaling statistics
    scaler = StandardScaler()
    train_mask = full_df["split"] == "train"
    scaler.fit(full_df.loc[train_mask, cont_cols])
    full_df.loc[:, cont_cols] = scaler.transform(full_df.loc[:, cont_cols])

    # Encoding
    # Fit on ALL (Transductive) to ensure all categories are known
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    full_df.loc[:, cat_cols] = oe.fit_transform(full_df.loc[:, cat_cols])

    # Ensure integer type for embedding lookups
    for col in cat_cols:
        full_df[col] = full_df[col].astype(int)

    # Calculate vocab sizes for embedding layers
    vocab_sizes = [int(full_df[col].max() + 1) for col in cat_cols]

    # Split back into datasets
    train_proc = full_df[full_df["split"] == "train"]
    val_proc = full_df[full_df["split"] == "val"]
    test_proc = full_df[full_df["split"] == "test"]

    # Extract numpy arrays
    y_train = train_proc["target"].values.astype(np.float32)
    y_val = val_proc["target"].values.astype(np.float32)

    X_train_cont = train_proc[cont_cols].values.astype(np.float32)
    X_train_cat = train_proc[cat_cols].values.astype(np.int64)

    X_val_cont = val_proc[cont_cols].values.astype(np.float32)
    X_val_cat = val_proc[cat_cols].values.astype(np.int64)

    X_test_cont = test_proc[cont_cols].values.astype(np.float32)
    X_test_cat = test_proc[cat_cols].values.astype(np.int64)

    dims = {
        "n_cont": len(cont_cols),
        "n_cat": len(cat_cols),
        "vocab_sizes": vocab_sizes,
    }

    data = {
        "X_train_cont": X_train_cont,
        "X_train_cat": X_train_cat,
        "y_train": y_train,
        "X_val_cont": X_val_cont,
        "X_val_cat": X_val_cat,
        "y_val": y_val,
        "X_test_cont": X_test_cont,
        "X_test_cat": X_test_cat,
        "dims": dims,
        "ids": test_ids,
    }

    # 3. Save to Cache
    try:
        np.save(cache_file, data)
        print(f"Data saved to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return data


def get_dataloaders(data, batch_size, num_workers):
    """
    Creates PyTorch DataLoaders from the processed data dictionary.
    """
    train_ds = ManufacturingDataset(
        data["X_train_cont"], data["X_train_cat"], data["y_train"]
    )
    val_ds = ManufacturingDataset(data["X_val_cont"], data["X_val_cat"], data["y_val"])
    test_ds = ManufacturingDataset(data["X_test_cont"], data["X_test_cat"], None)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders
