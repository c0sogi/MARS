import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    Holds preprocessed continuous features, categorical token indices, and targets.
    """

    def __init__(self, continuous_data, categorical_data, targets=None):
        """
        Args:
            continuous_data (np.ndarray): Normalized continuous features (N, 30).
            categorical_data (np.ndarray): Integer-encoded character sequences (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.categorical_data = torch.LongTensor(categorical_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        sample = {
            "continuous": self.continuous_data[idx],
            "categorical": self.categorical_data[idx],
        }
        if self.targets is not None:
            sample["target"] = self.targets[idx]
        return sample


def _process_f27(series):
    """
    Converts a Series of 10-character strings into a (N, 10) numpy array of integers.
    Maps 'A' -> 0, ..., 'Z' -> 25.
    """
    # Convert series to list of lists of ASCII values, then subtract 65 ('A')
    # This list comprehension is efficient enough for ~1M rows
    # ord('A') is 65.

    # Ensure series is string type
    series = series.astype(str)

    # Vectorized approach using list comprehension
    # We assume all strings are length 10 and contain A-Z
    int_matrix = np.array([[ord(c) - 65 for c in s] for s in series], dtype=np.int32)

    return int_matrix


def _load_and_preprocess(debug=False):
    """
    Internal function to load raw data, merge with metadata, and perform feature engineering.
    Returns dictionaries of arrays for train, val, and test.
    """
    print("Loading metadata...")
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    if debug:
        print(f"Debug mode: Subsetting metadata to {Config.DEBUG_SUBSET_SIZE} samples.")
        train_meta = train_meta.head(Config.DEBUG_SUBSET_SIZE)
        val_meta = val_meta.head(Config.DEBUG_SUBSET_SIZE)
        test_meta = test_meta.head(Config.DEBUG_SUBSET_SIZE)

    print("Loading raw data...")
    # Load full raw datasets
    df_train_raw = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"))
    df_test_raw = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))

    # Drop target from raw train data to avoid column collision with metadata target during merge
    if "target" in df_train_raw.columns:
        df_train_raw = df_train_raw.drop(columns=["target"])

    # Prepare lookups
    # We merge metadata with raw data on 'id'
    # Raw train contains both train and val samples
    print("Merging data...")
    train_merged = train_meta.merge(df_train_raw, on="id", how="left")
    val_merged = val_meta.merge(df_train_raw, on="id", how="left")
    test_merged = test_meta.merge(df_test_raw, on="id", how="left")

    # Define feature columns
    # Continuous: f_00 to f_30, excluding f_27
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    cat_col = "f_27"
    target_col = "target"

    # 1. Feature Engineering: Continuous Standardization
    print("Processing continuous features...")
    X_train_cont = train_merged[cont_cols].values.astype(np.float32)
    X_val_cont = val_merged[cont_cols].values.astype(np.float32)
    X_test_cont = test_merged[cont_cols].values.astype(np.float32)

    # Compute stats on TRAIN only
    mean = np.mean(X_train_cont, axis=0)
    std = np.std(X_train_cont, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0

    # Apply standardization
    X_train_cont = (X_train_cont - mean) / std
    X_val_cont = (X_val_cont - mean) / std
    X_test_cont = (X_test_cont - mean) / std

    # 2. Feature Engineering: Categorical Encoding
    print(f"Processing categorical feature {cat_col}...")
    X_train_cat = _process_f27(train_merged[cat_col])
    X_val_cat = _process_f27(val_merged[cat_col])
    X_test_cat = _process_f27(test_merged[cat_col])

    # 3. Targets
    y_train = train_merged[target_col].values.astype(np.float32).reshape(-1, 1)
    y_val = val_merged[target_col].values.astype(np.float32).reshape(-1, 1)
    # Test has no target, store IDs instead for submission mapping if needed,
    # though DataLoader usually just returns features. We don't need y_test.

    # Store test IDs for reference if needed later (though usually handled by metadata)
    test_ids = test_merged["id"].values

    data_dict = {
        "train": {"cont": X_train_cont, "cat": X_train_cat, "y": y_train},
        "val": {"cont": X_val_cont, "cat": X_val_cat, "y": y_val},
        "test": {"cont": X_test_cont, "cat": X_test_cat, "ids": test_ids},
    }

    return data_dict


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    debug=Config.DEBUG,
):
    """
    Main entry point to get PyTorch DataLoaders.
    Handles caching logic: checks for existing processed data in Config.CACHE_DIR.
    If not found or forced reload, processes from scratch and saves.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    data = None

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file) and not debug:
        print(f"Loading cached data from {cache_file}...")
        try:
            loaded = np.load(cache_file)
            data = {
                "train": {
                    "cont": loaded["train_cont"],
                    "cat": loaded["train_cat"],
                    "y": loaded["train_y"],
                },
                "val": {
                    "cont": loaded["val_cont"],
                    "cat": loaded["val_cat"],
                    "y": loaded["val_y"],
                },
                "test": {
                    "cont": loaded["test_cont"],
                    "cat": loaded["test_cat"],
                    "ids": loaded["test_ids"],
                },
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
            data = None

    # If data is not loaded (cache miss, load error, or debug mode forced reprocessing)
    if data is None:
        print("Processing data from scratch...")
        data = _load_and_preprocess(debug=debug)

        # Save to cache only if not debugging (to avoid overwriting full cache with subset)
        if not debug:
            print(f"Saving processed data to {cache_file}...")
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            np.savez(
                cache_file,
                train_cont=data["train"]["cont"],
                train_cat=data["train"]["cat"],
                train_y=data["train"]["y"],
                val_cont=data["val"]["cont"],
                val_cat=data["val"]["cat"],
                val_y=data["val"]["y"],
                test_cont=data["test"]["cont"],
                test_cat=data["test"]["cat"],
                test_ids=data["test"]["ids"],
            )

    # Create Datasets
    train_dataset = ManufacturingDataset(
        data["train"]["cont"], data["train"]["cat"], data["train"]["y"]
    )
    val_dataset = ManufacturingDataset(
        data["val"]["cont"], data["val"]["cat"], data["val"]["y"]
    )
    test_dataset = ManufacturingDataset(
        data["test"]["cont"], data["test"]["cat"], targets=None
    )

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
