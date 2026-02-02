import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    Holds continuous features, categorical sequence features, targets, and IDs.
    """

    def __init__(self, cont_data, cat_data, targets=None, ids=None):
        self.cont_data = torch.FloatTensor(cont_data)
        self.cat_data = torch.LongTensor(cat_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.ids = ids

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        item = {
            "cont_features": self.cont_data[idx],
            "cat_features": self.cat_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        if self.ids is not None:
            item["id"] = self.ids[idx]
        return item


def process_f27(series):
    """
    Decomposes the f_27 string feature into a (N, 10) array of integer indices.
    Mapping: 'A' -> 1, ..., 'Z' -> 26.
    """
    # Convert characters to 1-based indices (A=1, B=2, ...)
    # ord('A') is 65, so we subtract 64.
    return np.array([[ord(c) - 64 for c in s] for s in series], dtype=np.int64)


def get_dataloaders(load_cached_data=True):
    """
    Main function to load data, process it, and return DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load from cached .npz file.

    Returns:
        train_loader, val_loader, test_loader
    """
    cache_path = Config.PROCESSED_DATA_PATH

    train_cont, train_cat, train_y = None, None, None
    val_cont, val_cat, val_y = None, None, None
    test_cont, test_cat, test_ids = None, None, None

    loaded = False

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached data from {cache_path}")
            data = np.load(cache_path)
            train_cont = data["train_cont"]
            train_cat = data["train_cat"]
            train_y = data["train_y"]
            val_cont = data["val_cont"]
            val_cat = data["val_cat"]
            val_y = data["val_y"]
            test_cont = data["test_cont"]
            test_cat = data["test_cat"]
            test_ids = data["test_ids"]
            loaded = True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            loaded = False

    # 2. Process from scratch if not loaded
    if not loaded:
        print("Processing data from scratch...")

        # Load Metadata
        train_meta = pd.read_csv(Config.TRAIN_METADATA)
        val_meta = pd.read_csv(Config.VAL_METADATA)
        test_meta = pd.read_csv(Config.TEST_METADATA)

        # Load Raw Data
        # Indexing by ID allows efficient retrieval based on metadata
        full_train_df = pd.read_csv(Config.TRAIN_CSV).set_index("id")
        full_test_df = pd.read_csv(Config.TEST_CSV).set_index("id")

        # Retrieve IDs
        train_ids_list = train_meta["id"].values
        val_ids_list = val_meta["id"].values
        test_ids_list = test_meta["id"].values

        # Create DataFrames for each split
        # This ensures we respect the stratified split defined in metadata
        train_df = full_train_df.loc[train_ids_list]
        val_df = full_train_df.loc[val_ids_list]
        test_df = full_test_df.loc[test_ids_list]

        # --- Feature Engineering ---

        # 1. Continuous Features: Standardization
        scaler = StandardScaler()
        # Fit only on training data
        train_cont = scaler.fit_transform(
            train_df[Config.CONT_FEATURES].values.astype(np.float32)
        )
        val_cont = scaler.transform(
            val_df[Config.CONT_FEATURES].values.astype(np.float32)
        )
        test_cont = scaler.transform(
            test_df[Config.CONT_FEATURES].values.astype(np.float32)
        )

        # 2. Categorical Feature: Tokenization
        train_cat = process_f27(train_df[Config.CAT_FEATURE])
        val_cat = process_f27(val_df[Config.CAT_FEATURE])
        test_cat = process_f27(test_df[Config.CAT_FEATURE])

        # 3. Targets
        train_y = train_df[Config.TARGET_COL].values.astype(np.float32)
        val_y = val_df[Config.TARGET_COL].values.astype(np.float32)

        # 4. Test IDs (preserved for submission)
        test_ids = test_ids_list

        # Save to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
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
        print(f"Data processed and saved to {cache_path}")

    # 3. Handle Debug Mode (Slicing)
    if Config.DEBUG_SAMPLE_SIZE is not None:
        limit = int(Config.DEBUG_SAMPLE_SIZE)
        print(f"Debug mode active: Limiting dataset to {limit} samples.")
        train_cont = train_cont[:limit]
        train_cat = train_cat[:limit]
        train_y = train_y[:limit]

        val_cont = val_cont[:limit]
        val_cat = val_cat[:limit]
        val_y = val_y[:limit]

        test_cont = test_cont[:limit]
        test_cat = test_cat[:limit]
        test_ids = test_ids[:limit]

    # 4. Instantiate Datasets
    train_dataset = ManufacturingDataset(train_cont, train_cat, train_y)
    val_dataset = ManufacturingDataset(val_cont, val_cat, val_y)
    # Test dataset includes IDs for submission mapping
    test_dataset = ManufacturingDataset(test_cont, test_cat, targets=None, ids=test_ids)

    # 5. Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
