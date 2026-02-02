import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control task.
    Serves continuous features, decomposed categorical sequence, and targets.
    """

    def __init__(self, continuous, categorical, targets=None):
        self.continuous = torch.FloatTensor(continuous)
        self.categorical = torch.LongTensor(categorical)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.continuous)

    def __getitem__(self, idx):
        sample = {
            "continuous": self.continuous[idx],
            "categorical": self.categorical[idx],
        }
        if self.targets is not None:
            sample["target"] = self.targets[idx]
        return sample


def preprocess_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (normalization, tokenization),
    and caches the result to disk.

    Args:
        load_cached_data (bool): If True, attempts to load from Config.CACHE_PATH.

    Returns:
        Tuple of numpy arrays:
        (train_cont, train_cat, train_target, train_ids, test_cont, test_cat, test_ids)
    """
    # 1. Try loading from cache
    if load_cached_data and os.path.exists(Config.CACHE_PATH):
        print(f"Loading cached data from {Config.CACHE_PATH}")
        try:
            data = np.load(Config.CACHE_PATH)
            return (
                data["train_cont"],
                data["train_cat"],
                data["train_target"],
                data["train_ids"],
                data["test_cont"],
                data["test_cat"],
                data["test_ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch...")

    print("Preprocessing data from scratch...")

    # 2. Load Raw Data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 3. Process Continuous Features (f_00 - f_30, excluding f_27)
    # Identify continuous columns dynamically or by range
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    scaler = StandardScaler()
    # Fit on the full training set (as per standard practice for this dataset structure)
    train_cont = scaler.fit_transform(train_df[cont_cols].values.astype(np.float32))
    # Transform test set
    test_cont = scaler.transform(test_df[cont_cols].values.astype(np.float32))

    # 4. Process Categorical Feature (f_27)
    # Decompose string into 10 integer tokens (A=0, B=1, ..., Z=25)
    def encode_f27(series):
        # Convert series of strings to a numpy array of character lists
        # This is efficient for the given dataset size
        chars = np.array([list(s) for s in series.values])
        # Vectorized mapping: ord(c) - 65 ('A') -> 0-based index
        # Assumes inputs are uppercase A-Z
        tokens = np.vectorize(lambda x: ord(x) - 65)(chars).astype(np.int64)
        return tokens

    train_cat = encode_f27(train_df["f_27"])
    test_cat = encode_f27(test_df["f_27"])

    # 5. Extract Targets and IDs
    train_target = train_df["target"].values.astype(np.float32)
    train_ids = train_df["id"].values.astype(np.int64)
    test_ids = test_df["id"].values.astype(np.int64)

    # 6. Cache Results
    os.makedirs(os.path.dirname(Config.CACHE_PATH), exist_ok=True)
    np.savez(
        Config.CACHE_PATH,
        train_cont=train_cont,
        train_cat=train_cat,
        train_target=train_target,
        train_ids=train_ids,
        test_cont=test_cont,
        test_cat=test_cat,
        test_ids=test_ids,
    )
    print(f"Data cached to {Config.CACHE_PATH}")

    return train_cont, train_cat, train_target, train_ids, test_cont, test_cat, test_ids


def get_datasets(load_cached_data=True):
    """
    Factory function to create Train, Validation, and Test datasets.
    Uses metadata files to split the processed data correctly.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Load all processed data
    t_cont, t_cat, t_target, t_ids, te_cont, te_cat, te_ids = preprocess_data(
        load_cached_data
    )

    # Create fast lookup maps from ID to array index
    # This is necessary because metadata might reshuffle or subset the original data
    train_id_to_idx = pd.Series(data=np.arange(len(t_ids)), index=t_ids)
    test_id_to_idx = pd.Series(data=np.arange(len(te_ids)), index=te_ids)

    # Load Metadata for splits
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # Retrieve indices for each split
    # .loc ensures we get the exact index corresponding to the ID in metadata
    train_indices = train_id_to_idx.loc[train_meta["id"]].values
    val_indices = train_id_to_idx.loc[val_meta["id"]].values
    test_indices = test_id_to_idx.loc[test_meta["id"]].values

    # Instantiate Datasets
    train_dataset = ManufacturingDataset(
        continuous=t_cont[train_indices],
        categorical=t_cat[train_indices],
        targets=t_target[train_indices],
    )

    val_dataset = ManufacturingDataset(
        continuous=t_cont[val_indices],
        categorical=t_cat[val_indices],
        targets=t_target[val_indices],
    )

    test_dataset = ManufacturingDataset(
        continuous=te_cont[test_indices], categorical=te_cat[test_indices], targets=None
    )

    return train_dataset, val_dataset, test_dataset
