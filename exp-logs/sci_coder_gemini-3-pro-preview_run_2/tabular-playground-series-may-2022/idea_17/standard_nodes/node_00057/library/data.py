import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from library.config import Config


class FeatureEngineer:
    """
    Handles the preprocessing of manufacturing data:
    1. Vectorizes f_27 string feature into integer sequences.
    2. Normalizes continuous features (Z-score).
    3. Discretizes continuous features into quantile bins.
    """

    def __init__(self):
        self.continuous_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
        self.scaler = StandardScaler()
        # encode='ordinal' returns integer indices. strategy='quantile' creates equal-density bins.
        self.discretizer = KBinsDiscretizer(
            n_bins=Config.QUANTILE_BINS,
            encode="ordinal",
            strategy="quantile",
            subsample=200000,  # Subsample for speed on large datasets
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler and discretizer on the provided dataframe (training set).
        """
        X_cont = df[self.continuous_cols].values
        self.scaler.fit(X_cont)
        self.discretizer.fit(X_cont)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame):
        """
        Transforms the dataframe into the required numpy arrays.
        Returns:
            seq_data: (N, 10) int array for f_27
            raw_data: (N, 30) float array (normalized)
            binned_data: (N, 30) int array (quantized)
            targets: (N,) float array or None
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform.")

        # 1. Process Sequence (f_27)
        # Map A->1, B->2, ..., Z->26
        # f_27 is always length 10 based on EDA
        def encode_seq(s):
            return [ord(c) - ord("A") + 1 for c in s]

        # Use pandas apply for simplicity, can be optimized with numpy view if needed
        seq_list = df["f_27"].apply(encode_seq).tolist()
        seq_data = np.array(seq_list, dtype=np.int64)

        # 2. Process Continuous Features
        X_cont = df[self.continuous_cols].values

        # Path A: Raw Normalized
        raw_data = self.scaler.transform(X_cont).astype(np.float32)

        # Path B: Quantile Binned
        # KBinsDiscretizer returns floats by default even with ordinal, cast to int
        binned_data = self.discretizer.transform(X_cont).astype(np.int64)

        # 3. Targets
        if "target" in df.columns:
            targets = df["target"].values.astype(np.float32)
        else:
            targets = None

        return seq_data, raw_data, binned_data, targets


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Task.
    Serves:
        - x_seq: Tokenized sequence of f_27
        - x_raw: Normalized continuous features
        - x_binned: Quantized continuous features
        - target: Binary label
    """

    def __init__(self, seq_data, raw_data, binned_data, targets=None):
        self.seq_data = torch.from_numpy(seq_data).long()
        self.raw_data = torch.from_numpy(raw_data).float()
        self.binned_data = torch.from_numpy(binned_data).long()

        if targets is not None:
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = None

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        item = {
            "x_seq": self.seq_data[idx],
            "x_raw": self.raw_data[idx],
            "x_binned": self.binned_data[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        return item


def process_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, and caching.

    Logic:
    1. If cached .npz exists and load_cached_data is True, load and return.
    2. Else:
       - Load metadata to define splits.
       - Load raw CSVs.
       - Fit FeatureEngineer on Train split.
       - Transform Train, Val, Test.
       - Save to cache.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        required_keys = [
            "train_seq",
            "train_raw",
            "train_binned",
            "train_y",
            "val_seq",
            "val_raw",
            "val_binned",
            "val_y",
            "test_seq",
            "test_raw",
            "test_binned",
        ]
        if all(k in data for k in required_keys):
            return (
                data["train_seq"],
                data["train_raw"],
                data["train_binned"],
                data["train_y"],
                data["val_seq"],
                data["val_raw"],
                data["val_binned"],
                data["val_y"],
                data["test_seq"],
                data["test_raw"],
                data["test_binned"],
                None,  # Test y is None
            )
        print("Cached data is missing required keys or incompatible. Regenerating...")

    print("Processing data from scratch...")

    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Load Raw Data
    # We load full train and test files. Indexing by ID allows fast lookup.
    print("Loading raw CSV files...")
    df_train_full = pd.read_csv(Config.TRAIN_DATA_PATH).set_index("id")
    df_test_full = pd.read_csv(Config.TEST_DATA_PATH).set_index("id")

    # 3. Slice Dataframes based on Metadata
    # train_meta['id'] contains the IDs for the training set
    df_train = df_train_full.loc[train_meta["id"]].copy()
    df_val = df_train_full.loc[val_meta["id"]].copy()
    df_test = df_test_full.loc[test_meta["id"]].copy()

    # Ensure targets are present in training/val splits (sanity check)
    df_train["target"] = train_meta["target"].values
    df_val["target"] = val_meta["target"].values

    # 4. Feature Engineering
    print("Fitting feature engineer...")
    fe = FeatureEngineer()
    fe.fit(df_train)

    print("Transforming datasets...")
    train_seq, train_raw, train_binned, train_y = fe.transform(df_train)
    val_seq, val_raw, val_binned, val_y = fe.transform(df_val)
    test_seq, test_raw, test_binned, _ = fe.transform(df_test)

    # 5. Cache Results
    print(f"Saving processed data to {cache_path}...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_seq=train_seq,
        train_raw=train_raw,
        train_binned=train_binned,
        train_y=train_y,
        val_seq=val_seq,
        val_raw=val_raw,
        val_binned=val_binned,
        val_y=val_y,
        test_seq=test_seq,
        test_raw=test_raw,
        test_binned=test_binned,
    )

    return (
        train_seq,
        train_raw,
        train_binned,
        train_y,
        val_seq,
        val_raw,
        val_binned,
        val_y,
        test_seq,
        test_raw,
        test_binned,
        None,
    )


def get_dataloaders(load_cached_data=True, debug_subset=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        debug_subset (int, optional): If provided, reduces dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load arrays
    (
        train_seq,
        train_raw,
        train_binned,
        train_y,
        val_seq,
        val_raw,
        val_binned,
        val_y,
        test_seq,
        test_raw,
        test_binned,
        _,
    ) = process_data(load_cached_data)

    # Debugging / Subsampling
    if debug_subset is not None:
        print(f"DEBUG: Subsampling datasets to {debug_subset} samples.")
        train_seq = train_seq[:debug_subset]
        train_raw = train_raw[:debug_subset]
        train_binned = train_binned[:debug_subset]
        train_y = train_y[:debug_subset]

        val_seq = val_seq[:debug_subset]
        val_raw = val_raw[:debug_subset]
        val_binned = val_binned[:debug_subset]
        val_y = val_y[:debug_subset]

        test_seq = test_seq[:debug_subset]
        test_raw = test_raw[:debug_subset]
        test_binned = test_binned[:debug_subset]

    # Create Datasets
    train_dataset = ManufacturingDataset(train_seq, train_raw, train_binned, train_y)
    val_dataset = ManufacturingDataset(val_seq, val_raw, val_binned, val_y)
    test_dataset = ManufacturingDataset(test_seq, test_raw, test_binned, None)

    # Create DataLoaders
    # Use num_workers > 0 for efficiency, pin_memory for GPU transfer speed
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
