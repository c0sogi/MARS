import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    def __init__(self, continuous_features, categorical_features, targets=None):
        """
        Args:
            continuous_features (np.ndarray): Shape (N, 30) normalized features.
            categorical_features (np.ndarray): Shape (N, 10) integer-encoded tokens.
            targets (np.ndarray, optional): Shape (N,) binary targets.
        """
        self.continuous_features = torch.tensor(
            continuous_features, dtype=torch.float32
        )
        self.categorical_features = torch.tensor(categorical_features, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.continuous_features)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_features[idx],
            "categorical": self.categorical_features[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def process_f27(series):
    """
    Decomposes the string feature f_27 into 10 integer columns.
    Assumes characters are uppercase A-Z.
    """
    # Convert each string to a list of ASCII values, subtracted by ord('A')
    # This maps 'A' -> 0, 'B' -> 1, etc.
    # We expect fixed length of 10.
    return np.array([[ord(c) - ord("A") for c in s] for s in series], dtype=np.int32)


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.
    """
    cache_dir = "./working/idea_38"
    cache_file = os.path.join(cache_dir, "processed_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        data = np.load(cache_file)
        return (
            data["X_cont_train"],
            data["X_cat_train"],
            data["y_train"],
            data["X_cont_val"],
            data["X_cat_val"],
            data["y_val"],
            data["X_cont_test"],
            data["X_cat_test"],
            data["test_ids"],
        )

    print("Processing data from scratch...")
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Load Metadata
    meta_dir = "./metadata"
    train_meta = pd.read_csv(os.path.join(meta_dir, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(meta_dir, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(meta_dir, "test_metadata.csv"))

    # 3. Load Raw Data
    input_dir = "./input"
    raw_train = pd.read_csv(os.path.join(input_dir, "train.csv"))
    raw_test = pd.read_csv(os.path.join(input_dir, "test.csv"))

    # 4. Merge to get features aligned with splits
    # We use inner merge on ID to select the correct rows based on metadata
    df_train = train_meta.merge(raw_train, on="id", suffixes=("", "_raw"))
    df_val = val_meta.merge(raw_train, on="id", suffixes=("", "_raw"))
    df_test = test_meta.merge(raw_test, on="id", suffixes=("", "_raw"))

    # Handle potential target column duplication if 'target' is in raw_train
    if "target_raw" in df_train.columns:
        df_train = df_train.drop(columns=["target_raw"])
    if "target_raw" in df_val.columns:
        df_val = df_val.drop(columns=["target_raw"])

    # 5. Feature Engineering

    # Identify continuous columns: f_00 to f_30, excluding f_27
    # We can programmatically select them.
    all_cols = raw_train.columns.tolist()
    cont_cols = [c for c in all_cols if c.startswith("f_") and c != "f_27"]

    print(
        f"Processing {len(cont_cols)} continuous features and 1 categorical sequence (f_27)..."
    )

    # 5a. Continuous Features - Scaling
    scaler = StandardScaler()

    # Fit on TRAIN only
    X_cont_train = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    X_cont_val = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    X_cont_test = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # 5b. Categorical Feature - Tokenization
    X_cat_train = process_f27(df_train["f_27"])
    X_cat_val = process_f27(df_val["f_27"])
    X_cat_test = process_f27(df_test["f_27"])

    # 5c. Targets and IDs
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # 6. Save to Cache
    print(f"Saving processed data to {cache_file}...")
    np.savez(
        cache_file,
        X_cont_train=X_cont_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_cont_val=X_cont_val,
        X_cat_val=X_cat_val,
        y_val=y_val,
        X_cont_test=X_cont_test,
        X_cat_test=X_cat_test,
        test_ids=test_ids,
    )

    return (
        X_cont_train,
        X_cat_train,
        y_train,
        X_cont_val,
        X_cat_val,
        y_val,
        X_cont_test,
        X_cat_test,
        test_ids,
    )


def get_dataloaders(batch_size=1024, load_cached_data=True, num_workers=4):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    seed_everything()

    (
        X_cont_train,
        X_cat_train,
        y_train,
        X_cont_val,
        X_cat_val,
        y_val,
        X_cont_test,
        X_cat_test,
        test_ids,
    ) = process_data(load_cached_data=load_cached_data)

    train_dataset = ManufacturingDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = ManufacturingDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = ManufacturingDataset(X_cont_test, X_cat_test, targets=None)

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

    print(
        f"DataLoaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches."
    )

    return train_loader, val_loader, test_loader, test_ids
