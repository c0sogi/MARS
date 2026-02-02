import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.utils import set_seed


class ManufacturingDataset(Dataset):
    def __init__(self, continuous_data, categorical_data, targets=None):
        """
        Args:
            continuous_data (np.ndarray): Normalized continuous features (N, 30).
            categorical_data (np.ndarray): Integer-encoded categorical features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
        self.categorical_data = torch.tensor(categorical_data, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return (
                self.continuous_data[idx],
                self.categorical_data[idx],
                self.targets[idx],
            )
        else:
            # Return -1 or dummy for target if not available (e.g. test set)
            # Returning 0.0 as placeholder
            return (
                self.continuous_data[idx],
                self.categorical_data[idx],
                torch.tensor(0.0, dtype=torch.float32),
            )


def _process_f27(series):
    """
    Converts a Series of strings (length 10) into a numpy array of integers.
    Mapping: 'A' -> 1, 'B' -> 2, etc.
    """
    # Convert series to list of strings, then to list of lists of characters
    # This is reasonably fast for 1M rows
    # We assume uppercase A-Z. ord('A') is 65.
    # We want A=1. So ord(c) - 64.

    # Vectorized approach using list comprehension which is often faster than pandas apply for string ops
    # Create a lookup or just math.
    # Using math: ord(c) - 64

    # Ensure we handle the data as pure strings
    str_list = series.astype(str).tolist()

    # Build a matrix.
    # Note: This assumes all strings are length 10 and valid uppercase.
    # Based on EDA, this is a safe assumption.

    # Map characters to integers
    # ord('A') = 65. We map A->1, B->2...
    # 0 is reserved for padding if needed (not needed here).

    processed = []
    for s in str_list:
        processed.append([ord(c) - 64 for c in s])

    return np.array(processed, dtype=np.int32)


def get_dataloaders(
    batch_size=1024,
    num_workers=4,
    load_cached_data=True,
    cache_dir="./working/idea_13",
    input_dir="./input",
    metadata_dir="./metadata",
):
    """
    Loads data, processes it (scaling, tokenization), and returns PyTorch DataLoaders.
    Implements caching to speed up subsequent runs.
    """
    set_seed(42)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "processed_data.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            data = np.load(cache_file)
            X_cont_train = data["X_cont_train"]
            X_cat_train = data["X_cat_train"]
            y_train = data["y_train"]

            X_cont_val = data["X_cont_val"]
            X_cat_val = data["X_cat_val"]
            y_val = data["y_val"]

            X_cont_test = data["X_cont_test"]
            X_cat_test = data["X_cat_test"]
            # Test targets are not needed/available for prediction

            print("Data loaded successfully from cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")
            load_cached_data = False  # Force re-processing

    # 2. Process from scratch if cache not loaded
    if not load_cached_data or not os.path.exists(cache_file):
        print("Processing data from scratch...")

        # Load Metadata
        train_meta = pd.read_csv(os.path.join(metadata_dir, "train_metadata.csv"))
        val_meta = pd.read_csv(os.path.join(metadata_dir, "val_metadata.csv"))
        test_meta = pd.read_csv(os.path.join(metadata_dir, "test_metadata.csv"))

        # Load Raw Data
        # We read the full files and then index them.
        # This consumes RAM but is straightforward.
        print("Reading raw CSVs...")
        df_train_full = pd.read_csv(os.path.join(input_dir, "train.csv"))
        df_test_full = pd.read_csv(os.path.join(input_dir, "test.csv"))

        # Index by ID for fast lookup
        df_train_full.set_index("id", inplace=True)
        df_test_full.set_index("id", inplace=True)

        # Extract Subsets based on Metadata
        # Metadata IDs are integers.
        print("Splitting and aligning data...")
        train_ids = train_meta["id"].values
        val_ids = val_meta["id"].values
        test_ids = test_meta["id"].values

        # Locating rows.
        # Note: df_train_full contains both train and val samples originally,
        # but metadata splits them.
        df_train = df_train_full.loc[train_ids]
        df_val = df_train_full.loc[val_ids]
        df_test = df_test_full.loc[test_ids]

        # Feature Selection
        # Continuous features: f_00 to f_30
        cont_cols = [f"f_{i:02d}" for i in range(31)]
        # Categorical feature: f_27
        cat_col = "f_27"

        # Extract Continuous
        print("Processing continuous features...")
        X_cont_train = df_train[cont_cols].values.astype(np.float32)
        X_cont_val = df_val[cont_cols].values.astype(np.float32)
        X_cont_test = df_test[cont_cols].values.astype(np.float32)

        # Scaling
        # Fit on Train ONLY
        scaler = StandardScaler()
        X_cont_train = scaler.fit_transform(X_cont_train)
        X_cont_val = scaler.transform(X_cont_val)
        X_cont_test = scaler.transform(X_cont_test)

        # Extract Categorical
        print("Processing categorical feature f_27...")
        X_cat_train = _process_f27(df_train[cat_col])
        X_cat_val = _process_f27(df_val[cat_col])
        X_cat_test = _process_f27(df_test[cat_col])

        # Extract Targets
        y_train = df_train["target"].values.astype(np.float32)
        y_val = df_val["target"].values.astype(np.float32)

        # Save to cache
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
        )

    # 3. Create Datasets
    train_dataset = ManufacturingDataset(X_cont_train, X_cat_train, y_train)
    val_dataset = ManufacturingDataset(X_cont_val, X_cat_val, y_val)
    test_dataset = ManufacturingDataset(X_cont_test, X_cat_test, targets=None)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
