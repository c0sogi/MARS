import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# Ensure reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(42)


class ManufacturingDataset(Dataset):
    def __init__(self, continuous_data, sequence_data, targets=None):
        """
        Args:
            continuous_data (np.ndarray): Normalized continuous features (N, 30).
            sequence_data (np.ndarray): Integer encoded sequence features (N, 10).
            targets (np.ndarray, optional): Binary targets (N,).
        """
        self.continuous_data = torch.FloatTensor(continuous_data)
        self.sequence_data = torch.LongTensor(sequence_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.continuous_data[idx], self.sequence_data[idx], self.targets[idx]
        else:
            return self.continuous_data[idx], self.sequence_data[idx]


def process_sequence_feature(series):
    """
    Converts a pandas Series of strings (e.g., 'ABAC...') into a numpy array of integers.
    Assumes fixed length of 10 and characters 'A'-'Z'.
    Maps 'A' -> 0, 'B' -> 1, etc.
    """
    # Convert series to list of strings
    s_list = series.astype(str).tolist()

    # Use list comprehension for efficient character-to-int mapping
    # ord('A') is 65. So 'A' becomes 0, 'B' becomes 1.
    data = [[ord(c) - 65 for c in s] for s in s_list]

    return np.array(data, dtype=np.int64)


def get_dataloaders(
    batch_size=1024,
    load_cached_data=True,
    max_samples=None,
    data_dir="./input",
    metadata_dir="./metadata",
    cache_dir="./working/idea_24",
):
    """
    Loads data, processes features, and returns PyTorch DataLoaders.
    Implements caching to speed up subsequent runs.
    """

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "processed_data.npz")

    data_loaded = False

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            cached = np.load(cache_path)
            X_cont_train = cached["X_cont_train"]
            X_seq_train = cached["X_seq_train"]
            y_train = cached["y_train"]

            X_cont_val = cached["X_cont_val"]
            X_seq_val = cached["X_seq_val"]
            y_val = cached["y_val"]

            X_cont_test = cached["X_cont_test"]
            X_seq_test = cached["X_seq_test"]
            test_ids = cached["test_ids"]

            data_loaded = True
            print("Cache loaded successfully.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")
            data_loaded = False

    # 2. Process from scratch if needed
    if not data_loaded:
        print("Processing data from scratch...")

        # Load Raw Data
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Raw data files not found in ./input")

        df_raw_train = pd.read_csv(train_path)
        df_raw_test = pd.read_csv(test_path)

        # Load Metadata for Splitting
        train_meta = pd.read_csv(os.path.join(metadata_dir, "train_metadata.csv"))
        val_meta = pd.read_csv(os.path.join(metadata_dir, "val_metadata.csv"))
        test_meta = pd.read_csv(os.path.join(metadata_dir, "test_metadata.csv"))

        # Set ID as index for fast alignment
        df_raw_train.set_index("id", inplace=True)
        df_raw_test.set_index("id", inplace=True)

        # Align data with metadata splits
        df_train = df_raw_train.loc[train_meta["id"]]
        df_val = df_raw_train.loc[val_meta["id"]]
        df_test = df_raw_test.loc[test_meta["id"]]

        # Define Feature Columns
        # Continuous: f_00 to f_30, excluding f_27
        cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
        seq_col = "f_27"
        target_col = "target"

        # Process Sequence Features (f_27)
        print("Encoding sequence features...")
        X_seq_train = process_sequence_feature(df_train[seq_col])
        X_seq_val = process_sequence_feature(df_val[seq_col])
        X_seq_test = process_sequence_feature(df_test[seq_col])

        # Process Continuous Features
        print("Normalizing continuous features...")
        X_cont_train_raw = df_train[cont_cols].values.astype(np.float32)
        X_cont_val_raw = df_val[cont_cols].values.astype(np.float32)
        X_cont_test_raw = df_test[cont_cols].values.astype(np.float32)

        # Fit Scaler ONLY on Training Data
        scaler = StandardScaler()
        X_cont_train = scaler.fit_transform(X_cont_train_raw)
        X_cont_val = scaler.transform(X_cont_val_raw)
        X_cont_test = scaler.transform(X_cont_test_raw)

        # Extract Targets and IDs
        y_train = df_train[target_col].values.astype(np.float32)
        y_val = df_val[target_col].values.astype(np.float32)
        test_ids = test_meta["id"].values

        # Save to Cache
        print(f"Saving processed data to {cache_path}...")
        np.savez(
            cache_path,
            X_cont_train=X_cont_train,
            X_seq_train=X_seq_train,
            y_train=y_train,
            X_cont_val=X_cont_val,
            X_seq_val=X_seq_val,
            y_val=y_val,
            X_cont_test=X_cont_test,
            X_seq_test=X_seq_test,
            test_ids=test_ids,
        )

    # 3. Subsampling (Optional for debugging)
    if max_samples is not None:
        print(f"Subsampling data to {max_samples} training samples...")
        X_cont_train = X_cont_train[:max_samples]
        X_seq_train = X_seq_train[:max_samples]
        y_train = y_train[:max_samples]

        # Scale validation set proportionally (approx 20% of max_samples)
        val_limit = max(1, int(max_samples * 0.25))
        X_cont_val = X_cont_val[:val_limit]
        X_seq_val = X_seq_val[:val_limit]
        y_val = y_val[:val_limit]

    # 4. Create PyTorch Datasets
    train_dataset = ManufacturingDataset(X_cont_train, X_seq_train, y_train)
    val_dataset = ManufacturingDataset(X_cont_val, X_seq_val, y_val)
    test_dataset = ManufacturingDataset(X_cont_test, X_seq_test, None)

    # 5. Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
