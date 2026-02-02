import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import library.config as config
from library.utils import seed_everything

# ------------------------------------------------------------------------------
# Feature Engineering Logic
# ------------------------------------------------------------------------------


def process_f27(series):
    """
    Decomposes the f_27 string feature into 10 position-distinct integer tokens.
    Mapping: token = position * 26 + (char_val - 'A')
    """
    # Convert series to list of strings
    strings = series.values.astype(str)

    # Initialize array: (n_samples, 10)
    n_samples = len(strings)
    seq_len = 10
    token_array = np.zeros((n_samples, seq_len), dtype=np.int32)

    # Vectorized processing is tricky with strings in numpy,
    # but list comprehension is fast enough for 1M rows.
    # Alternatively, view as uint8 (byte) array if fixed width.
    # Given the constraints and likely ASCII, we can do a fast char map.

    # Create a map for 'A' through 'Z' to 0..25
    # ord('A') is 65.

    # We can convert the whole column to a 2D view of bytes
    # strings are fixed length 10.
    # We need to ensure they are strictly length 10.

    # Fast approach:
    # 1. Convert to a contiguous array of characters (bytes)
    # Pandas series of strings to numpy array of objects, then to char view?
    # Safer approach for robustness:

    for i in range(seq_len):
        # Extract the i-th character
        # Using str accessor is convenient but can be slow.
        # For 1M rows, it takes a few seconds.
        chars = series.str[i].apply(lambda x: ord(x) - ord("A")).values

        # Shared vocabulary: No position offset
        token_array[:, i] = chars

    return token_array


# ------------------------------------------------------------------------------
# Data Processing & Caching
# ------------------------------------------------------------------------------


def preprocess_data(load_cached_data=True):
    """
    Loads raw data, performs feature engineering and scaling, and caches the result.
    Returns dictionary containing processed numpy arrays for full train and test sets.
    """
    cache_file = os.path.join(config.CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        try:
            data = np.load(cache_file)
            return {k: data[k] for k in data.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing...")

    print("Processing data from scratch...")

    # Load Raw Data
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    # 1. Continuous Features (f_00 - f_30)
    # Identify continuous columns (exclude id, f_27, target)
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]

    # Fit Scaler on Full Training Data
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train_df[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(test_df[cont_cols].values.astype(np.float32))

    # 2. Categorical Feature (f_27)
    print("Processing categorical feature f_27...")
    train_cat = process_f27(train_df["f_27"])
    test_cat = process_f27(test_df["f_27"])

    # 3. Targets and IDs
    train_target = train_df[config.TARGET_COL].values.astype(np.float32)
    train_ids = train_df[config.ID_COL].values.astype(np.int64)
    test_ids = test_df[config.ID_COL].values.astype(np.int64)

    # Pack into dictionary
    processed_data = {
        "train_cont": train_cont,
        "train_cat": train_cat,
        "train_target": train_target,
        "train_ids": train_ids,
        "test_cont": test_cont,
        "test_cat": test_cat,
        "test_ids": test_ids,
    }

    # Save to cache
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_file, **processed_data)
    print(f"Data processed and saved to {cache_file}")

    return processed_data


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class ManufacturingDataset(Dataset):
    def __init__(
        self, metadata_df, cont_data, cat_data, target_data=None, ids_data=None
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id' column for this split.
            cont_data (np.ndarray): Full array of continuous features.
            cat_data (np.ndarray): Full array of categorical features.
            target_data (np.ndarray, optional): Full array of targets.
            ids_data (np.ndarray): Full array of IDs corresponding to the data arrays.
        """
        self.metadata = metadata_df.copy()

        # Create a mapping from ID to index in the full data arrays
        # This assumes ids_data corresponds to the rows of cont_data/cat_data
        print(f"Indexing dataset with {len(metadata_df)} samples...")

        # We need to map the IDs in metadata to the indices in the provided data arrays.
        # Since ids_data might be large (800k), we create a lookup.
        # However, to speed up __getitem__, we will pre-slice the arrays.

        # Create a hash map for the full data IDs
        full_id_map = {id_val: idx for idx, id_val in enumerate(ids_data)}

        # Find indices for the current split
        # We filter metadata to ensure all IDs exist in the source data
        valid_indices = []
        valid_rows = []

        for _, row in self.metadata.iterrows():
            uid = row["id"]
            if uid in full_id_map:
                valid_indices.append(full_id_map[uid])
                valid_rows.append(row)

        if len(valid_indices) != len(self.metadata):
            print(
                f"Warning: {len(self.metadata) - len(valid_indices)} IDs from metadata not found in source data."
            )

        # Pre-slice the data for this dataset instance
        indices = np.array(valid_indices)

        self.cont = torch.tensor(cont_data[indices], dtype=torch.float32)
        self.cat = torch.tensor(cat_data[indices], dtype=torch.long)
        self.ids = torch.tensor(ids_data[indices], dtype=torch.long)

        if target_data is not None:
            self.targets = torch.tensor(target_data[indices], dtype=torch.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "continuous": self.cont[idx],
            "categorical": self.cat[idx],
            "id": self.ids[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        return item


# ------------------------------------------------------------------------------
# Data Loaders
# ------------------------------------------------------------------------------


def get_dataloaders(
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    load_cached_data=True,
    debug=None,
):
    """
    Creates DataLoaders for train, validation, and test sets using metadata splits.
    """
    if debug is None:
        debug = config.DEBUG

    seed_everything(config.RANDOM_STATE)

    # 1. Get processed full data
    data = preprocess_data(load_cached_data=load_cached_data)

    # 2. Load Metadata
    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # Debugging: Subsample metadata
    if debug:
        print(f"DEBUG MODE: Subsampling to {config.DEBUG_SUBSET_SIZE} samples.")
        train_meta = train_meta.head(config.DEBUG_SUBSET_SIZE)
        val_meta = val_meta.head(config.DEBUG_SUBSET_SIZE)
        test_meta = test_meta.head(config.DEBUG_SUBSET_SIZE)

    # 3. Create Datasets
    # Train and Val come from the "train" source data
    train_dataset = ManufacturingDataset(
        metadata_df=train_meta,
        cont_data=data["train_cont"],
        cat_data=data["train_cat"],
        target_data=data["train_target"],
        ids_data=data["train_ids"],
    )

    val_dataset = ManufacturingDataset(
        metadata_df=val_meta,
        cont_data=data["train_cont"],
        cat_data=data["train_cat"],
        target_data=data["train_target"],
        ids_data=data["train_ids"],
    )

    # Test comes from the "test" source data
    test_dataset = ManufacturingDataset(
        metadata_df=test_meta,
        cont_data=data["test_cont"],
        cat_data=data["test_cat"],
        target_data=None,
        ids_data=data["test_ids"],
    )

    # 4. Create DataLoaders
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
