import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, continuous_data, sequence_data, targets=None):
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
        self.sequence_data = torch.tensor(sequence_data, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "sequence": self.sequence_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def preprocess_data(load_cached_data=True):
    """
    Loads raw data, performs preprocessing (scaling, tokenization), and caches the result.
    Returns a dictionary containing processed arrays and ID maps.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached processed data from {cache_path}...")
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            return {key: loaded[key] for key in loaded.files}
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing data.")

    print("Processing data from scratch...")

    # 2. Load Raw Data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Load train metadata to identify strict training set for scaler fitting
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    train_ids_set = set(train_meta["id"].values)

    # Concatenate for consistent processing (features only)
    # We keep track of indices: 0 to len(train_df)-1 are train/val, rest are test
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 3. Process Sequence Feature (f_27)
    # Decompose string into 10 integer columns
    print("Processing sequence feature f_27...")
    # Vectorized conversion of string to list of ASCII codes, then shift by 'A' (65)
    # We assume f_27 is always length 10 and uppercase A-Z
    # Using a list comprehension is efficient enough for ~1M rows
    sequence_list = [
        [ord(c) - 65 for c in s] for s in full_df[Config.SEQUENCE_FEATURE].values
    ]
    sequence_data = np.array(sequence_list, dtype=np.int32)

    # 4. Process Continuous Features
    print("Processing continuous features...")
    # Identify continuous columns: f_00 to f_30
    cont_cols = [f"f_{i:02d}" for i in range(31)]
    # Verify columns exist
    cont_cols = [c for c in cont_cols if c in full_df.columns]

    continuous_data = full_df[cont_cols].values.astype(np.float32)

    # Scale features
    # Fit ONLY on the training subset defined in metadata to avoid leakage
    # We need to find the indices in full_df that correspond to train_ids_set
    # Since full_df[:len(train_df)] is the train_df, we can look up there.
    train_indices_mask = full_df["id"].isin(train_ids_set)

    scaler = StandardScaler()
    print("Fitting scaler on training subset...")
    scaler.fit(continuous_data[train_indices_mask])

    print("Transforming all data...")
    continuous_data = scaler.transform(continuous_data)

    # 5. Extract Targets and IDs
    # Targets only exist for the first len(train_df) rows
    targets = np.full(len(full_df), -1.0, dtype=np.float32)
    targets[: len(train_df)] = train_df[Config.TARGET_COL].values

    ids = full_df[Config.ID_COL].values

    # 6. Save to Cache
    # We save everything as a single archive.
    # To reconstruct splits, we will use the ID mapping.
    print(f"Saving processed data to {cache_path}...")

    # Create an ID to Index map for fast retrieval
    # We can't save dicts in npz easily without pickle, so we save IDs and rely on order
    # or we can reconstruct the map after loading.

    data_dict = {
        "continuous_data": continuous_data,
        "sequence_data": sequence_data,
        "targets": targets,
        "ids": ids,
    }

    np.savez(cache_path, **data_dict)

    return data_dict


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets based on metadata splits.
    """
    # 1. Get processed data
    data = preprocess_data(load_cached_data=load_cached_data)

    continuous_all = data["continuous_data"]
    sequence_all = data["sequence_data"]
    targets_all = data["targets"]
    ids_all = data["ids"]

    # Create a fast lookup map: ID -> Index
    id_to_idx = {id_val: idx for idx, id_val in enumerate(ids_all)}

    # 2. Helper to create dataset from metadata file
    def create_dataset_from_meta(meta_path, is_test=False):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        meta_df = pd.read_csv(meta_path)
        meta_ids = meta_df["id"].values

        # Map IDs to indices in the processed arrays
        indices = [id_to_idx[i] for i in meta_ids]

        # Slice arrays
        X_cont = continuous_all[indices]
        X_seq = sequence_all[indices]

        if is_test:
            y = None
        else:
            y = targets_all[indices]

        return ManufacturingDataset(X_cont, X_seq, y)

    print("Creating datasets...")
    train_dataset = create_dataset_from_meta(Config.TRAIN_METADATA, is_test=False)
    val_dataset = create_dataset_from_meta(Config.VAL_METADATA, is_test=False)
    test_dataset = create_dataset_from_meta(Config.TEST_METADATA, is_test=True)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    # 3. Create DataLoaders
    # Use num_workers=0 or higher depending on system. Given 12 vCPUs, 4 is safe.
    # Pin memory for GPU transfer.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Good for training stability
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

    return train_loader, val_loader, test_loader
