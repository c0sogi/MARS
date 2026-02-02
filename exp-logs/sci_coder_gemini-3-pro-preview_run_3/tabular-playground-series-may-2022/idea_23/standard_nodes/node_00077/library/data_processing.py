import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    """

    def __init__(self, x_cont, x_cat, ids, targets=None):
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cat = torch.LongTensor(x_cat)
        self.ids = torch.LongTensor(ids)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        item = {
            "x_cont": self.x_cont[idx],
            "x_cat": self.x_cat[idx],
            "id": self.ids[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _process_f27(df):
    """
    Decomposes f_27 into character columns and computes unique character count.
    """
    # 1. Compute unique character count (Continuous Feature)
    unique_counts = df["f_27"].apply(lambda x: len(set(x))).values.reshape(-1, 1)

    # 2. Decompose string into fixed position characters (Categorical Features)
    # Assumes f_27 is always length 10 based on Config.F27_SEQ_LEN
    # We use a vectorized approach by converting the series of strings to a list of lists
    char_matrix = np.array(df["f_27"].apply(list).tolist())

    # Create DataFrame for the characters
    char_cols = [f"f_27_{i}" for i in range(char_matrix.shape[1])]
    df_chars = pd.DataFrame(char_matrix, columns=char_cols, index=df.index)

    return df_chars, unique_counts, char_cols


def prepare_data(load_cached_data=True):
    """
    Loads, preprocesses, and batches the data.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        train_loader, val_loader, test_loader, vocab_sizes
    """

    # 1. Caching Logic
    cache_files_exist = (
        os.path.exists(Config.TRAIN_PROCESSED_PATH)
        and os.path.exists(Config.VAL_PROCESSED_PATH)
        and os.path.exists(Config.TEST_PROCESSED_PATH)
        and os.path.exists(Config.METADATA_CACHE_PATH)
    )

    if load_cached_data and cache_files_exist:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(Config.TRAIN_PROCESSED_PATH)
        val_df = pd.read_parquet(Config.VAL_PROCESSED_PATH)
        test_df = pd.read_parquet(Config.TEST_PROCESSED_PATH)
        vocab_sizes = np.load(Config.METADATA_CACHE_PATH, allow_pickle=True)

        # Identify columns from loaded data
        # We assume the last column is target (for train/val) and id is present
        # We need to reconstruct the column lists based on naming conventions
        cat_cols = [c for c in train_df.columns if c.startswith("cat_")]
        cont_cols = [c for c in train_df.columns if c.startswith("cont_")]

    else:
        print("Processing data from scratch...")

        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Debug: Subsample if configured
        if Config.DEBUG:
            print(f"DEBUG MODE: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")
            train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
            val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

        # --- Feature Engineering ---

        # 1. Process f_27 for all sets
        print("Engineering features from f_27...")
        train_chars, train_unique, char_col_names = _process_f27(train_df)
        val_chars, val_unique, _ = _process_f27(val_df)
        test_chars, test_unique, _ = _process_f27(test_df)

        # Add unique count to dataframes temporarily to group continuous vars
        train_df["unique_char_count"] = train_unique
        val_df["unique_char_count"] = val_unique
        test_df["unique_char_count"] = test_unique

        # Add char columns to dataframes
        train_df = pd.concat([train_df, train_chars], axis=1)
        val_df = pd.concat([val_df, val_chars], axis=1)
        test_df = pd.concat([test_df, test_chars], axis=1)

        # 2. Define Column Groups
        # Continuous: f_00..f_26, f_28, unique_char_count
        # Note: f_27 is excluded (replaced by components), f_29/f_30 are categorical
        raw_cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
            "unique_char_count"
        ]

        # Categorical: f_27_0..f_27_9, f_29, f_30
        # f_29 and f_30 are treated as categorical as per strategy
        raw_cat_cols = char_col_names + ["f_29", "f_30"]

        # --- Transductive Vocabulary Alignment ---
        print("Performing Transductive Ordinal Encoding...")

        # Concatenate all categorical data to fit encoder
        # Ensure f_29 and f_30 are treated as strings/objects to match char columns for safety,
        # though OrdinalEncoder handles mixed types, consistency is good.
        # We convert all cat cols to string to ensure robust encoding of discrete integers.
        for col in raw_cat_cols:
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

        all_cat_data = pd.concat(
            [train_df[raw_cat_cols], val_df[raw_cat_cols], test_df[raw_cat_cols]],
            axis=0,
        )

        encoder = OrdinalEncoder(dtype=np.int64)
        encoder.fit(all_cat_data)

        # Transform datasets
        train_df[raw_cat_cols] = encoder.transform(train_df[raw_cat_cols])
        val_df[raw_cat_cols] = encoder.transform(val_df[raw_cat_cols])
        test_df[raw_cat_cols] = encoder.transform(test_df[raw_cat_cols])

        # Calculate vocab sizes (max index + 1) for each column
        # Since we fit on all data, max index is consistent
        vocab_sizes = [int(all_cat_data[col].nunique()) for col in raw_cat_cols]
        vocab_sizes = np.array(vocab_sizes)  # Save as numpy array

        # --- Scaling Continuous Features ---
        print("Scaling continuous features...")
        scaler = StandardScaler()

        # Fit on TRAIN only
        scaler.fit(train_df[raw_cont_cols])

        # Transform all
        train_df[raw_cont_cols] = scaler.transform(train_df[raw_cont_cols])
        val_df[raw_cont_cols] = scaler.transform(val_df[raw_cont_cols])
        test_df[raw_cont_cols] = scaler.transform(test_df[raw_cont_cols])

        # --- Rename and Reorganize for Cache/Loader ---
        # We prefix columns to easily identify them after loading from parquet

        # Helper to rename and select
        def organize_df(df, is_test=False):
            new_df = pd.DataFrame()
            new_df["id"] = df["id"]

            if not is_test:
                new_df["target"] = df["target"]

            # Add Continuous
            for i, col in enumerate(raw_cont_cols):
                new_df[f"cont_{i}"] = df[col]

            # Add Categorical
            for i, col in enumerate(raw_cat_cols):
                new_df[f"cat_{i}"] = df[col]

            return new_df

        train_df = organize_df(train_df)
        val_df = organize_df(val_df)
        test_df = organize_df(test_df, is_test=True)

        # Update column lists for the loader creation step
        cont_cols = [c for c in train_df.columns if c.startswith("cont_")]
        cat_cols = [c for c in train_df.columns if c.startswith("cat_")]

        # --- Save to Cache ---
        print("Saving processed data to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        train_df.to_parquet(Config.TRAIN_PROCESSED_PATH, index=False)
        val_df.to_parquet(Config.VAL_PROCESSED_PATH, index=False)
        test_df.to_parquet(Config.TEST_PROCESSED_PATH, index=False)
        np.save(Config.METADATA_CACHE_PATH, vocab_sizes)

    # 3. Create DataLoaders
    print("Creating DataLoaders...")

    # Extract arrays
    def get_arrays(df, has_target=True):
        x_cont = df[cont_cols].values.astype(np.float32)
        x_cat = df[cat_cols].values.astype(np.int64)
        ids = df["id"].values.astype(np.int64)
        y = df["target"].values.astype(np.float32) if has_target else None
        return x_cont, x_cat, ids, y

    x_cont_train, x_cat_train, ids_train, y_train = get_arrays(
        train_df, has_target=True
    )
    x_cont_val, x_cat_val, ids_val, y_val = get_arrays(val_df, has_target=True)
    x_cont_test, x_cat_test, ids_test, _ = get_arrays(test_df, has_target=False)

    # Create Datasets
    train_dataset = ManufacturingDataset(x_cont_train, x_cat_train, ids_train, y_train)
    val_dataset = ManufacturingDataset(x_cont_val, x_cat_val, ids_val, y_val)
    test_dataset = ManufacturingDataset(x_cont_test, x_cat_test, ids_test, targets=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, vocab_sizes
