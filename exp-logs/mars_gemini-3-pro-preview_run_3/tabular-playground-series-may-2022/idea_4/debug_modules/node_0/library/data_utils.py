import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config


class ManufacturingDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.is_test = is_test

        # Extract continuous features
        self.cont_features = df[Config.CONT_FEATURES].values.astype(np.float32)

        # Extract categorical features
        self.cat_features = df[Config.CAT_FEATURES].values.astype(np.int64)

        # Extract target if not test set
        if not self.is_test:
            self.targets = df[Config.TARGET_COL].values.astype(np.float32)

        # Extract IDs for submission
        self.ids = df[Config.ID_COL].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            "cont_features": torch.tensor(self.cont_features[idx]),
            "cat_features": torch.tensor(self.cat_features[idx]),
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx])

        return item


def decompose_string_features(df):
    """
    Decomposes f_27 into character columns and computes unique character count.
    """
    # Ensure f_27 is string
    s = df["f_27"].astype(str)

    # 1. Compute unique_char_count
    df["unique_char_count"] = s.apply(lambda x: len(set(x)))

    # 2. Split f_27 into 10 separate columns
    # We assume fixed length of 10 based on problem description/EDA
    for i in range(10):
        df[f"f_27_{i}"] = s.str[i]

    return df


def get_dataloaders(load_cached_data=True):
    """
    Loads data, processes features, and returns DataLoaders.
    Implements caching using Parquet for data and NPY for metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    vocab_sizes_path = os.path.join(Config.WORKING_DIR, "vocab_sizes.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.TRAIN_PROCESSED_PATH)
        and os.path.exists(Config.VAL_PROCESSED_PATH)
        and os.path.exists(Config.TEST_PROCESSED_PATH)
        and os.path.exists(vocab_sizes_path)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(Config.TRAIN_PROCESSED_PATH)
        val_df = pd.read_parquet(Config.VAL_PROCESSED_PATH)
        test_df = pd.read_parquet(Config.TEST_PROCESSED_PATH)
        vocab_sizes = np.load(vocab_sizes_path, allow_pickle=True).item()

    else:
        print("Processing data from scratch...")

        # Load raw data from metadata splits
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Feature Engineering
        print("Applying feature engineering...")
        train_df = decompose_string_features(train_df)
        val_df = decompose_string_features(val_df)
        test_df = decompose_string_features(test_df)

        # --- Categorical Encoding ---
        print("Encoding categorical features...")
        # We use OrdinalEncoder. handle_unknown='use_encoded_value' with -1 allows us to
        # detect unknown categories in Val/Test and map them to a specific index (0).
        # We shift all indices by +1, so 0 is reserved for unknown, and 1..N are known.
        encoder = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        )

        # Fit on Train only
        encoder.fit(train_df[Config.CAT_FEATURES])

        # Transform all sets
        train_df[Config.CAT_FEATURES] = encoder.transform(train_df[Config.CAT_FEATURES])
        val_df[Config.CAT_FEATURES] = encoder.transform(val_df[Config.CAT_FEATURES])
        test_df[Config.CAT_FEATURES] = encoder.transform(test_df[Config.CAT_FEATURES])

        # Shift indices by +1 to handle unknowns (which are -1)
        for col in Config.CAT_FEATURES:
            train_df[col] = train_df[col] + 1
            val_df[col] = val_df[col] + 1
            test_df[col] = test_df[col] + 1

        # Calculate vocab sizes (max index + 1 for safety)
        vocab_sizes = {}
        for i, col in enumerate(Config.CAT_FEATURES):
            # Max index in train + 1 (for 0-based indexing) + 1 (buffer/unknown)
            # Since we shifted by 1, max index is effectively len(categories).
            # We use the number of categories found during fit + 1 (for index 0).
            vocab_sizes[col] = len(encoder.categories_[i]) + 1

        # --- Continuous Normalization ---
        print("Normalizing continuous features...")
        scaler = StandardScaler()

        # Fit on Train only
        scaler.fit(train_df[Config.CONT_FEATURES])

        # Transform all sets
        train_df[Config.CONT_FEATURES] = scaler.transform(
            train_df[Config.CONT_FEATURES]
        )
        val_df[Config.CONT_FEATURES] = scaler.transform(val_df[Config.CONT_FEATURES])
        test_df[Config.CONT_FEATURES] = scaler.transform(test_df[Config.CONT_FEATURES])

        # --- Save to Cache ---
        print("Saving processed data to cache...")
        train_df.to_parquet(Config.TRAIN_PROCESSED_PATH, index=False)
        val_df.to_parquet(Config.VAL_PROCESSED_PATH, index=False)
        test_df.to_parquet(Config.TEST_PROCESSED_PATH, index=False)
        np.save(vocab_sizes_path, vocab_sizes)

    # Create Datasets
    train_dataset = ManufacturingDataset(train_df, is_test=False)
    val_dataset = ManufacturingDataset(val_df, is_test=False)
    test_dataset = ManufacturingDataset(test_df, is_test=True)

    # Create DataLoaders
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

    return train_loader, val_loader, test_loader, vocab_sizes
