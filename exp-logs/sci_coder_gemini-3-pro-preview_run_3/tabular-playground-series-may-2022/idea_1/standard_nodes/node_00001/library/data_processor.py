import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    CONTINUOUS_FEATURES,
    DISCRETE_FEATURES,
    STRING_COL,
    TARGET_COL,
    ID_COL,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    SEQUENCE_LENGTH,
)
from library.utils import seed_everything


class ManufacturingDataset(Dataset):
    def __init__(
        self, continuous_data, categorical_data, targets=None, vocab_sizes=None
    ):
        self.continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
        self.categorical_data = torch.tensor(categorical_data, dtype=torch.long)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        )

        # Calculate vocab sizes for each categorical feature (max index + 1)
        # This is useful for defining embedding layers in the model
        if vocab_sizes is not None:
            self.vocab_sizes = vocab_sizes
        elif self.categorical_data.numel() > 0:
            self.vocab_sizes = (self.categorical_data.max(dim=0)[0] + 1).tolist()
        else:
            self.vocab_sizes = []

    def __len__(self):
        return len(self.continuous_data)

    def __getitem__(self, idx):
        item = {
            "continuous": self.continuous_data[idx],
            "categorical": self.categorical_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def _split_string_feature(df, col_name):
    """Splits a string column into multiple character columns."""
    # Convert string column to list of characters and then to DataFrame
    chars = np.array([list(s) for s in df[col_name].values])
    # Create column names
    cols = [f"{col_name}_char_{i}" for i in range(chars.shape[1])]
    return pd.DataFrame(chars, columns=cols, index=df.index)


def _process_data_from_scratch():
    """Loads raw data, performs feature engineering, scaling, and encoding."""
    print("Processing data from scratch...")

    # Load raw CSVs
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Feature Engineering: Split f_27
    # We do this for all splits
    train_chars = _split_string_feature(train_df, STRING_COL)
    val_chars = _split_string_feature(val_df, STRING_COL)
    test_chars = _split_string_feature(test_df, STRING_COL)

    # Identify new categorical columns (character columns)
    char_cols = train_chars.columns.tolist()

    # Concatenate features back to main df (dropping original string col)
    train_df = pd.concat([train_df.drop(columns=[STRING_COL]), train_chars], axis=1)
    val_df = pd.concat([val_df.drop(columns=[STRING_COL]), val_chars], axis=1)
    test_df = pd.concat([test_df.drop(columns=[STRING_COL]), test_chars], axis=1)

    # Define all categorical features (original discrete + new chars)
    # Ensure order is preserved
    all_cat_features = DISCRETE_FEATURES + char_cols

    # 1. Categorical Encoding
    # Fit OrdinalEncoder on ALL data to ensure we capture the full vocabulary
    # and maintain consistent mapping.
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )

    # Combine for fitting
    combined_cats = pd.concat(
        [
            train_df[all_cat_features],
            val_df[all_cat_features],
            test_df[all_cat_features],
        ],
        axis=0,
    )

    encoder.fit(combined_cats)

    train_df[all_cat_features] = encoder.transform(train_df[all_cat_features])
    val_df[all_cat_features] = encoder.transform(val_df[all_cat_features])
    test_df[all_cat_features] = encoder.transform(test_df[all_cat_features])

    # Handle any potential unknowns (though fitting on all prevents this for existing values)
    # We shift indices by +1 if we wanted to reserve 0 for padding/unknown,
    # but here we just ensure non-negative.
    # OrdinalEncoder outputs 0..N-1.

    # 2. Continuous Scaling
    # Fit StandardScaler ONLY on Training data
    scaler = StandardScaler()
    scaler.fit(train_df[CONTINUOUS_FEATURES])

    train_df[CONTINUOUS_FEATURES] = scaler.transform(train_df[CONTINUOUS_FEATURES])
    val_df[CONTINUOUS_FEATURES] = scaler.transform(val_df[CONTINUOUS_FEATURES])
    test_df[CONTINUOUS_FEATURES] = scaler.transform(test_df[CONTINUOUS_FEATURES])

    # Ensure ID is preserved and Target is handled
    # We return the full dataframes; splitting into X/y happens in Dataset init

    return train_df, val_df, test_df, all_cat_features


def make_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True):
    """
    Orchestrates data loading, processing/caching, and DataLoader creation.

    Args:
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(SEED)

    # Define cache paths
    train_cache = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_processed.parquet")
    meta_cache = os.path.join(CACHE_DIR, "metadata.npy")  # To store feature names list

    data_loaded = False

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(meta_cache)
        ):
            print("Loading data from cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                all_cat_features = np.load(meta_cache, allow_pickle=True).tolist()
                data_loaded = True
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

    # Process if not loaded
    if not data_loaded:
        train_df, val_df, test_df, all_cat_features = _process_data_from_scratch()

        # Save to cache
        print("Saving processed data to cache...")
        os.makedirs(CACHE_DIR, exist_ok=True)
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)
        np.save(meta_cache, np.array(all_cat_features))

    # Prepare Dataset objects
    # Extract numpy arrays for speed

    # Train
    train_cont = train_df[CONTINUOUS_FEATURES].values
    train_cat = train_df[all_cat_features].values
    train_y = train_df[TARGET_COL].values

    # Val
    val_cont = val_df[CONTINUOUS_FEATURES].values
    val_cat = val_df[all_cat_features].values
    val_y = val_df[TARGET_COL].values

    # Test (No target)
    test_cont = test_df[CONTINUOUS_FEATURES].values
    test_cat = test_df[all_cat_features].values
    # Test set might not have target, or we don't use it. Pass None or dummy.
    # The submission format requires ID, so we might want to keep track of IDs.
    # But the Dataset class structure defined usually returns features/target.
    # We will handle IDs in the prediction loop by reading the ID column separately or
    # ensuring the test loader preserves order (which it does).

    # Calculate global vocab sizes to ensure embeddings cover all splits
    # We use the max index across all datasets because the OrdinalEncoder was fitted globally.
    all_cats_combined = np.concatenate([train_cat, val_cat, test_cat], axis=0)
    vocab_sizes = (np.max(all_cats_combined, axis=0) + 1).tolist()

    train_dataset = ManufacturingDataset(
        train_cont, train_cat, train_y, vocab_sizes=vocab_sizes
    )
    val_dataset = ManufacturingDataset(
        val_cont, val_cat, val_y, vocab_sizes=vocab_sizes
    )
    test_dataset = ManufacturingDataset(
        test_cont, test_cat, targets=None, vocab_sizes=vocab_sizes
    )

    # Attach ID information to test dataset for convenience if needed,
    # though usually we just iterate the loader and the original ID list in parallel.
    test_dataset.ids = test_df[ID_COL].values

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
