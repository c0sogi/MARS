import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config

# --------------------------------------------------------------------------
# Feature Engineering Helpers
# --------------------------------------------------------------------------


def decompose_f27(df):
    """
    Splits the 'f_27' string column into 10 separate character columns.
    """
    # f_27 is a string of length 10. We extract each character position.
    for i in range(10):
        df[f"f_27_{i}"] = df["f_27"].str[i]
    return df


def add_unique_count(df):
    """
    Adds a feature counting the number of unique characters in 'f_27'.
    """
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))
    return df


# --------------------------------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------------------------------


def preprocess_pipeline(load_cached_data=True):
    """
    Loads, cleans, encodes, and normalizes data.
    Implements Transductive Vocabulary Alignment and Caching.

    Returns:
        train_df (pd.DataFrame): Processed training data.
        val_df (pd.DataFrame): Processed validation data.
        test_df (pd.DataFrame): Processed test data.
        metadata (dict): Contains 'vocab_sizes', 'cat_cols', 'cont_cols'.
    """
    Config.create_directories()

    # Check for cached files
    files_exist = (
        os.path.exists(Config.TRAIN_CACHE_PATH)
        and os.path.exists(Config.VAL_CACHE_PATH)
        and os.path.exists(Config.TEST_CACHE_PATH)
        and os.path.exists(Config.METADATA_CACHE_PATH)
    )

    if load_cached_data and files_exist:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(Config.TRAIN_CACHE_PATH)
        val_df = pd.read_parquet(Config.VAL_CACHE_PATH)
        test_df = pd.read_parquet(Config.TEST_CACHE_PATH)
        metadata = np.load(Config.METADATA_CACHE_PATH, allow_pickle=True).item()
        return train_df, val_df, test_df, metadata

    print("Processing data from scratch...")

    # Load raw data using metadata paths
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # 1. Feature Engineering
    for df in [train_df, val_df, test_df]:
        df = decompose_f27(df)
        df = add_unique_count(df)

    # Define Column Groups
    # Categorical: f_27 decomposed chars + f_29 + f_30
    cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00 to f_26, f_28, unique_character_count
    # Note: f_27 is excluded (replaced), f_29/f_30 are categorical
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]
    # Remove f_29, f_30 from cont_cols if they were accidentally included by range
    cont_cols = [c for c in cont_cols if c not in ["f_29", "f_30"]]

    # 2. Transductive Vocabulary Alignment (Ordinal Encoding)
    # Fit on Train + Val + Test to ensure all tokens are handled
    print("Fitting Transductive Ordinal Encoder...")
    all_cats = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(all_cats)

    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    val_df[cat_cols] = encoder.transform(val_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Calculate vocab sizes for embeddings (max index + 1)
    # We use the encoder categories to be precise
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # 3. Normalization (StandardScaler)
    # Fit ONLY on Train, transform all
    print("Fitting StandardScaler on Training data...")
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    train_df[cont_cols] = scaler.transform(train_df[cont_cols])
    val_df[cont_cols] = scaler.transform(val_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    # Cast to appropriate types to save memory/ensure compatibility
    for col in cont_cols:
        train_df[col] = train_df[col].astype(np.float32)
        val_df[col] = val_df[col].astype(np.float32)
        test_df[col] = test_df[col].astype(np.float32)

    for col in cat_cols:
        train_df[col] = train_df[col].astype(np.int64)
        val_df[col] = val_df[col].astype(np.int64)
        test_df[col] = test_df[col].astype(np.int64)

    # Prepare Metadata
    metadata = {
        "vocab_sizes": vocab_sizes,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
    }

    # 4. Caching
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    train_df.to_parquet(Config.TRAIN_CACHE_PATH, index=False)
    val_df.to_parquet(Config.VAL_CACHE_PATH, index=False)
    test_df.to_parquet(Config.TEST_CACHE_PATH, index=False)
    np.save(Config.METADATA_CACHE_PATH, metadata)

    return train_df, val_df, test_df, metadata


# --------------------------------------------------------------------------
# Dataset Class
# --------------------------------------------------------------------------


class ManufacturingDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cat_features = df[cat_cols].values
        self.cont_features = df[cont_cols].values

        if target_col and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        item = {
            "cat_features": torch.tensor(self.cat_features[idx], dtype=torch.long),
            "cont_features": torch.tensor(self.cont_features[idx], dtype=torch.float32),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


# --------------------------------------------------------------------------
# DataLoader Factory
# --------------------------------------------------------------------------


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Orchestrates preprocessing and returns PyTorch DataLoaders.
    """
    train_df, val_df, test_df, metadata = preprocess_pipeline(
        load_cached_data=load_cached_data
    )

    cat_cols = metadata["cat_cols"]
    cont_cols = metadata["cont_cols"]
    vocab_sizes = metadata["vocab_sizes"]

    # Create Datasets
    train_dataset = ManufacturingDataset(
        train_df, cat_cols, cont_cols, target_col="target"
    )
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, target_col="target")
    test_dataset = ManufacturingDataset(test_df, cat_cols, cont_cols, target_col=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return train_loader, val_loader, test_loader, metadata
