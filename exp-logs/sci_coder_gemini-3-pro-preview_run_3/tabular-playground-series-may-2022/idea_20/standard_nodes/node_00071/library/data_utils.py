import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Separates continuous and categorical features for the SDPE architecture.
    """

    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64)

        self.target_col = target_col
        if target_col and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

        self.ids = df["id"].values if "id" in df.columns else None

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        item = {
            "x_cont": torch.tensor(self.cont_data[idx], dtype=torch.float32),
            "x_cat": torch.tensor(self.cat_data[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def preprocess_pipeline(load_cached_data=True):
    """
    Loads data, performs feature engineering, transductive encoding, and normalization.
    Implements caching using Parquet and NPY files.
    """
    # Ensure working directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN)
        and os.path.exists(Config.CACHE_VAL)
        and os.path.exists(Config.CACHE_TEST)
        and os.path.exists(Config.CACHE_METADATA)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        df_train = pd.read_parquet(Config.CACHE_TRAIN)
        df_val = pd.read_parquet(Config.CACHE_VAL)
        df_test = pd.read_parquet(Config.CACHE_TEST)

        metadata = np.load(Config.CACHE_METADATA, allow_pickle=True).item()
        vocab_sizes = [int(v) for v in metadata["vocab_sizes"]]
        cat_cols = metadata["cat_cols"]
        cont_cols = metadata["cont_cols"]

        return df_train, df_val, df_test, vocab_sizes, cat_cols, cont_cols

    print("Processing data from scratch...")

    # Load metadata splits
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Define feature groups
    # f_27 is string, f_29 and f_30 are categorical integers
    # All others are continuous
    base_cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]

    # --- Feature Engineering ---
    def process_df(df):
        # 1. Decompose f_27
        # Split string into list of characters, then to columns
        chars = df["f_27"].apply(list).tolist()
        char_df = pd.DataFrame(
            chars, columns=[f"char_{i}" for i in range(10)], index=df.index
        )

        # 2. Unique character count
        df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))

        # Merge
        df = pd.concat([df, char_df], axis=1)
        return df

    df_train = process_df(df_train)
    df_val = process_df(df_val)
    df_test = process_df(df_test)

    # Define Column Groups
    char_cols = [f"char_{i}" for i in range(10)]
    # f_29, f_30 are categorical
    cat_cols = char_cols + ["f_29", "f_30"]
    cont_cols = base_cont_cols + ["unique_char_count"]

    # --- Transductive Vocabulary Alignment ---
    # Fit OrdinalEncoder on Train + Val + Test
    print("Fitting transductive OrdinalEncoder...")
    full_cat_data = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(full_cat_data)

    # Transform
    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_val[cat_cols] = encoder.transform(df_val[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    # Calculate vocab sizes (max index + 1 for embedding layer)
    # Since we fit on all data, max index is unique_values - 1.
    # Vocab size = number of unique categories.
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # --- Normalization ---
    # Fit StandardScaler on Train ONLY
    print("Fitting StandardScaler on Training set...")
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    df_train[cont_cols] = scaler.transform(df_train[cont_cols])
    df_val[cont_cols] = scaler.transform(df_val[cont_cols])
    df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    # Drop f_27 as it is now decomposed
    df_train = df_train.drop(columns=["f_27"])
    df_val = df_val.drop(columns=["f_27"])
    df_test = df_test.drop(columns=["f_27"])

    # --- Caching ---
    print("Saving processed data to cache...")
    df_train.to_parquet(Config.CACHE_TRAIN, index=False)
    df_val.to_parquet(Config.CACHE_VAL, index=False)
    df_test.to_parquet(Config.CACHE_TEST, index=False)

    metadata = {
        "vocab_sizes": vocab_sizes,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
    }
    np.save(Config.CACHE_METADATA, metadata)

    return df_train, df_val, df_test, vocab_sizes, cat_cols, cont_cols


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates the data pipeline and returns PyTorch DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, vocab_sizes
    """
    # Run pipeline
    df_train, df_val, df_test, vocab_sizes, cat_cols, cont_cols = preprocess_pipeline(
        load_cached_data
    )

    # Create Datasets
    train_dataset = ManufacturingDataset(
        df_train, cat_cols, cont_cols, target_col="target"
    )
    val_dataset = ManufacturingDataset(df_val, cat_cols, cont_cols, target_col="target")
    test_dataset = ManufacturingDataset(df_test, cat_cols, cont_cols, target_col=None)

    # Create DataLoaders
    # Use drop_last=True for train to maintain batch statistics consistency
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
