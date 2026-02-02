import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config

# Set fixed seeds for reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Separates continuous and categorical features for the Input-Attentive model.
    """

    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cont_features = df[cont_cols].values.astype(np.float32)
        self.cat_features = df[cat_cols].values.astype(np.int64)

        if target_col and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        x_cont = torch.tensor(self.cont_features[idx], dtype=torch.float32)
        x_cat = torch.tensor(self.cat_features[idx], dtype=torch.long)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_cont, x_cat, y
        else:
            return x_cont, x_cat


def decompose_f27(df):
    """
    Splits the 'f_27' string column into 10 separate character columns.
    """
    # f_27 is a string of length 10. We split it into 10 columns.
    # Using apply(list) creates a list of chars, then we expand to columns.
    chars = pd.DataFrame(df["f_27"].apply(list).tolist(), index=df.index)
    chars.columns = [f"char_{i}" for i in range(10)]
    return chars


def get_unique_char_count(df):
    """
    Calculates the number of unique characters in the 'f_27' string.
    """
    return df["f_27"].apply(lambda x: len(set(x)))


def process_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, encoding, and scaling.
    Implements caching mechanism using Parquet files.
    """

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_files_exist = (
        os.path.exists(Config.TRAIN_CACHE)
        and os.path.exists(Config.VAL_CACHE)
        and os.path.exists(Config.TEST_CACHE)
        and os.path.exists(Config.METADATA_CACHE)
    )

    if load_cached_data and cache_files_exist:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(Config.TRAIN_CACHE)
        val_df = pd.read_parquet(Config.VAL_CACHE)
        test_df = pd.read_parquet(Config.TEST_CACHE)
        metadata = np.load(Config.METADATA_CACHE, allow_pickle=True).item()
        return train_df, val_df, test_df, metadata

    print("Processing data from scratch...")

    # Load raw data from metadata directory
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Feature Engineering Wrapper
    def engineer_features(df):
        # Decompose f_27
        chars_df = decompose_f27(df)
        df = pd.concat([df, chars_df], axis=1)

        # Unique character count
        df["unique_char_count"] = get_unique_char_count(df)
        return df

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Define Feature Groups
    # Categorical: decomposed chars + f_29 + f_30
    char_cols = [f"char_{i}" for i in range(10)]
    cat_cols = char_cols + ["f_29", "f_30"]

    # Continuous: f_00..f_28 (excluding f_27) + unique_char_count
    # f_29 and f_30 are treated as categorical based on strategy
    raw_cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_cols = raw_cont_cols + ["unique_char_count"]

    # Transductive Ordinal Encoding
    # Fit on ALL data to ensure common vocabulary and handle test tokens
    print("Fitting OrdinalEncoder on Train + Val + Test...")
    encoder = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )

    # Concatenate for fitting
    all_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )

    encoder.fit(all_cat)

    # Transform
    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    val_df[cat_cols] = encoder.transform(val_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Calculate Vocab Sizes (max index + 1)
    # Since we fit on all data, max index in encoder categories is sufficient
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # Standard Scaling
    # Fit ONLY on Train
    print("Fitting StandardScaler on Train...")
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    # Transform
    train_df[cont_cols] = scaler.transform(train_df[cont_cols])
    val_df[cont_cols] = scaler.transform(val_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    # Metadata
    metadata = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "vocab_sizes": vocab_sizes,
        "num_continuous": len(cont_cols),
    }

    # Save to Cache
    print("Saving processed data to cache...")
    train_df.to_parquet(Config.TRAIN_CACHE, index=False)
    val_df.to_parquet(Config.VAL_CACHE, index=False)
    test_df.to_parquet(Config.TEST_CACHE, index=False)
    np.save(Config.METADATA_CACHE, metadata)

    return train_df, val_df, test_df, metadata


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Main entry point. Processes data and returns PyTorch DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, num_continuous, vocab_sizes
    """
    train_df, val_df, test_df, metadata = process_data(
        load_cached_data=load_cached_data
    )

    cat_cols = metadata["cat_cols"]
    cont_cols = metadata["cont_cols"]
    vocab_sizes = metadata["vocab_sizes"]
    num_continuous = metadata["num_continuous"]

    # Create Datasets
    train_dataset = ManufacturingDataset(
        train_df, cat_cols, cont_cols, target_col="target"
    )
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, target_col="target")
    test_dataset = ManufacturingDataset(
        test_df, cat_cols, cont_cols, target_col=None
    )  # No target in test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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

    return train_loader, val_loader, test_loader, num_continuous, vocab_sizes
