import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManufacturingDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col=None, is_test=False):
        self.cat_features = df[cat_cols].values.astype(np.int64)
        self.cont_features = df[cont_cols].values.astype(np.float32)

        if not is_test and target_col in df.columns:
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


def process_f27(df):
    """
    Decomposes f_27 into 10 character columns and adds unique_character_count.
    """
    # 1. Unique character count
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

    # 2. Decompose string into columns
    # Assuming fixed length of 10 based on Config.F27_SEQ_LEN or data analysis
    for i in range(10):
        df[f"f_27_{i}"] = df["f_27"].str[i]

    return df


def preprocess_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, encoding, and scaling.
    Implements caching to speed up subsequent runs.
    """
    set_seed(Config.SEED)

    cache_dir = Config.CACHE_DIR
    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    meta_cache = os.path.join(cache_dir, "metadata.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        # print("Loading processed data from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()
        return df_train, df_val, df_test, metadata

    # print("Processing data from scratch...")

    # Load raw data
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Feature Engineering
    for df in [df_train, df_val, df_test]:
        # Decompose f_27
        process_f27(df)

        # Cast f_29 and f_30 to string to treat as categorical
        df["f_29"] = df["f_29"].astype(str)
        df["f_30"] = df["f_30"].astype(str)

    # Define Column Groups
    # Categorical: decomposed f_27 chars + f_29 + f_30
    cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00..f_26, f_28, unique_character_count
    # Exclude f_27 (string) and target/id/source_path
    exclude_cols = ["id", "target", "source_path", "f_27"] + cat_cols
    cont_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Transductive Vocabulary Alignment (Fit on All)
    # Concatenate for fitting encoder
    all_cat_data = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    encoder.fit(all_cat_data)

    # Transform
    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_val[cat_cols] = encoder.transform(df_val[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    # Calculate vocab sizes (max index + 1)
    # We add 1 to ensure coverage. Since we fit on all data, max index is known.
    vocab_sizes = [int(all_cat_data[col].nunique()) for col in cat_cols]

    # Normalization (Fit on Train Only)
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    df_train[cont_cols] = scaler.transform(df_train[cont_cols])
    df_val[cont_cols] = scaler.transform(df_val[cont_cols])
    df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    # Prepare Metadata
    metadata = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "vocab_sizes": vocab_sizes,
        "num_cont_features": len(cont_cols),
    }

    # Save to Cache
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)
    np.save(meta_cache, metadata)

    return df_train, df_val, df_test, metadata


def get_dataloaders(load_cached_data=True, verbose=False):
    """
    Orchestrates data loading and DataLoader creation.
    """
    df_train, df_val, df_test, metadata = preprocess_data(
        load_cached_data=load_cached_data
    )

    cat_cols = metadata["cat_cols"]
    cont_cols = metadata["cont_cols"]

    # Subsampling for debugging
    if Config.MAX_SAMPLES is not None:
        if verbose:
            print(f"Subsampling training data to {Config.MAX_SAMPLES} samples...")
        df_train = df_train.iloc[: Config.MAX_SAMPLES].copy()
        # We usually keep validation full or subsample proportionally, but for debug fixed size is fine.
        df_val = df_val.iloc[: min(len(df_val), Config.MAX_SAMPLES)].copy()

    # Create Datasets
    train_dataset = ManufacturingDataset(
        df_train, cat_cols, cont_cols, target_col=Config.TARGET_COL, is_test=False
    )
    val_dataset = ManufacturingDataset(
        df_val, cat_cols, cont_cols, target_col=Config.TARGET_COL, is_test=False
    )
    test_dataset = ManufacturingDataset(
        df_test, cat_cols, cont_cols, target_col=None, is_test=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for training stability with batch norm/dropout
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

    return train_loader, val_loader, test_loader, metadata
