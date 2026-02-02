import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config
from library.utils import set_seed


class ManufacturingDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.targets = (
            df[target_col].values.astype(np.float32)
            if target_col in df.columns
            else None
        )

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        cat = torch.tensor(self.cat_data[idx], dtype=torch.long)
        cont = torch.tensor(self.cont_data[idx], dtype=torch.float32)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return cat, cont, target
        return cat, cont


def engineer_features(df):
    """
    Performs feature engineering:
    1. Calculates unique character count in f_27.
    2. Decomposes f_27 into 10 separate character columns.
    3. Drops original f_27.
    """
    if "f_27" not in df.columns:
        return df

    # Unique character count
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

    # Decompose f_27 into 10 columns
    # Using list comprehension and DataFrame constructor is efficient
    chars = df["f_27"].apply(list)
    char_df = pd.DataFrame(
        chars.tolist(), columns=[f"f_27_{i}" for i in range(10)], index=df.index
    )

    # Drop original and concat
    df = df.drop(columns=["f_27"])
    df = pd.concat([df, char_df], axis=1)

    return df


def preprocess_data(load_cached_data=True, max_samples=None, config=Config):
    """
    Loads, processes, and caches data.
    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        max_samples (int, optional): Number of samples to load for debugging.
        config (class): Configuration class containing paths and settings.
    Returns:
        tuple: (df_train, df_val, df_test, metadata)
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(config.CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(config.CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(config.CACHE_DIR, "test_processed.parquet")
    meta_cache = os.path.join(config.CACHE_DIR, "metadata.npy")

    # Check cache availability
    # If max_samples is set, we bypass cache to ensure we get the correct subset/fresh data
    use_cache = (
        load_cached_data
        and max_samples is None
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    )

    if use_cache:
        print("Loading cached data...")
        metadata = np.load(meta_cache, allow_pickle=True).item()

        # Cite debug_lesson_1: Validate schema compatibility when loading cached data
        required_keys = ["cat_cols", "cont_cols", "vocab_sizes"]
        if all(k in metadata for k in required_keys):
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)
            return df_train, df_val, df_test, metadata
        else:
            print("Cached metadata missing required keys. Reprocessing...")

    print("Processing data from scratch...")
    # Load raw data
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_val = pd.read_csv(config.VAL_PATH)
    df_test = pd.read_csv(config.TEST_PATH)

    # Debugging: Subsample if requested
    if max_samples is not None:
        df_train = df_train.iloc[:max_samples]
        df_val = df_val.iloc[:max_samples]
        df_test = df_test.iloc[:max_samples]

    # Feature Engineering
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Identify Columns
    # Categorical: f_29, f_30, and decomposed f_27_*
    cat_cols = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]

    # Continuous: All others except metadata columns
    exclude_cols = ["id", "target", "source_path"] + cat_cols
    cont_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Transductive Encoding
    # Fit on all available data to ensure vocabulary consistency
    full_cat = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    encoder.fit(full_cat)

    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_val[cat_cols] = encoder.transform(df_val[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    # Calculate Vocab Sizes
    # We use the size of categories for each column
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # Scaling Continuous Features
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    df_train[cont_cols] = scaler.transform(df_train[cont_cols]).astype(np.float32)
    df_val[cont_cols] = scaler.transform(df_val[cont_cols]).astype(np.float32)
    df_test[cont_cols] = scaler.transform(df_test[cont_cols]).astype(np.float32)

    metadata = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "vocab_sizes": vocab_sizes,
    }

    # Save to cache only if not debugging
    if max_samples is None:
        df_train.to_parquet(train_cache)
        df_val.to_parquet(val_cache)
        df_test.to_parquet(test_cache)
        np.save(meta_cache, metadata)

    return df_train, df_val, df_test, metadata


def get_dataloaders(batch_size, load_cached_data=True, max_samples=None, config=Config):
    """
    Orchestrates data processing and returns DataLoaders.
    """
    df_train, df_val, df_test, metadata = preprocess_data(
        load_cached_data=load_cached_data, max_samples=max_samples, config=config
    )

    train_dataset = ManufacturingDataset(
        df_train, metadata["cat_cols"], metadata["cont_cols"], "target"
    )
    val_dataset = ManufacturingDataset(
        df_val, metadata["cat_cols"], metadata["cont_cols"], "target"
    )
    test_dataset = ManufacturingDataset(
        df_test, metadata["cat_cols"], metadata["cont_cols"], None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, metadata
