import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Serves categorical and continuous features separately for the RSPFE model.
    """

    def __init__(self, cat_data, cont_data, targets=None):
        self.cat_data = torch.LongTensor(cat_data)
        self.cont_data = torch.FloatTensor(cont_data)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        item = {
            "cat_features": self.cat_data[idx],
            "cont_features": self.cont_data[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item


def feature_engineering(df):
    """
    Applies feature engineering:
    1. Decomposes f_27 into 10 character columns.
    2. Computes unique_character_count for f_27.
    """
    # 1. Decompose f_27
    # We expect f_27 to be a string of length 10
    # We create columns f_27_0 to f_27_9
    chars = df["f_27"].apply(list)
    chars_df = pd.DataFrame(chars.tolist(), index=df.index)
    chars_df.columns = [f"f_27_{i}" for i in range(10)]

    # Concatenate the new char columns
    df = pd.concat([df, chars_df], axis=1)

    # 2. Unique character count
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

    return df


def load_and_preprocess(load_cached_data=True):
    """
    Loads data, performs feature engineering, transductive encoding, and scaling.
    Implements caching to speed up subsequent runs.
    """
    set_seed(Config.SEED)

    # Cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    vocab_cache = os.path.join(cache_dir, "vocab_sizes.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(vocab_cache)
    ):
        print("Loading processed data from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
        vocab_sizes = np.load(vocab_cache, allow_pickle=False)
        return df_train, df_val, df_test, vocab_sizes.tolist()

    print("Processing data from scratch...")

    # Load raw data based on metadata splits
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print(f"Debug mode: Subsampling {Config.DEBUG_SAMPLES} samples.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLES].copy()
        df_val = df_val.iloc[: min(len(df_val), Config.DEBUG_SAMPLES // 5)].copy()
        df_test = df_test.iloc[: min(len(df_test), Config.DEBUG_SAMPLES // 5)].copy()

    # Feature Engineering
    print("Applying feature engineering...")
    df_train = feature_engineering(df_train)
    df_val = feature_engineering(df_val)
    df_test = feature_engineering(df_test)

    # Define feature groups
    cat_cols = Config.CAT_FEATURES
    cont_cols = Config.CONT_FEATURES

    # Transductive Categorical Encoding
    # Concatenate all splits to ensure global vocabulary alignment
    print("Performing transductive categorical encoding...")
    all_cat = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    # Initialize Ordinal Encoder
    # Handle unknown is not strictly necessary due to transductive approach,
    # but good practice. We use encoded values as indices.
    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(all_cat)

    # Transform
    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_val[cat_cols] = encoder.transform(df_val[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    # Calculate vocab sizes for embeddings (max index + 1)
    # We can get this from the categories_ attribute of the encoder
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # Continuous Normalization
    print("Normalizing continuous features...")
    scaler = StandardScaler()

    # Fit only on Train
    scaler.fit(df_train[cont_cols])

    # Transform all
    df_train[cont_cols] = scaler.transform(df_train[cont_cols])
    df_val[cont_cols] = scaler.transform(df_val[cont_cols])
    df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    # Save to cache
    print("Saving processed data to cache...")
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)
    np.save(vocab_cache, np.array(vocab_sizes))

    return df_train, df_val, df_test, vocab_sizes


def get_dataloaders(df_train, df_val, df_test, batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    """
    cat_cols = Config.CAT_FEATURES
    cont_cols = Config.CONT_FEATURES

    # Prepare Train
    train_dataset = ManufacturingDataset(
        cat_data=df_train[cat_cols].values,
        cont_data=df_train[cont_cols].values,
        targets=df_train["target"].values,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Prepare Val
    val_dataset = ManufacturingDataset(
        cat_data=df_val[cat_cols].values,
        cont_data=df_val[cont_cols].values,
        targets=df_val["target"].values,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Prepare Test
    # Test set does not have targets
    test_dataset = ManufacturingDataset(
        cat_data=df_test[cat_cols].values,
        cont_data=df_test[cont_cols].values,
        targets=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
