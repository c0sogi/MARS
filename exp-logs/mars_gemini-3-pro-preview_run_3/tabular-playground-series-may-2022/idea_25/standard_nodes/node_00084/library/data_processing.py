import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    Serves categorical indices and normalized continuous features.
    """

    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.cont_data = df[cont_cols].values.astype(np.float32)

        if target_col and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        cat = torch.from_numpy(self.cat_data[idx])
        cont = torch.from_numpy(self.cont_data[idx])

        if self.targets is not None:
            target = torch.tensor(self.targets[idx])
            return cat, cont, target
        else:
            return cat, cont


def process_data(load_cached_data=True):
    """
    Ingests, cleans, and processes the manufacturing data.
    Implements feature engineering, transductive ordinal encoding, and standard scaling.
    Caches the processed dataframes to disk to speed up subsequent runs.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_parquet = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    val_parquet = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    test_parquet = os.path.join(Config.CACHE_DIR, "test_processed.parquet")
    meta_path = os.path.join(Config.CACHE_DIR, "metadata.npy")

    # Check if cache exists and should be used
    if (
        load_cached_data
        and os.path.exists(train_parquet)
        and os.path.exists(val_parquet)
        and os.path.exists(test_parquet)
        and os.path.exists(meta_path)
    ):
        print("Loading cached data...")
        train_df = pd.read_parquet(train_parquet)
        val_df = pd.read_parquet(val_parquet)
        test_df = pd.read_parquet(test_parquet)
        meta_dict = np.load(meta_path, allow_pickle=True).item()
        return train_df, val_df, test_df, meta_dict

    print("Processing data from scratch...")

    # Load raw data from metadata directory (stratified splits)
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    def engineer_features(df):
        # Decompose f_27 into 10 separate character columns
        chars = df["f_27"].apply(lambda x: list(x))
        char_df = pd.DataFrame(
            chars.tolist(), columns=[f"char_{i}" for i in range(10)], index=df.index
        )

        # Feature: Count of unique characters in the sequence
        unique_counts = df["f_27"].apply(lambda x: len(set(x)))

        # Concatenate new features and drop original f_27
        df_eng = pd.concat([df, char_df], axis=1)
        df_eng["unique_char_count"] = unique_counts
        df_eng = df_eng.drop(columns=["f_27"])
        return df_eng

    print("Engineering features...")
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Define Column Groups
    # Categorical: char_0..char_9, plus f_29 and f_30 as requested
    cat_cols = [f"char_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: All other columns except IDs and metadata
    exclude_cols = set(cat_cols + ["id", "target", "source_path"])
    cont_cols = [c for c in df_train.columns if c not in exclude_cols]

    # Transductive Ordinal Encoding
    # Fit on Train + Val + Test to ensure consistent mapping for all tokens
    print("Encoding categorical features...")
    all_cat = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    # Convert all categorical columns to string to handle mixed types safely
    for col in cat_cols:
        all_cat[col] = all_cat[col].astype(str)
        df_train[col] = df_train[col].astype(str)
        df_val[col] = df_val[col].astype(str)
        df_test[col] = df_test[col].astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    encoder.fit(all_cat)

    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_val[cat_cols] = encoder.transform(df_val[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    # Calculate vocabulary sizes for embedding layers
    vocab_sizes = {col: len(encoder.categories_[i]) for i, col in enumerate(cat_cols)}

    # Standard Scaling for Continuous Features
    # Fit only on training data to prevent leakage
    print("Scaling continuous features...")
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    df_train[cont_cols] = scaler.transform(df_train[cont_cols]).astype(np.float32)
    df_val[cont_cols] = scaler.transform(df_val[cont_cols]).astype(np.float32)
    df_test[cont_cols] = scaler.transform(df_test[cont_cols]).astype(np.float32)

    # Save processed data to cache
    print("Saving to cache...")
    df_train.to_parquet(train_parquet)
    df_val.to_parquet(val_parquet)
    df_test.to_parquet(test_parquet)

    meta_dict = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "vocab_sizes": vocab_sizes,
    }
    np.save(meta_path, meta_dict)

    return df_train, df_val, df_test, meta_dict
