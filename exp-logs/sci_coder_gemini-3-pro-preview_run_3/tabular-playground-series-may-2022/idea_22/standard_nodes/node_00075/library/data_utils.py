import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


class TabularDataset(Dataset):
    """
    PyTorch Dataset for the Anchor-Variant Parallel Funnel Ensemble.
    Serves categorical indices and normalized continuous features.
    """

    def __init__(self, x_cat, x_cont, y=None):
        self.x_cat = torch.LongTensor(x_cat)
        self.x_cont = torch.FloatTensor(x_cont)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]


def _split_f27(df):
    """
    Decomposes the 10-character string 'f_27' into 10 separate columns.
    """
    # Create 10 new columns based on character positions
    # Using list comprehension with str accessor is generally efficient for this
    for i in range(Config.N_F27_CHARS):
        df[f"f_27_{i}"] = df["f_27"].str[i]
    return df


def preprocess_raw_df(df):
    """
    Applies feature engineering:
    1. Decomposes f_27.
    2. Computes unique_character_count.
    """
    # 1. Decompose f_27
    df = _split_f27(df)

    # 2. Unique character count
    # We can map the string to the count of unique characters
    df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))

    return df


def load_and_preprocess_data(load_cached_data=True):
    """
    Main data loading pipeline.
    1. Checks for cached processed parquet files.
    2. If not found or forced reload:
       - Loads raw metadata CSVs.
       - Performs feature engineering.
       - Fits Transductive OrdinalEncoder (Train+Val+Test).
       - Fits StandardScaler (Train only).
       - Saves processed data to cache.
    3. Returns TabularDataset objects for Train, Val, and Test.
    """

    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)

        # Identify columns based on naming convention or type
        # We assume the parquet files contain the final transformed features
        # We need to separate them back into x_cat, x_cont, y

        # Helper to extract arrays
        def extract_arrays(df, is_test=False):
            # Categorical columns: f_27_0...f_27_9, f_29, f_30
            # Continuous columns: f_00...f_26, f_28, unique_char_count
            # Target: target

            # We need to dynamically identify columns because we don't save the list in parquet
            # However, we know the schema from the processing step below.

            # Re-identify columns logic (must match the processing logic)
            cat_cols = [
                f"f_27_{i}" for i in range(Config.N_F27_CHARS)
            ] + Config.DISCRETE_FEATURES
            cont_cols = [
                c
                for c in df.columns
                if c.startswith("f_") and c not in cat_cols and c != "f_27"
            ]
            if "unique_char_count" in df.columns:
                cont_cols.append("unique_char_count")

            # Ensure correct order
            cat_cols = sorted(cat_cols)
            cont_cols = sorted(cont_cols)

            x_cat = df[cat_cols].values.astype(np.int64)
            x_cont = df[cont_cols].values.astype(np.float32)

            y = None
            if not is_test and "target" in df.columns:
                y = df["target"].values.astype(np.float32)

            return x_cat, x_cont, y

        x_train_cat, x_train_cont, y_train = extract_arrays(df_train)
        x_val_cat, x_val_cont, y_val = extract_arrays(df_val)
        x_test_cat, x_test_cont, _ = extract_arrays(df_test, is_test=True)

    else:
        print("Processing data from scratch...")

        # Load raw data
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # Feature Engineering
        print("Applying feature engineering...")
        df_train = preprocess_raw_df(df_train)
        df_val = preprocess_raw_df(df_val)
        df_test = preprocess_raw_df(df_test)

        # Define Column Groups
        # Categorical: Decomposed f_27 chars + f_29 + f_30
        cat_cols = [
            f"f_27_{i}" for i in range(Config.N_F27_CHARS)
        ] + Config.DISCRETE_FEATURES
        cat_cols = sorted(cat_cols)

        # Continuous: All f_XX not in cat_cols, excluding f_27 (string), id, target, source_path
        # plus unique_char_count
        exclude_cols = ["id", "target", "source_path", "f_27"] + cat_cols
        cont_cols = [c for c in df_train.columns if c not in exclude_cols]
        cont_cols = sorted(cont_cols)

        # 1. Transductive Categorical Encoding
        print("Fitting transductive OrdinalEncoder...")
        # Concatenate all data for encoder fitting
        full_cat_data = pd.concat(
            [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
        )

        encoder = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        )
        encoder.fit(full_cat_data)

        # Transform
        df_train[cat_cols] = encoder.transform(df_train[cat_cols])
        df_val[cat_cols] = encoder.transform(df_val[cat_cols])
        df_test[cat_cols] = encoder.transform(df_test[cat_cols])

        # 2. Continuous Normalization
        print("Fitting StandardScaler on training data...")
        scaler = StandardScaler()
        scaler.fit(df_train[cont_cols])

        # Transform
        df_train[cont_cols] = scaler.transform(df_train[cont_cols])
        df_val[cont_cols] = scaler.transform(df_val[cont_cols])
        df_test[cont_cols] = scaler.transform(df_test[cont_cols])

        # Save to cache
        print(f"Saving processed data to {cache_dir}...")
        # We save the full dataframes including target for train/val
        # We drop columns that are not features or target to save space/confusion
        keep_cols_train = cat_cols + cont_cols + ["target"]
        keep_cols_test = cat_cols + cont_cols

        df_train[keep_cols_train].to_parquet(train_cache, index=False)
        df_val[keep_cols_train].to_parquet(val_cache, index=False)
        df_test[keep_cols_test].to_parquet(test_cache, index=False)

        # Prepare arrays for Dataset
        x_train_cat = df_train[cat_cols].values.astype(np.int64)
        x_train_cont = df_train[cont_cols].values.astype(np.float32)
        y_train = df_train["target"].values.astype(np.float32)

        x_val_cat = df_val[cat_cols].values.astype(np.int64)
        x_val_cont = df_val[cont_cols].values.astype(np.float32)
        y_val = df_val["target"].values.astype(np.float32)

        x_test_cat = df_test[cat_cols].values.astype(np.int64)
        x_test_cont = df_test[cont_cols].values.astype(np.float32)

    # Create Datasets
    train_dataset = TabularDataset(x_train_cat, x_train_cont, y_train)
    val_dataset = TabularDataset(x_val_cat, x_val_cont, y_val)
    test_dataset = TabularDataset(x_test_cat, x_test_cont, None)

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
