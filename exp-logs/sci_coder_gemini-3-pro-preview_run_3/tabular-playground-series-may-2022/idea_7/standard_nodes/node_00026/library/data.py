import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from library.config import Config
from library.utils import set_seed


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Separates continuous and categorical features for the Dual-Stream architecture.
    """

    def __init__(self, df, is_test=False):
        self.is_test = is_test

        # Extract continuous features
        self.cont_features = df[Config.CONT_FEATURES].values.astype(np.float32)

        # Extract categorical features
        # Ensure they are long/int type for embedding lookups
        self.cat_features = df[Config.CAT_FEATURES].values.astype(np.int64)

        # Extract target if not test set
        if not self.is_test:
            self.targets = df[Config.TARGET_COL].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        x_cont = self.cont_features[idx]
        x_cat = self.cat_features[idx]

        if self.is_test:
            return {
                "x_cont": torch.tensor(x_cont, dtype=torch.float32),
                "x_cat": torch.tensor(x_cat, dtype=torch.long),
                "id": idx,  # Placeholder, actual IDs are handled via order
            }
        else:
            y = self.targets[idx]
            return {
                "x_cont": torch.tensor(x_cont, dtype=torch.float32),
                "x_cat": torch.tensor(x_cat, dtype=torch.long),
                "target": torch.tensor(y, dtype=torch.float32),
            }


def _engineer_features(df):
    """
    Performs feature engineering:
    1. Decomposes f_27 into 10 character columns.
    2. Computes unique_character_count.
    """
    # Avoid modifying original dataframe
    df = df.copy()

    # 1. Decompose f_27
    # Vectorized string splitting
    # We expect f_27 to be a string of length 10
    # We create columns f_27_0 ... f_27_9
    f27_series = df[Config.F_27_COL].astype(str)

    # Create a temporary dataframe with split characters
    # This is much faster than apply(list)
    # We assume fixed length of 10 based on analysis
    splits = f27_series.str.split("", expand=True).iloc[:, 1:11]
    splits.columns = [f"f_27_{i}" for i in range(10)]

    # Concatenate splits back to df
    df = pd.concat([df, splits], axis=1)

    # 2. Unique Character Count
    # Calculate number of unique characters in the string
    df[Config.UNIQUE_CHAR_COUNT_COL] = f27_series.apply(lambda x: len(set(x)))

    return df


def prepare_data(load_cached_data=True, debug=False):
    """
    Main function to load, process, and return DataLoaders.

    Args:
        load_cached_data (bool): If True, attempts to load processed parquet files.
        debug (bool): If True, subsamples the data for rapid prototyping.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed()

    # Define cache paths
    train_cache = Config.CACHE_TRAIN_PATH
    val_cache = Config.CACHE_VAL_PATH
    test_cache = Config.CACHE_TEST_PATH

    # -------------------------------------------------------------------------
    # 1. Try Loading from Cache
    # -------------------------------------------------------------------------
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading processed data from cache...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)

            # If debug, sample after loading
            if debug:
                print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
                df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
                df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
                df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

            # Skip to DataLoader creation
            return _create_dataloaders(df_train, df_val, df_test)
        else:
            print("Cache not found or incomplete. Processing from scratch...")

    # -------------------------------------------------------------------------
    # 2. Load Raw Data
    # -------------------------------------------------------------------------
    print("Loading raw data...")
    df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
    df_val = pd.read_csv(Config.VAL_DATA_PATH)
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # -------------------------------------------------------------------------
    # 3. Feature Engineering
    # -------------------------------------------------------------------------
    print("Engineering features...")
    df_train = _engineer_features(df_train)
    df_val = _engineer_features(df_val)
    df_test = _engineer_features(df_test)

    # -------------------------------------------------------------------------
    # 4. Transductive Categorical Encoding
    # -------------------------------------------------------------------------
    print("Fitting Transductive Ordinal Encoder...")
    # Concatenate all data to ensure vocabulary alignment
    # We only need the categorical columns for fitting
    cat_cols = Config.CAT_FEATURES

    full_cat_data = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )

    encoder.fit(full_cat_data)

    # Transform each split
    df_train[cat_cols] = encoder.transform(df_train[cat_cols])
    df_val[cat_cols] = encoder.transform(df_val[cat_cols])
    df_test[cat_cols] = encoder.transform(df_test[cat_cols])

    # Handle any potential unknown values (though transductive fit minimizes this)
    # Map -1 to a safe index if necessary, but here we assume fit covers all.
    # If unknown_value was used, we might want to shift indices or handle it,
    # but OrdinalEncoder with full data covers the vocab.

    # -------------------------------------------------------------------------
    # 5. Continuous Feature Scaling
    # -------------------------------------------------------------------------
    print("Scaling continuous features...")
    cont_cols = Config.CONT_FEATURES

    scaler = StandardScaler()

    # Fit only on training data
    scaler.fit(df_train[cont_cols])

    # Transform all
    df_train[cont_cols] = scaler.transform(df_train[cont_cols])
    df_val[cont_cols] = scaler.transform(df_val[cont_cols])
    df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    # -------------------------------------------------------------------------
    # 6. Save to Cache
    # -------------------------------------------------------------------------
    if not debug:
        print("Saving processed data to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        df_train.to_parquet(train_cache)
        df_val.to_parquet(val_cache)
        df_test.to_parquet(test_cache)

    return _create_dataloaders(df_train, df_val, df_test)


def _create_dataloaders(df_train, df_val, df_test):
    """
    Internal helper to wrap dataframes in Datasets and DataLoaders.
    """
    print("Creating DataLoaders...")

    train_dataset = ManufacturingDataset(df_train, is_test=False)
    val_dataset = ManufacturingDataset(df_val, is_test=False)
    test_dataset = ManufacturingDataset(df_test, is_test=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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

    return train_loader, val_loader, test_loader
