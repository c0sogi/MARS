import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


def feature_engineering(df):
    """
    Applies feature engineering transformations:
    1. Computes unique character count for f_27.
    2. Decomposes f_27 string into 10 separate character columns.
    """
    if "f_27" in df.columns:
        # 1. Unique character count
        df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(str(x))))

        # 2. Split string into columns (f_27_char_0 ... f_27_char_9)
        # We assume f_27 is fixed length 10.
        # Convert to list of chars then expand to columns
        chars = df["f_27"].apply(lambda x: list(str(x)))
        chars_df = pd.DataFrame(chars.tolist(), index=df.index)
        chars_df.columns = [f"f_27_char_{i}" for i in range(10)]

        # Concatenate new features
        df = pd.concat([df, chars_df], axis=1)

    return df


def preprocess_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads data, performs feature engineering, encoding, and scaling.
    Handles caching and debug sampling.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, samples data and skips saving to cache.

    Returns:
        train_df, val_df, test_df, metadata
    """
    # Define cache paths
    train_cache = Config.TRAIN_PROCESSED_PATH
    val_cache = Config.VAL_PROCESSED_PATH
    test_cache = Config.TEST_PROCESSED_PATH
    meta_cache = Config.METADATA_CACHE_PATH

    # 1. Try Loading from Cache (if not debugging)
    if (
        not debug
        and load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):

        print(f"Loading cached data from {Config.WORKING_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()
        return train_df, val_df, test_df, metadata

    # 2. Load Raw Data
    print("Loading raw data from metadata paths...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Debug Sampling
    if debug:
        print(f"Debug mode enabled: Sampling {Config.DEBUG_SAMPLES} rows per split.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLES].copy()
        val_df = val_df.iloc[: Config.DEBUG_SAMPLES].copy()
        test_df = test_df.iloc[: Config.DEBUG_SAMPLES].copy()

    # 3. Feature Engineering
    print("Performing feature engineering...")
    train_df = feature_engineering(train_df)
    val_df = feature_engineering(val_df)
    test_df = feature_engineering(test_df)

    # Define Column Groups from Config
    cont_cols = Config.CONTINUOUS_FEATURES
    cat_cols = Config.CATEGORICAL_FEATURES

    # 4. Transductive Categorical Encoding
    # Fit on Train + Val + Test to ensure global vocabulary alignment
    print("Encoding categorical features...")
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )

    combined_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )

    encoder.fit(combined_cat)

    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    val_df[cat_cols] = encoder.transform(val_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Calculate vocab sizes for the model
    vocab_sizes_map = {}
    for i, col in enumerate(cat_cols):
        vocab_sizes_map[col] = len(encoder.categories_[i])

    # 5. Normalization of Continuous Features
    # Fit on Train only to prevent leakage
    print("Scaling continuous features...")
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    train_df[cont_cols] = scaler.transform(train_df[cont_cols])
    val_df[cont_cols] = scaler.transform(val_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    # Prepare Metadata
    metadata = {
        "vocab_sizes": vocab_sizes_map,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
    }

    # 6. Save to Cache (only if not debugging)
    if not debug:
        print(f"Saving processed data to {Config.WORKING_DIR}...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        train_df.to_parquet(train_cache)
        val_df.to_parquet(val_cache)
        test_df.to_parquet(test_cache)
        np.save(meta_cache, metadata)

    return train_df, val_df, test_df, metadata


class ManufacturingDataset(Dataset):
    def __init__(self, df, metadata, is_test=False):
        """
        PyTorch Dataset for Manufacturing Control Data.

        Args:
            df (pd.DataFrame): Preprocessed dataframe.
            metadata (dict): Metadata dictionary containing column names.
            is_test (bool): If True, target column is not expected.
        """
        self.df = df
        self.is_test = is_test
        self.cat_cols = metadata["cat_cols"]
        self.cont_cols = metadata["cont_cols"]

        # Convert to numpy arrays for efficient indexing
        self.cont_data = self.df[self.cont_cols].values.astype(np.float32)
        self.cat_data = self.df[self.cat_cols].values.astype(np.int64)
        self.ids = self.df[Config.ID_COL].values

        if not self.is_test:
            self.targets = self.df[Config.TARGET_COL].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Continuous features
        cont_x = torch.tensor(self.cont_data[idx], dtype=torch.float32)

        # Categorical features
        cat_x = torch.tensor(self.cat_data[idx], dtype=torch.long)

        result = {"cont": cont_x, "cat": cat_x, "id": self.ids[idx]}

        if not self.is_test:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            result["target"] = target

        return result
