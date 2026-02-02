import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularDataset(Dataset):
    """
    PyTorch Dataset for the manufacturing control data.
    Separates features into categorical (for embeddings) and continuous (for dense layers).
    """

    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cat_features = df[cat_cols].values.astype(np.int64)
        self.cont_features = df[cont_cols].values.astype(np.float32)

        if target_col and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.cat_features)

    def __getitem__(self, idx):
        x_cat = torch.tensor(self.cat_features[idx], dtype=torch.long)
        x_cont = torch.tensor(self.cont_features[idx], dtype=torch.float32)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_cat, x_cont, y
        else:
            return x_cat, x_cont


def decompose_f27(df):
    """
    Splits the 'f_27' string column into 10 separate character columns.
    """
    # Vectorized string splitting
    for i in range(10):
        df[f"f_27_{i}"] = df["f_27"].str[i]
    return df


def add_unique_char_count(df):
    """
    Adds a feature counting unique characters in 'f_27'.
    """
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))
    return df


def preprocess_pipeline(load_cached_data=True):
    """
    End-to-end data preprocessing pipeline with caching.

    1. Checks for cached parquet/npy files.
    2. If not found, loads raw CSVs.
    3. Performs feature engineering (f_27 decomposition, unique count).
    4. Concatenates Train/Val/Test for transductive Ordinal Encoding and Scaling.
    5. Splits back and saves to cache.

    Returns:
        train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE
    meta_cache = Config.PREPROCESSOR_CACHE

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):

        print(f"Loading cached data from {Config.WORKING_DIR}...")
        try:
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)

            meta_dict = np.load(meta_cache, allow_pickle=True).item()
            vocab_sizes = meta_dict["vocab_sizes"]
            cat_cols = meta_dict["cat_cols"]
            cont_cols = meta_dict["cont_cols"]

            return train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Processing data from scratch...")

    # 2. Load Raw Data
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if Config.DEBUG:
        print(f"DEBUG MODE: Using {Config.DEBUG_SAMPLES} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        test_df = test_df.head(Config.DEBUG_SAMPLES)

    # 3. Feature Engineering
    # We apply this to all dataframes
    for df in [train_df, val_df, test_df]:
        df = decompose_f27(df)
        df = add_unique_char_count(df)

    # 4. Define Column Groups
    # Categorical: The 10 chars from f_27, plus f_29 and f_30 (as per strategy)
    cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00 to f_28 (excluding f_27), plus the new unique_character_count
    # Note: f_29 and f_30 are moved to categorical
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]

    # 5. Transductive Processing
    # We concatenate everything to fit encoders/scalers on the global distribution
    # This ensures consistent vocabulary mapping for embeddings

    # Add split identifier
    train_df["split_id"] = 0
    val_df["split_id"] = 1
    test_df["split_id"] = 2

    combined_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # Encode Categoricals
    print("Fitting OrdinalEncoder on combined data...")
    ord_enc = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )
    combined_df[cat_cols] = ord_enc.fit_transform(combined_df[cat_cols])

    # Calculate vocab sizes for embeddings (max index + 1)
    vocab_sizes = [int(combined_df[col].max() + 1) for col in cat_cols]

    # Scale Continuous Features
    print("Fitting StandardScaler on combined data...")
    scaler = StandardScaler()
    combined_df[cont_cols] = scaler.fit_transform(combined_df[cont_cols])

    # 6. Split Back
    train_processed = combined_df[combined_df["split_id"] == 0].copy()
    val_processed = combined_df[combined_df["split_id"] == 1].copy()
    test_processed = combined_df[combined_df["split_id"] == 2].copy()

    # Clean up auxiliary columns
    for df in [train_processed, val_processed, test_processed]:
        df.drop(columns=["split_id"], inplace=True)
        # We also drop the original f_27 and source_path if they exist to save space
        # though Parquet compression handles unused cols well, it's cleaner to drop
        cols_to_drop = ["f_27", "source_path"]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # 7. Save to Cache
    print("Saving processed data to cache...")
    train_processed.to_parquet(train_cache, index=False)
    val_processed.to_parquet(val_cache, index=False)
    test_processed.to_parquet(test_cache, index=False)

    meta_dict = {
        "vocab_sizes": vocab_sizes,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
    }
    np.save(meta_cache, meta_dict)

    return (
        train_processed,
        val_processed,
        test_processed,
        vocab_sizes,
        cat_cols,
        cont_cols,
    )
