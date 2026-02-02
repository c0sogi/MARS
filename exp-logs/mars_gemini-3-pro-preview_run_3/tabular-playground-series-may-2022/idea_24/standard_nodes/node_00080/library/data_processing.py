import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control Data.
    Returns:
        x_cont (torch.Tensor): Normalized continuous features.
        x_cat (torch.Tensor): Ordinal encoded categorical features.
        y (torch.Tensor): Target variable (float32).
    """

    def __init__(self, df, cat_cols, cont_cols, target_col="target", mode="train"):
        self.mode = mode

        # Extract Continuous Data (float32)
        self.cont_data = df[cont_cols].values.astype(np.float32)

        # Extract Categorical Data (long/int64)
        self.cat_data = df[cat_cols].values.astype(np.int64)

        # Extract Target
        # Test set might not have target, or we might want to return dummy
        if self.mode != "test" and target_col in df.columns:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            # Return zeros for test set or if target missing
            self.targets = np.zeros(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        x_cont = torch.tensor(self.cont_data[idx], dtype=torch.float32)
        x_cat = torch.tensor(self.cat_data[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x_cont, x_cat, y


def _decompose_f27(df):
    """
    Helper function to decompose the string feature 'f_27'.
    - Splits 'f_27' into 10 separate character columns.
    - Computes 'unique_char_count'.
    """
    # Decompose string into characters (assuming fixed length of 10)
    # Using vectorized string slicing is efficient
    chars = {}
    for i in range(10):
        chars[f"ch_{i}"] = df["f_27"].str[i]

    chars_df = pd.DataFrame(chars, index=df.index)

    # Compute unique character count
    unique_count = df["f_27"].apply(lambda x: len(set(x))).rename("unique_char_count")

    # Concatenate new features
    df = pd.concat([df, chars_df, unique_count], axis=1)
    return df


def preprocess_features(load_cached_data=True, config=Config):
    """
    Main data processing pipeline.
    - Loads data from metadata paths.
    - Performs feature engineering (f_27 decomposition).
    - Applies Transductive Ordinal Encoding (fit on Train+Val+Test).
    - Applies Standardization (fit on Train).
    - Caches processed data to parquet/npy.

    Returns:
        train_df, val_df, test_df (pd.DataFrame): Processed dataframes.
        vocab_sizes (list): List of integers representing vocab size for each categorical column.
        cat_cols (list): List of categorical column names.
        cont_cols (list): List of continuous column names.
    """

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Cache file paths
    train_cache = config.CACHE_TRAIN_PATH
    val_cache = config.CACHE_VAL_PATH
    test_cache = config.CACHE_TEST_PATH
    meta_cache = config.CACHE_META_PATH

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
            and os.path.exists(meta_cache)
        ):

            print(f"Loading cached data from {config.WORKING_DIR}...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)

            metadata = np.load(meta_cache, allow_pickle=True).item()
            vocab_sizes = metadata["vocab_sizes"]
            cat_cols = metadata["cat_cols"]
            cont_cols = metadata["cont_cols"]

            return train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols

    print("Cache not found or reload requested. Processing data from scratch...")

    # 2. Load Raw Data
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df = pd.read_csv(config.VAL_PATH)
    test_df = pd.read_csv(config.TEST_PATH)

    # Debug Mode
    if config.DEBUG:
        print(f"DEBUG mode: Subsampling to {config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.iloc[: config.DEBUG_SAMPLE_SIZE].copy()
        val_df = val_df.iloc[: config.DEBUG_SAMPLE_SIZE].copy()
        test_df = test_df.iloc[: config.DEBUG_SAMPLE_SIZE].copy()

    # 3. Feature Engineering
    # Define initial column groups
    # f_00 to f_26 are continuous, f_28 is continuous
    raw_cont_cols = [f"f_{i:02d}" for i in range(27)] + ["f_28"]
    # f_29, f_30 are categorical
    raw_cat_cols = ["f_29", "f_30"]

    # Apply decomposition
    train_df = _decompose_f27(train_df)
    val_df = _decompose_f27(val_df)
    test_df = _decompose_f27(test_df)

    # Define Final Column Lists
    # New categorical columns from f_27
    char_cols = [f"ch_{i}" for i in range(10)]
    all_cat_cols = raw_cat_cols + char_cols

    # New continuous columns
    all_cont_cols = raw_cont_cols + ["unique_char_count"]

    # 4. Transductive Ordinal Encoding
    # Fit on Train + Val + Test to ensure alignment
    print("Applying Transductive Ordinal Encoding...")

    # Convert all categorical columns to string to ensure safety with OrdinalEncoder
    for df in [train_df, val_df, test_df]:
        for col in all_cat_cols:
            df[col] = df[col].astype(str)

    # Concatenate for fitting
    full_cat_data = pd.concat(
        [train_df[all_cat_cols], val_df[all_cat_cols], test_df[all_cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(full_cat_data)

    # Transform
    train_df[all_cat_cols] = encoder.transform(train_df[all_cat_cols])
    val_df[all_cat_cols] = encoder.transform(val_df[all_cat_cols])
    test_df[all_cat_cols] = encoder.transform(test_df[all_cat_cols])

    # Calculate vocab sizes (number of unique categories per column)
    # Since we fit on all data, this covers the full vocabulary
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # 5. Normalization (StandardScaler)
    # Fit ONLY on Train
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(train_df[all_cont_cols])

    train_df[all_cont_cols] = scaler.transform(train_df[all_cont_cols])
    val_df[all_cont_cols] = scaler.transform(val_df[all_cont_cols])
    test_df[all_cont_cols] = scaler.transform(test_df[all_cont_cols])

    # 6. Save to Cache
    print("Saving processed data to cache...")

    # Select columns to save (ID + Features + Target if exists)
    cols_to_save = ["id"] + all_cont_cols + all_cat_cols

    # Save Train
    train_save_cols = (
        cols_to_save + ["target"] if "target" in train_df.columns else cols_to_save
    )
    train_df[train_save_cols].to_parquet(train_cache, index=False)

    # Save Val
    val_save_cols = (
        cols_to_save + ["target"] if "target" in val_df.columns else cols_to_save
    )
    val_df[val_save_cols].to_parquet(val_cache, index=False)

    # Save Test
    test_df[cols_to_save].to_parquet(test_cache, index=False)

    # Save Metadata
    metadata = {
        "vocab_sizes": vocab_sizes,
        "cat_cols": all_cat_cols,
        "cont_cols": all_cont_cols,
    }
    np.save(meta_cache, metadata)

    return train_df, val_df, test_df, vocab_sizes, all_cat_cols, all_cont_cols
