import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SEED,
    NUM_CHAR_POSITIONS,
    TARGET_COL,
    ID_COL,
)


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # pandas doesn't have a direct seed, but operations relying on numpy will be seeded


# =============================================================================
# Feature Engineering Helpers
# =============================================================================
def decompose_f27(df):
    """
    Decomposes the 'f_27' string column into 10 separate character columns.
    """
    # Fast decomposition using list comprehension
    # f_27 is guaranteed to be length 10
    chars = np.array([list(s) for s in df["f_27"].values])
    for i in range(NUM_CHAR_POSITIONS):
        df[f"p_{i}"] = chars[:, i]
    return df


def compute_unique_char_count(df):
    """
    Computes the number of unique characters in 'f_27'.
    """
    df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))
    return df


def compute_frequency_encoding(df, char_cols):
    """
    Computes frequency encoding for character columns.
    Adds 'freq_{i}' columns containing the normalized count of the character.
    """
    for i, col in enumerate(char_cols):
        # Compute normalized value counts (frequency)
        freq_map = df[col].value_counts(normalize=True)
        # Map back to the dataframe
        df[f"freq_{i}"] = df[col].map(freq_map)
    return df


# =============================================================================
# Main Data Processing
# =============================================================================
def process_data(load_cached_data=True):
    """
    Loads, processes, and caches the data.

    Args:
        load_cached_data (bool): If True, attempts to load from CACHE_DIR.

    Returns:
        train_df (pd.DataFrame): Processed training data.
        val_df (pd.DataFrame): Processed validation data.
        test_df (pd.DataFrame): Processed test data.
        vocab_sizes (list): List of integers representing vocab size for each char position.
        continuous_cols (list): List of continuous feature column names.
        categorical_cols (list): List of categorical feature column names.
    """
    set_seed(SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    cache_train = os.path.join(CACHE_DIR, "train_processed.parquet")
    cache_val = os.path.join(CACHE_DIR, "val_processed.parquet")
    cache_test = os.path.join(CACHE_DIR, "test_processed.parquet")
    cache_meta = os.path.join(CACHE_DIR, "metadata.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
            and os.path.exists(cache_meta)
        ):
            print("Loading data from cache...")
            train_df = pd.read_parquet(cache_train)
            val_df = pd.read_parquet(cache_val)
            test_df = pd.read_parquet(cache_test)
            metadata = np.load(cache_meta, allow_pickle=True).item()
            return (
                train_df,
                val_df,
                test_df,
                metadata["vocab_sizes"],
                metadata["continuous_cols"],
                metadata["categorical_cols"],
            )
        else:
            print("Cache not found or incomplete. Processing from scratch...")

    # 2. Load Raw Data
    print("Loading raw data from metadata...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Keep track of indices to split back later
    train_len = len(train_df)
    val_len = len(val_df)

    # Concatenate for transductive processing
    # Note: 'target' will be NaN for test, but we don't use it for feature engineering
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # 3. Feature Engineering
    print("Performing feature engineering...")

    # A. Decompose f_27
    full_df = decompose_f27(full_df)
    categorical_cols = [f"p_{i}" for i in range(NUM_CHAR_POSITIONS)]

    # 4. Encoding & Scaling
    print("Encoding and Scaling...")

    # A. Ordinal Encoding for Categorical Features
    # Transductive: Fit on full dataset
    enc = OrdinalEncoder(dtype=np.int64)
    full_df[categorical_cols] = enc.fit_transform(full_df[categorical_cols])

    # Get vocab sizes (max index + 1) for embeddings
    vocab_sizes = [int(full_df[col].max() + 1) for col in categorical_cols]

    # B. Identify Continuous Columns
    # Original continuous features: f_00 to f_26, f_28 to f_30
    # Exclude f_27 (string), id, target, source_path
    exclude_cols = [ID_COL, TARGET_COL, "source_path", "f_27"] + categorical_cols
    original_cont_cols = [
        c for c in train_df.columns if c not in exclude_cols and c != "f_27"
    ]

    # Combine all continuous features (No explicit statistical encodings - Cite solution_lesson_node_00101)
    continuous_cols = original_cont_cols

    # C. Split back to fit scaler on Train only
    train_proc = full_df.iloc[:train_len].copy()
    val_proc = full_df.iloc[train_len : train_len + val_len].copy()
    test_proc = full_df.iloc[train_len + val_len :].copy()

    # D. StandardScaler
    scaler = StandardScaler()
    # Fit on Train
    scaler.fit(train_proc[continuous_cols])

    # Transform All
    train_proc[continuous_cols] = scaler.transform(train_proc[continuous_cols])
    val_proc[continuous_cols] = scaler.transform(val_proc[continuous_cols])
    test_proc[continuous_cols] = scaler.transform(test_proc[continuous_cols])

    # Ensure data types
    # Continuous -> float32
    train_proc[continuous_cols] = train_proc[continuous_cols].astype(np.float32)
    val_proc[continuous_cols] = val_proc[continuous_cols].astype(np.float32)
    test_proc[continuous_cols] = test_proc[continuous_cols].astype(np.float32)

    # Categorical -> int64 (for embedding lookup)
    train_proc[categorical_cols] = train_proc[categorical_cols].astype(np.int64)
    val_proc[categorical_cols] = val_proc[categorical_cols].astype(np.int64)
    test_proc[categorical_cols] = test_proc[categorical_cols].astype(np.int64)

    # 5. Save to Cache
    print("Saving processed data to cache...")
    train_proc.to_parquet(cache_train, index=False)
    val_proc.to_parquet(cache_val, index=False)
    test_proc.to_parquet(cache_test, index=False)

    metadata = {
        "vocab_sizes": vocab_sizes,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
    }
    np.save(cache_meta, metadata)

    return (
        train_proc,
        val_proc,
        test_proc,
        vocab_sizes,
        continuous_cols,
        categorical_cols,
    )


# =============================================================================
# Dataset Class
# =============================================================================
class ManufacturingDataset(Dataset):
    def __init__(
        self,
        df,
        continuous_cols,
        categorical_cols,
        target_col=TARGET_COL,
        is_test=False,
    ):
        """
        PyTorch Dataset for Manufacturing Control Data.

        Args:
            df (pd.DataFrame): Processed dataframe.
            continuous_cols (list): List of continuous feature names.
            categorical_cols (list): List of categorical feature names.
            target_col (str): Name of the target column.
            is_test (bool): If True, does not look for target column.
        """
        self.continuous_features = df[continuous_cols].values.astype(np.float32)
        self.categorical_features = df[categorical_cols].values.astype(np.int64)
        self.is_test = is_test

        if not self.is_test:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None

        self.ids = df[ID_COL].values

    def __len__(self):
        return len(self.continuous_features)

    def __getitem__(self, idx):
        # Continuous: (N_cont,)
        cont = torch.tensor(self.continuous_features[idx], dtype=torch.float32)

        # Categorical: (10,)
        cat = torch.tensor(self.categorical_features[idx], dtype=torch.long)

        result = {"continuous": cont, "categorical": cat, "id": self.ids[idx]}

        if not self.is_test:
            # Target: (1,)
            target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
            result["target"] = target

        return result
