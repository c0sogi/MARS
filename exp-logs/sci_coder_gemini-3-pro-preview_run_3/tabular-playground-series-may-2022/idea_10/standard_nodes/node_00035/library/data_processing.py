import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config
from library.utils import set_seed


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Serves continuous features, categorical features, and targets.
    """

    def __init__(self, cont_features, cat_features, targets=None):
        self.cont_features = torch.FloatTensor(cont_features)
        self.cat_features = torch.LongTensor(cat_features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.cont_features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.cont_features[idx], self.cat_features[idx], self.targets[idx]
        else:
            return self.cont_features[idx], self.cat_features[idx]


def process_features(df):
    """
    Performs feature engineering on the raw dataframe.
    - Decomposes f_27 into unigrams and bigrams.
    - Computes unique character count.
    """
    # Ensure f_27 is string
    df["f_27"] = df["f_27"].astype(str)

    # 1. Unigrams (10 positions)
    for i in range(Config.F27_SEQ_LEN):
        df[f"f_27_uni_{i}"] = df["f_27"].str[i]

    # 2. Bigrams (9 sliding windows)
    for i in range(Config.F27_BIGRAM_LEN):
        df[f"f_27_bi_{i}"] = df["f_27"].str[i : i + 2]

    # 3. Unique character count
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

    return df


def get_data_loaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug=Config.DEBUG,
    debug_samples=Config.DEBUG_SAMPLES,
):
    """
    Main function to load data, process features, and return DataLoaders.
    Implements caching and transductive vocabulary alignment.
    """
    set_seed()

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")
    vocab_cache = os.path.join(cache_dir, "vocab_sizes.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(vocab_cache)
    )

    if load_cached_data and cache_exists and not debug:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        vocab_sizes = np.load(vocab_cache, allow_pickle=True).tolist()

        # Identify feature columns from cached dataframe
        # We assume the last column in train/val is target, and test doesn't have it.
        # Cat cols are int, Cont cols are float.
        # However, to be precise, we reconstruct the column lists based on naming convention
        # or just use the known schema.

        # Re-identifying columns based on schema used during saving
        # This relies on the deterministic nature of the processing below.
        pass  # Data is loaded, we will convert to tensors later

    else:
        print("Processing data from scratch...")
        # Load raw data
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        if debug:
            print(f"DEBUG MODE: Subsampling {debug_samples} samples.")
            train_df = train_df.iloc[:debug_samples].copy()
            val_df = val_df.iloc[:debug_samples].copy()
            test_df = test_df.iloc[:debug_samples].copy()

        # Feature Engineering
        print("Engineering features...")
        train_df = process_features(train_df)
        val_df = process_features(val_df)
        test_df = process_features(test_df)

        # Define Column Groups
        # Categorical: Unigrams + Bigrams + f_29 + f_30
        unigram_cols = [f"f_27_uni_{i}" for i in range(Config.F27_SEQ_LEN)]
        bigram_cols = [f"f_27_bi_{i}" for i in range(Config.F27_BIGRAM_LEN)]
        other_cat_cols = ["f_29", "f_30"]
        cat_cols = unigram_cols + bigram_cols + other_cat_cols

        # Continuous: f_00 to f_28 (excluding f_27) + unique_character_count
        # Note: f_29 and f_30 are treated as categorical
        cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
        cont_cols.append("unique_character_count")

        # Transductive Label Encoding
        # Fit on all data to handle vocabulary alignment
        print("Fitting encoders and scalers...")
        all_cat_data = pd.concat(
            [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
        )

        # Convert to string to ensure consistent encoding
        all_cat_data = all_cat_data.astype(str)

        encoder = OrdinalEncoder(dtype=np.int64)
        encoder.fit(all_cat_data)

        # Transform
        train_df[cat_cols] = encoder.transform(train_df[cat_cols].astype(str))
        val_df[cat_cols] = encoder.transform(val_df[cat_cols].astype(str))
        test_df[cat_cols] = encoder.transform(test_df[cat_cols].astype(str))

        # Calculate vocab sizes for embedding layers
        vocab_sizes = [len(cats) for cats in encoder.categories_]

        # Normalization (StandardScaler)
        # Fit ONLY on Train
        scaler = StandardScaler()
        scaler.fit(train_df[cont_cols])

        train_df[cont_cols] = scaler.transform(train_df[cont_cols])
        val_df[cont_cols] = scaler.transform(val_df[cont_cols])
        test_df[cont_cols] = scaler.transform(test_df[cont_cols])

        # Select only necessary columns for the final dataframe
        # We keep ID for tracking in test, Target for train/val
        final_cols = cont_cols + cat_cols

        # Save to cache (if not debugging)
        if not debug:
            print("Saving processed data to cache...")
            # We save the full dataframes including target and id to keep context
            train_df.to_parquet(train_cache)
            val_df.to_parquet(val_cache)
            test_df.to_parquet(test_cache)
            np.save(vocab_cache, np.array(vocab_sizes))

    # Prepare Tensors
    # Re-define columns to ensure order is preserved when loading from cache
    unigram_cols = [f"f_27_uni_{i}" for i in range(Config.F27_SEQ_LEN)]
    bigram_cols = [f"f_27_bi_{i}" for i in range(Config.F27_BIGRAM_LEN)]
    other_cat_cols = ["f_29", "f_30"]
    cat_cols = unigram_cols + bigram_cols + other_cat_cols

    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_cols.append("unique_character_count")

    # Extract arrays
    train_X_cont = train_df[cont_cols].values.astype(np.float32)
    train_X_cat = train_df[cat_cols].values.astype(np.int64)
    train_y = train_df[Config.TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    val_X_cont = val_df[cont_cols].values.astype(np.float32)
    val_X_cat = val_df[cat_cols].values.astype(np.int64)
    val_y = val_df[Config.TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    test_X_cont = test_df[cont_cols].values.astype(np.float32)
    test_X_cat = test_df[cat_cols].values.astype(np.int64)
    # Test set has no target

    # Create Datasets
    train_dataset = ManufacturingDataset(train_X_cont, train_X_cat, train_y)
    val_dataset = ManufacturingDataset(val_X_cont, val_X_cat, val_y)
    test_dataset = ManufacturingDataset(test_X_cont, test_X_cat, targets=None)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    print(f"Vocab sizes: {vocab_sizes}")

    return train_loader, val_loader, test_loader, vocab_sizes
