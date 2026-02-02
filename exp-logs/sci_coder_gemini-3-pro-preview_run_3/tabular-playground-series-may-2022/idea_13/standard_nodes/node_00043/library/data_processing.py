import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import (
    Config,
    set_seed,
    feature_engineering as preprocess_data,
    ManufacturingDataset,
)


def create_transductive_encoders(train_df, val_df, test_df, cat_cols):
    """
    Fits an OrdinalEncoder on the concatenation of Train, Validation, and Test sets
    to ensure vocabulary alignment across all splits.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        cat_cols (list): List of categorical column names.

    Returns:
        tuple: (fitted encoder, list of vocabulary sizes)
    """
    full_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    encoder.fit(full_cat)

    # Calculate vocab sizes (number of unique categories per column)
    vocab_sizes = [int(full_cat[col].nunique()) for col in cat_cols]

    return encoder, vocab_sizes


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.

    Steps:
    1. Checks for cached parquet/npy files.
    2. If not found or load_cached_data is False:
       - Loads raw metadata CSVs.
       - Applies feature engineering (f_27 decomposition).
       - Performs transductive categorical encoding.
       - Normalizes continuous features (fit on train, apply to all).
       - Saves processed data to cache.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df, meta_dict)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_files = {
        "train": os.path.join(Config.WORKING_DIR, "train_processed.parquet"),
        "val": os.path.join(Config.WORKING_DIR, "val_processed.parquet"),
        "test": os.path.join(Config.WORKING_DIR, "test_processed.parquet"),
        "meta": os.path.join(Config.WORKING_DIR, "meta.npy"),
    }

    # 1. Try Loading Cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached data...")
        try:
            train_df = pd.read_parquet(cache_files["train"])
            val_df = pd.read_parquet(cache_files["val"])
            test_df = pd.read_parquet(cache_files["test"])
            meta = np.load(cache_files["meta"], allow_pickle=True).item()
            return train_df, val_df, test_df, meta
        except Exception as e:
            print(f"Failed to load cache: {e}. Processing from scratch...")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Apply Feature Engineering
    print("Applying feature engineering...")
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    test_df = preprocess_data(test_df)

    # Define Columns
    # Continuous: f_00 to f_28 (excluding f_27) + unique_character_count
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]
    # Categorical: f_29, f_30 + f_27_0...f_27_9
    cat_cols = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]

    # Transductive Encoding
    print("Encoding categorical features...")
    encoder, vocab_sizes = create_transductive_encoders(
        train_df, val_df, test_df, cat_cols
    )

    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    val_df[cat_cols] = encoder.transform(val_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Normalization (StandardScaler)
    print("Normalizing continuous features...")
    scaler = StandardScaler()
    train_df[cont_cols] = scaler.fit_transform(train_df[cont_cols])
    val_df[cont_cols] = scaler.transform(val_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    # Save to Cache
    print(f"Saving to cache at {Config.WORKING_DIR}...")
    train_df.to_parquet(cache_files["train"])
    val_df.to_parquet(cache_files["val"])
    test_df.to_parquet(cache_files["test"])

    meta = {"cont_cols": cont_cols, "cat_cols": cat_cols, "vocab_sizes": vocab_sizes}
    np.save(cache_files["meta"], meta)

    return train_df, val_df, test_df, meta


def get_dataloaders(
    train_df,
    val_df,
    test_df,
    meta,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for train, val, and test sets using the ManufacturingDataset class.

    Args:
        train_df (pd.DataFrame): Processed training data.
        val_df (pd.DataFrame): Processed validation data.
        test_df (pd.DataFrame): Processed test data.
        meta (dict): Metadata containing column lists.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of worker processes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = ManufacturingDataset(
        train_df, meta["cont_cols"], meta["cat_cols"], is_test=False
    )
    val_dataset = ManufacturingDataset(
        val_df, meta["cont_cols"], meta["cat_cols"], is_test=False
    )
    test_dataset = ManufacturingDataset(
        test_df, meta["cont_cols"], meta["cat_cols"], is_test=True
    )

    # Pin memory for faster host-to-device transfer if CUDA is available
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader
