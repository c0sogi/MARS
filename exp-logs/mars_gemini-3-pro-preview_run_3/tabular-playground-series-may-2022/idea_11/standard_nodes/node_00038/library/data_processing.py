import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import Config


def decompose_f27(df):
    """
    Decomposes the 'f_27' column into 10 separate character columns
    and adds a 'unique_character_count' feature.
    """
    # Extract the 10 characters
    # We expect f_27 to be a string of length 10
    # Vectorized string splitting
    f27_series = df["f_27"].astype(str)

    # Create 10 new columns for characters
    char_cols = []
    for i in range(Config.F_27_LENGTH):
        col_name = f"f_27_{i}"
        df[col_name] = f27_series.str[i]
        char_cols.append(col_name)

    # Compute unique character count
    # Apply a lambda is slow, but robust.
    # For speed on large data, we can map to sets, but this is sufficient.
    df["unique_character_count"] = f27_series.apply(lambda x: len(set(x))).astype(float)

    return df, char_cols


def process_and_cache_data(load_cached_data=True):
    """
    Loads data, performs feature engineering and preprocessing, and caches the result.
    Returns processed numpy arrays and metadata.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_files = [
        Config.CACHE_PATH_TRAIN,
        Config.CACHE_PATH_VAL,
        Config.CACHE_PATH_TEST,
        Config.METADATA_CACHE_PATH,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(Config.CACHE_PATH_TRAIN)
        val_df = pd.read_parquet(Config.CACHE_PATH_VAL)
        test_df = pd.read_parquet(Config.CACHE_PATH_TEST)

        # Load metadata (vocab sizes)
        metadata = np.load(Config.METADATA_CACHE_PATH, allow_pickle=True).item()
        vocab_sizes = metadata["vocab_sizes"]

        # Separate features and targets
        # The parquet files contain processed features + target
        # We need to separate them for the Dataset

        # Identify columns based on metadata or naming convention
        # We assume the columns in parquet are ordered: [cat_0, ..., cat_n, cont_0, ..., cont_m, target]
        # But safer to use the column names saved in metadata or reconstruct them.

        # Let's reconstruct column lists to be safe
        cat_cols = metadata["cat_cols"]
        cont_cols = metadata["cont_cols"]

        X_train_cat = train_df[cat_cols].values.astype(np.int64)
        X_train_cont = train_df[cont_cols].values.astype(np.float32)
        y_train = train_df["target"].values.astype(np.float32)

        X_val_cat = val_df[cat_cols].values.astype(np.int64)
        X_val_cont = val_df[cont_cols].values.astype(np.float32)
        y_val = val_df["target"].values.astype(np.float32)

        X_test_cat = test_df[cat_cols].values.astype(np.int64)
        X_test_cont = test_df[cont_cols].values.astype(np.float32)
        # Test has no target, or dummy target. The metadata/test.csv has no target.
        # We handle test ids separately
        test_ids = test_df["id"].values

        return (
            (X_train_cat, X_train_cont, y_train),
            (X_val_cat, X_val_cont, y_val),
            (X_test_cat, X_test_cont, test_ids),
            vocab_sizes,
        )

    print("Cache not found or ignored. Processing data from scratch...")

    # Load raw data using metadata paths
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # 1. Feature Engineering: Decompose f_27
    df_train, char_cols = decompose_f27(df_train)
    df_val, _ = decompose_f27(df_val)
    df_test, _ = decompose_f27(df_test)

    # Define Feature Groups
    # Categorical: f_29, f_30 + decomposed chars
    # Note: f_29 and f_30 are in Config.CATEGORICAL_COLS
    cat_cols = Config.CATEGORICAL_COLS + char_cols

    # Continuous: f_00..f_28 (excl 27) + unique_character_count
    cont_cols = Config.CONTINUOUS_COLS

    # 2. Transductive Categorical Encoding
    # Concatenate all sets to define global vocabulary
    all_cat_data = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    # Initialize Ordinal Encoder
    # handle_unknown is not strictly needed due to transductive approach,
    # but good practice.
    encoder = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )
    encoder.fit(all_cat_data)

    # Transform
    X_train_cat = encoder.transform(df_train[cat_cols])
    X_val_cat = encoder.transform(df_val[cat_cols])
    X_test_cat = encoder.transform(df_test[cat_cols])

    # Get vocab sizes (max index + 1)
    # We add 1 because indices are 0-based.
    vocab_sizes = [
        int(np.max(col_vals) + 1) for col_vals in encoder.transform(all_cat_data).T
    ]

    # 3. Continuous Normalization
    # Fit StandardScaler ONLY on Train
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    X_train_cont = scaler.transform(df_train[cont_cols])
    X_val_cont = scaler.transform(df_val[cont_cols])
    X_test_cont = scaler.transform(df_test[cont_cols])

    # 4. Prepare for Caching
    # We construct DataFrames to save as parquet
    # We need to keep column names to retrieve them easily later

    # Helper to create DF
    def create_processed_df(cat_data, cont_data, original_df, is_test=False):
        data = {}
        # Add categorical
        for i, col in enumerate(cat_cols):
            data[col] = cat_data[:, i]
        # Add continuous
        for i, col in enumerate(cont_cols):
            data[col] = cont_data[:, i]

        # Add ID
        data["id"] = original_df["id"].values

        # Add Target if not test
        if not is_test:
            data["target"] = original_df["target"].values

        return pd.DataFrame(data)

    train_processed_df = create_processed_df(X_train_cat, X_train_cont, df_train)
    val_processed_df = create_processed_df(X_val_cat, X_val_cont, df_val)
    test_processed_df = create_processed_df(
        X_test_cat, X_test_cont, df_test, is_test=True
    )

    # Save Parquet
    train_processed_df.to_parquet(Config.CACHE_PATH_TRAIN, index=False)
    val_processed_df.to_parquet(Config.CACHE_PATH_VAL, index=False)
    test_processed_df.to_parquet(Config.CACHE_PATH_TEST, index=False)

    # Save Metadata
    metadata = {
        "vocab_sizes": vocab_sizes,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
    }
    np.save(Config.METADATA_CACHE_PATH, metadata)

    print("Data processing complete and cached.")

    # Return numpy arrays
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    test_ids = df_test["id"].values

    return (
        (X_train_cat, X_train_cont, y_train),
        (X_val_cat, X_val_cont, y_val),
        (X_test_cat, X_test_cont, test_ids),
        vocab_sizes,
    )


def get_dataloaders(batch_size=None, load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test sets, plus vocab sizes.

    Returns:
        train_loader, val_loader, test_loader, vocab_sizes
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Get processed data (numpy arrays)
    train_data, val_data, test_data, vocab_sizes = process_and_cache_data(
        load_cached_data
    )

    X_train_cat, X_train_cont, y_train = train_data
    X_val_cat, X_val_cont, y_val = val_data
    X_test_cat, X_test_cont, test_ids = test_data

    # Convert to Tensors
    # Train
    train_cat_tensor = torch.tensor(X_train_cat, dtype=torch.long)
    train_cont_tensor = torch.tensor(X_train_cont, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Val
    val_cat_tensor = torch.tensor(X_val_cat, dtype=torch.long)
    val_cont_tensor = torch.tensor(X_val_cont, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Test
    test_cat_tensor = torch.tensor(X_test_cat, dtype=torch.long)
    test_cont_tensor = torch.tensor(X_test_cont, dtype=torch.float32)
    # For test dataset, we include IDs to track predictions
    test_id_tensor = torch.tensor(test_ids, dtype=torch.long)

    # Create Datasets
    train_dataset = TensorDataset(train_cat_tensor, train_cont_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_cat_tensor, val_cont_tensor, val_y_tensor)
    test_dataset = TensorDataset(test_cat_tensor, test_cont_tensor, test_id_tensor)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader, vocab_sizes
