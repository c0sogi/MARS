import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from library.config import Config


def decompose_f27(series):
    """
    Decomposes the string feature 'f_27' into a sequence of integer tokens.
    Maps 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26.

    Args:
        series (pd.Series): A pandas Series containing the string feature.

    Returns:
        np.ndarray: A numpy array of shape (N, 10) with integer encodings.
    """
    # Convert series to list of strings for processing
    strings = series.astype(str).tolist()

    # Vectorized list comprehension for speed
    # ord('A') is 65. We want 'A' -> 1, so we subtract 64.
    # We slice [:Config.F_27_SEQ_LENGTH] to ensure fixed length (10).
    encoded_data = [
        [ord(char) - 64 for char in s[: Config.F_27_SEQ_LENGTH]] for s in strings
    ]

    return np.array(encoded_data, dtype=np.int32)


def process_data(load_cached_data=True):
    """
    Loads, processes, and splits the data based on metadata.
    Implements caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: A dictionary containing numpy arrays for train/val/test splits.
              Keys: X_num_train, X_cat_train, y_train,
                    X_num_val, X_cat_val, y_val,
                    X_num_test, X_cat_test
    """

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(Config.PROCESSED_DATA_PATH):
        print(f"Loading cached processed data from {Config.PROCESSED_DATA_PATH}...")
        try:
            loaded = np.load(Config.PROCESSED_DATA_PATH)
            # Convert NpzFile to a standard dict to allow modification (e.g., debug slicing)
            data_dict = {k: loaded[k] for k in loaded.files}

            # Apply DEBUG slicing if enabled
            if Config.DEBUG:
                print(
                    f"DEBUG mode enabled: Slicing data to {Config.DEBUG_SAMPLE_SIZE} samples."
                )
                limit = Config.DEBUG_SAMPLE_SIZE
                for key in data_dict:
                    data_dict[key] = data_dict[key][:limit]

            return data_dict
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load Metadata
    print("Loading metadata...")
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # Load Raw Data
    # We load the full CSVs once.
    print("Loading raw CSV files...")
    df_train_full = pd.read_csv(Config.TRAIN_CSV)
    df_test_full = pd.read_csv(Config.TEST_CSV)

    # Index by ID for fast lookup and alignment
    df_train_full.set_index("id", inplace=True)
    df_test_full.set_index("id", inplace=True)

    # 3. Align Data with Metadata
    # We use the IDs from metadata to extract the exact rows for each split
    print("Aligning data splits...")
    X_train_raw = df_train_full.loc[train_meta["id"]]
    X_val_raw = df_train_full.loc[val_meta["id"]]
    X_test_raw = df_test_full.loc[test_meta["id"]]

    # Extract Targets
    y_train = train_meta["target"].values.astype(np.float32)
    y_val = val_meta["target"].values.astype(np.float32)

    # 4. Feature Engineering

    # Identify Continuous Columns: f_00 to f_30, excluding f_27 and target
    # We filter purely based on name pattern to be robust
    cont_cols = [
        c
        for c in df_train_full.columns
        if c.startswith("f_") and c != "f_27" and c != "target"
    ]
    cont_cols.sort()  # Ensure deterministic order

    if len(cont_cols) != Config.NUM_CONTINUOUS_FEATURES:
        raise ValueError(
            f"Expected {Config.NUM_CONTINUOUS_FEATURES} continuous features, found {len(cont_cols)}"
        )

    print("Standardizing continuous features...")
    scaler = StandardScaler()

    # Fit on Train ONLY to prevent leakage
    X_num_train = scaler.fit_transform(X_train_raw[cont_cols].values.astype(np.float32))

    # Transform Val and Test using statistics from Train
    X_num_val = scaler.transform(X_val_raw[cont_cols].values.astype(np.float32))
    X_num_test = scaler.transform(X_test_raw[cont_cols].values.astype(np.float32))

    # Process Categorical Feature (f_27)
    print("Decomposing categorical feature f_27...")
    X_cat_train = decompose_f27(X_train_raw["f_27"])
    X_cat_val = decompose_f27(X_val_raw["f_27"])
    X_cat_test = decompose_f27(X_test_raw["f_27"])

    # 5. Save to Cache
    print(f"Saving processed data to {Config.PROCESSED_DATA_PATH}...")
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)

    np.savez_compressed(
        Config.PROCESSED_DATA_PATH,
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_num_val=X_num_val,
        X_cat_val=X_cat_val,
        y_val=y_val,
        X_num_test=X_num_test,
        X_cat_test=X_cat_test,
    )

    # Construct return dictionary
    data_dict = {
        "X_num_train": X_num_train,
        "X_cat_train": X_cat_train,
        "y_train": y_train,
        "X_num_val": X_num_val,
        "X_cat_val": X_cat_val,
        "y_val": y_val,
        "X_num_test": X_num_test,
        "X_cat_test": X_cat_test,
    }

    # Apply DEBUG slicing if enabled (post-processing logic)
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled: Slicing data to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        limit = Config.DEBUG_SAMPLE_SIZE
        for key in data_dict:
            data_dict[key] = data_dict[key][:limit]

    return data_dict
