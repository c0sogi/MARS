import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


def decompose_f27(df):
    """
    Decomposes the 'f_27' string column into 10 separate character columns
    and computes the unique character count as a new feature.
    """
    # Ensure f_27 is string
    s = df["f_27"].astype(str)

    # 1. Create 10 character columns
    # We assume f_27 always has length 10 based on dataset analysis
    # Using a vectorized approach with pandas str accessor
    for i in range(10):
        df[f"f_27_char_{i}"] = s.str[i]

    # 2. Compute unique character count
    # set(x) gets unique chars, len() counts them
    df["unique_character_count"] = s.apply(lambda x: len(set(x)))

    return df


def get_preprocessed_data(load_from_cache=True):
    """
    Loads, preprocesses, and returns the training, validation, and test data.
    Implements caching to speed up subsequent runs.

    Returns:
        train_data (dict): {'cat': np.array, 'cont': np.array, 'target': np.array}
        val_data (dict):   {'cat': np.array, 'cont': np.array, 'target': np.array}
        test_data (dict):  {'cat': np.array, 'cont': np.array, 'ids': np.array}
        vocab_sizes (list): List of integers representing vocabulary size for each categorical feature.
    """

    # Check if cache exists
    cache_files = [
        Config.TRAIN_PROCESSED_PARQUET,
        Config.VAL_PROCESSED_PARQUET,
        Config.TEST_PROCESSED_PARQUET,
        Config.VOCAB_SIZES_NPY,
    ]

    cache_exists = all(os.path.exists(f) for f in cache_files)

    if load_from_cache and cache_exists:
        print("Loading preprocessed data from cache...")
        train_df = pd.read_parquet(Config.TRAIN_PROCESSED_PARQUET)
        val_df = pd.read_parquet(Config.VAL_PROCESSED_PARQUET)
        test_df = pd.read_parquet(Config.TEST_PROCESSED_PARQUET)
        vocab_sizes = np.load(Config.VOCAB_SIZES_NPY).tolist()

        # Helper to extract arrays
        def extract_arrays(df, is_test=False):
            # Identify columns based on naming convention
            cat_cols = [c for c in df.columns if c.startswith("cat_")]
            cont_cols = [c for c in df.columns if c.startswith("cont_")]

            # Sort to ensure order is preserved
            cat_cols.sort()
            cont_cols.sort()

            X_cat = df[cat_cols].values.astype(np.int64)
            X_cont = df[cont_cols].values.astype(np.float32)

            if is_test:
                ids = df["id"].values
                return {"cat": X_cat, "cont": X_cont, "ids": ids}
            else:
                y = df["target"].values.astype(np.float32)
                return {"cat": X_cat, "cont": X_cont, "target": y}

        return (
            extract_arrays(train_df),
            extract_arrays(val_df),
            extract_arrays(test_df, is_test=True),
            vocab_sizes,
        )

    print("Processing data from scratch...")

    # 1. Load Data
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Feature Engineering (Decompose f_27)
    df_train = decompose_f27(df_train)
    df_val = decompose_f27(df_val)
    df_test = decompose_f27(df_test)

    # 3. Define Column Groups
    # Categorical: f_29, f_30, and the 10 chars from f_27
    # Note: f_27 original column is dropped/ignored after decomposition
    cat_features = [f"f_27_char_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00 to f_28 (excluding f_27) + unique_character_count
    # Generate f_00 ... f_28 list
    original_cont_features = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_features = original_cont_features + ["unique_character_count"]

    # 4. Transductive Ordinal Encoding
    # Concatenate all data to ensure global vocabulary
    all_cat = pd.concat(
        [df_train[cat_features], df_val[cat_features], df_test[cat_features]], axis=0
    )

    encoder = OrdinalEncoder(dtype=np.int64)
    all_cat_encoded = encoder.fit_transform(all_cat)

    # Get vocabulary sizes (max index + 1) for embedding layers
    vocab_sizes = [
        int(all_cat_encoded[:, i].max() + 1) for i in range(len(cat_features))
    ]

    # Split back
    n_train = len(df_train)
    n_val = len(df_val)

    train_cat_encoded = all_cat_encoded[:n_train]
    val_cat_encoded = all_cat_encoded[n_train : n_train + n_val]
    test_cat_encoded = all_cat_encoded[n_train + n_val :]

    # 5. Standard Scaling
    # Fit on TRAIN only, transform all
    scaler = StandardScaler()
    train_cont_scaled = scaler.fit_transform(df_train[cont_features])
    val_cont_scaled = scaler.transform(df_val[cont_features])
    test_cont_scaled = scaler.transform(df_test[cont_features])

    # 6. Prepare DataFrames for Caching (Parquet is efficient)
    # We rename columns to generic cat_0, cat_1... cont_0, cont_1... for easy retrieval
    def create_processed_df(cat_arr, cont_arr, original_df, is_test=False):
        data = {}

        # Add ID
        data["id"] = original_df["id"].values

        # Add Target if not test
        if not is_test:
            data["target"] = original_df["target"].values

        # Add Categorical
        for i in range(cat_arr.shape[1]):
            data[f"cat_{i:02d}"] = cat_arr[:, i]

        # Add Continuous
        for i in range(cont_arr.shape[1]):
            data[f"cont_{i:02d}"] = cont_arr[:, i]

        return pd.DataFrame(data)

    train_processed_df = create_processed_df(
        train_cat_encoded, train_cont_scaled, df_train
    )
    val_processed_df = create_processed_df(val_cat_encoded, val_cont_scaled, df_val)
    test_processed_df = create_processed_df(
        test_cat_encoded, test_cont_scaled, df_test, is_test=True
    )

    # 7. Save to Cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    train_processed_df.to_parquet(Config.TRAIN_PROCESSED_PARQUET, index=False)
    val_processed_df.to_parquet(Config.VAL_PROCESSED_PARQUET, index=False)
    test_processed_df.to_parquet(Config.TEST_PROCESSED_PARQUET, index=False)
    np.save(Config.VOCAB_SIZES_NPY, np.array(vocab_sizes))

    # 8. Return formatted data
    # We can reuse the loading logic helper or just construct dicts directly here.
    # Constructing directly to avoid reloading from disk immediately.

    train_data = {
        "cat": train_cat_encoded.astype(np.int64),
        "cont": train_cont_scaled.astype(np.float32),
        "target": df_train["target"].values.astype(np.float32),
    }

    val_data = {
        "cat": val_cat_encoded.astype(np.int64),
        "cont": val_cont_scaled.astype(np.float32),
        "target": df_val["target"].values.astype(np.float32),
    }

    test_data = {
        "cat": test_cat_encoded.astype(np.int64),
        "cont": test_cont_scaled.astype(np.float32),
        "ids": df_test["id"].values,
    }

    return train_data, val_data, test_data, vocab_sizes
