import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


def load_raw_data():
    """
    Loads raw data from the metadata paths defined in Config.
    Returns train, val, and test DataFrames.
    """
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    return train_df, val_df, test_df


def engineer_features(df):
    """
    Performs feature engineering:
    1. Decomposes f_27 into 10 character columns.
    2. Computes unique_character_count from f_27.
    """
    # Decompose f_27 into 10 separate columns
    # Vectorized string slicing is efficient
    for i in range(10):
        df[f"f_27_{i}"] = df["f_27"].str[i]

    # Compute unique character count
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

    return df


def _prepare_output(train_df, val_df, test_df, metadata):
    """
    Helper function to format the output dictionary with numpy arrays.
    """
    cat_cols = metadata["cat_cols"]
    cont_cols = metadata["cont_cols"]
    target_col = metadata["target_col"]
    id_col = metadata["id_col"]

    output = {
        "train": {
            "X_cat": train_df[cat_cols].values.astype(np.int64),
            "X_cont": train_df[cont_cols].values.astype(np.float32),
            "y": train_df[target_col].values.astype(np.float32),
        },
        "val": {
            "X_cat": val_df[cat_cols].values.astype(np.int64),
            "X_cont": val_df[cont_cols].values.astype(np.float32),
            "y": val_df[target_col].values.astype(np.float32),
        },
        "test": {
            "X_cat": test_df[cat_cols].values.astype(np.int64),
            "X_cont": test_df[cont_cols].values.astype(np.float32),
            "id": test_df[id_col].values,
        },
        "meta": {
            "vocab_sizes": metadata["vocab_sizes"],
            "cat_cols": cat_cols,
            "cont_cols": cont_cols,
        },
    }
    return output


def get_data(load_cached_data=True):
    """
    Main function to get processed data.
    Handles caching, feature engineering, encoding, and scaling.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        dict: A dictionary containing 'train', 'val', 'test' data dictionaries and 'meta'.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_processed.parquet")
    val_cache_path = os.path.join(cache_dir, "val_processed.parquet")
    test_cache_path = os.path.join(cache_dir, "test_processed.parquet")
    meta_cache_path = os.path.join(cache_dir, "metadata.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
            and os.path.exists(meta_cache_path)
        ):

            print("Loading data from cache...")
            train_df = pd.read_parquet(train_cache_path)
            val_df = pd.read_parquet(val_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            metadata = np.load(meta_cache_path, allow_pickle=True).item()

            return _prepare_output(train_df, val_df, test_df, metadata)
        else:
            print("Cache missing or incomplete. Processing from scratch...")
    else:
        print("Ignoring cache. Processing from scratch...")

    # 2. Load Raw Data
    print("Loading raw data...")
    train_df, val_df, test_df = load_raw_data()

    # 3. Feature Engineering
    print("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Define Column Groups
    # Continuous: f_00 to f_28 (excluding f_27), plus unique_character_count
    # Note: f_28 is continuous based on range analysis (-1229 to 1157)
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_cols.append("unique_character_count")

    # Categorical: f_29, f_30, and decomposed f_27 characters
    cat_cols = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]

    # 4. Transductive Ordinal Encoding
    print("Encoding categorical features (Transductive)...")
    # Concatenate all splits to ensure global vocabulary
    # Convert to string to handle mixed types (numeric f_29/30 vs char f_27_x)
    all_cats = pd.concat(
        [
            train_df[cat_cols].astype(str),
            val_df[cat_cols].astype(str),
            test_df[cat_cols].astype(str),
        ],
        axis=0,
    )

    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(all_cats)

    # Transform each split
    train_df[cat_cols] = encoder.transform(train_df[cat_cols].astype(str))
    val_df[cat_cols] = encoder.transform(val_df[cat_cols].astype(str))
    test_df[cat_cols] = encoder.transform(test_df[cat_cols].astype(str))

    # Calculate vocab sizes for embeddings
    vocab_sizes = [len(cats) for cats in encoder.categories_]

    # 5. Standard Scaling
    print("Scaling continuous features...")
    scaler = StandardScaler()
    # Fit only on training data
    scaler.fit(train_df[cont_cols])

    # Transform all splits
    train_df[cont_cols] = scaler.transform(train_df[cont_cols])
    val_df[cont_cols] = scaler.transform(val_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    # Optimize types
    train_df[cont_cols] = train_df[cont_cols].astype(np.float32)
    val_df[cont_cols] = val_df[cont_cols].astype(np.float32)
    test_df[cont_cols] = test_df[cont_cols].astype(np.float32)

    # 6. Save to Cache
    print("Saving processed data to cache...")

    metadata = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "vocab_sizes": vocab_sizes,
        "target_col": Config.TARGET_COL,
        "id_col": Config.ID_COL,
    }

    train_df.to_parquet(train_cache_path)
    val_df.to_parquet(val_cache_path)
    test_df.to_parquet(test_cache_path)
    np.save(meta_cache_path, metadata)

    return _prepare_output(train_df, val_df, test_df, metadata)
