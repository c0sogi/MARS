import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from library import config, features


def load_dataset(metadata_path, macro_cache_path, load_cached_data=True):
    """
    Loads the dataset from metadata, extracts/loads macro features, and merges them.

    Args:
        metadata_path (str): Path to the metadata CSV (train/val/test).
        macro_cache_path (str): Path to the parquet cache for macro features.
        load_cached_data (bool): Whether to use cached macro features.

    Returns:
        pd.DataFrame: Merged DataFrame containing IDs, targets (if available),
                      Global features, and Macro features.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 1. Load Metadata (contains ID, Species, Global Features)
    df_meta = pd.read_csv(metadata_path)

    # 2. Get Macro Features (computed or cached via features.py)
    # features.process_image_batch handles the caching logic internally
    df_macro = features.process_image_batch(
        df_meta, macro_cache_path, load_cached_data=load_cached_data
    )

    # 3. Merge
    # Ensure 'id' is the join key.
    # df_macro has 'id' as a column and matches the index of df_meta if passed correctly.
    # To be safe, we merge on 'id'.
    if "id" in df_macro.columns:
        df_merged = pd.merge(df_meta, df_macro, on="id", how="left")
    else:
        # Fallback if id is not in columns but index is aligned
        df_merged = pd.concat([df_meta, df_macro], axis=1)

    return df_merged


def prepare_matrices(df, is_train=True, label_encoder=None):
    """
    Splits the DataFrame into Global and Macro feature matrices and targets.
    Strictly casts features to float64.

    Args:
        df (pd.DataFrame): The merged DataFrame.
        is_train (bool): Whether this is a training/validation set (has labels).
        label_encoder (LabelEncoder, optional): Encoder for species.

    Returns:
        tuple:
            If is_train: (X_global, X_macro, y, label_encoder)
            If not is_train: (X_global, X_macro, ids)
    """
    # 1. Identify Feature Columns
    # Global Features: margin_*, shape_*, texture_*
    global_cols = [
        c for c in df.columns if c.startswith(("margin", "shape", "texture"))
    ]

    # Macro Features: hu_*, aspect_ratio, solidity, extent, eccentricity
    # We identify them by exclusion or explicit list based on features.py
    known_macro_cols = [f"hu_{i}" for i in range(1, 8)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]

    # Filter to ensure they exist in df
    macro_cols = [c for c in known_macro_cols if c in df.columns]

    # 2. Extract Matrices
    X_global = df[global_cols].values.astype(config.FLOAT_PRECISION)
    X_macro = df[macro_cols].values.astype(config.FLOAT_PRECISION)

    # 3. Handle Targets/IDs
    if is_train:
        if "species" not in df.columns:
            raise ValueError("Training data must contain 'species' column.")

        y_raw = df["species"].values

        if label_encoder is None:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)
        else:
            y = label_encoder.transform(y_raw)

        return X_global, X_macro, y, label_encoder
    else:
        ids = df["id"].values
        return X_global, X_macro, ids


def get_data(load_cached_data=True):
    """
    Orchestrates the loading of Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached feature extraction.

    Returns:
        dict: Dictionary containing:
            'train': (X_global, X_macro, y)
            'val': (X_global, X_macro, y)
            'test': (X_global, X_macro, ids)
            'label_encoder': The fitted LabelEncoder
            'feature_names': {'global': [...], 'macro': [...]}
    """
    print("Loading Train Data...")
    df_train = load_dataset(
        config.TRAIN_METADATA_PATH, config.CACHE_MACRO_TRAIN, load_cached_data
    )

    print("Loading Validation Data...")
    df_val = load_dataset(
        config.VAL_METADATA_PATH, config.CACHE_MACRO_VAL, load_cached_data
    )

    print("Loading Test Data...")
    df_test = load_dataset(
        config.TEST_METADATA_PATH, config.CACHE_MACRO_TEST, load_cached_data
    )

    # Prepare Train
    X_train_global, X_train_macro, y_train, le = prepare_matrices(
        df_train, is_train=True, label_encoder=None
    )

    # Prepare Val
    X_val_global, X_val_macro, y_val, _ = prepare_matrices(
        df_val, is_train=True, label_encoder=le
    )

    # Prepare Test
    X_test_global, X_test_macro, test_ids = prepare_matrices(df_test, is_train=False)

    # Store feature names for reference
    global_cols = [
        c for c in df_train.columns if c.startswith(("margin", "shape", "texture"))
    ]
    macro_cols = [f"hu_{i}" for i in range(1, 8)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]
    macro_cols = [c for c in macro_cols if c in df_train.columns]

    return {
        "train": (X_train_global, X_train_macro, y_train),
        "val": (X_val_global, X_val_macro, y_val),
        "test": (X_test_global, X_test_macro, test_ids),
        "label_encoder": le,
        "feature_names": {"global": global_cols, "macro": macro_cols},
    }
