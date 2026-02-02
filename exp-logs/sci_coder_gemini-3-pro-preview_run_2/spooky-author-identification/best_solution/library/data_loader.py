import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from library.utils import set_seed, generate_config_hash

# Constants
CACHE_DIR = "./working/idea_3"
METADATA_DIR = "./metadata"


def load_and_process_data(config, load_cached_data=True):
    """
    Loads data from metadata, merges train/val for CV, encodes labels,
    creates stratified folds, and caches the result.

    Args:
        config (dict): Configuration dictionary containing:
            - seed (int): Random seed.
            - n_folds (int): Number of CV folds.
            - debug (bool): Whether to use a subset of data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, test_df, label_classes)
            - train_df (pd.DataFrame): Labeled data with 'fold' and 'author_encoded' columns.
            - test_df (pd.DataFrame): Test data.
            - label_classes (np.ndarray): Array of class names corresponding to encodings.
    """
    # Ensure reproducibility
    set_seed(config.get("seed", 42))

    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate config hash for cache versioning
    config_hash = generate_config_hash(config)

    # Define cache file paths
    train_cache_path = os.path.join(CACHE_DIR, f"train_folds_{config_hash}.parquet")
    test_cache_path = os.path.join(CACHE_DIR, f"test_{config_hash}.parquet")
    classes_cache_path = os.path.join(CACHE_DIR, f"classes_{config_hash}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(test_cache_path)
            and os.path.exists(classes_cache_path)
        ):

            print(f"Loading data from cache: {CACHE_DIR}")
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            label_classes = np.load(classes_cache_path, allow_pickle=True)
            return train_df, test_df, label_classes

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load metadata
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    if not all(
        os.path.exists(p) for p in [train_meta_path, val_meta_path, test_meta_path]
    ):
        raise FileNotFoundError(f"Metadata files missing in {METADATA_DIR}")

    df_train_part = pd.read_csv(train_meta_path)
    df_val_part = pd.read_csv(val_meta_path)
    df_test = pd.read_csv(test_meta_path)

    # Merge train and val to create a full training set for CV
    # We ignore the specific split in metadata for the purpose of creating our own K-Folds
    df_train = pd.concat([df_train_part, df_val_part], axis=0).reset_index(drop=True)

    # Handle Debug Mode
    if config.get("debug", False):
        print("Debug mode enabled: Subsampling data.")
        df_train = df_train.sample(
            n=min(1000, len(df_train)), random_state=config.get("seed", 42)
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(500, len(df_test)), random_state=config.get("seed", 42)
        ).reset_index(drop=True)

    # Label Encoding
    le = LabelEncoder()
    df_train["author_encoded"] = le.fit_transform(df_train["author"])
    label_classes = le.classes_

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=config.get("n_folds", 5),
        shuffle=True,
        random_state=config.get("seed", 42),
    )
    df_train["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train, df_train["author_encoded"])
    ):
        df_train.loc[val_idx, "fold"] = fold

    # Save to cache
    print(f"Saving processed data to cache: {CACHE_DIR}")
    df_train.to_parquet(train_cache_path, index=False)
    df_test.to_parquet(test_cache_path, index=False)
    np.save(classes_cache_path, label_classes)

    return df_train, df_test, label_classes
