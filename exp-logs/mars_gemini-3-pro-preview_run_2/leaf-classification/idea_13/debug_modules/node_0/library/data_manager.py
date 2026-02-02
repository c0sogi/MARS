import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library import config, image_features


def load_and_prepare_data(load_cached_data=True):
    """
    Loads metadata, performs morphological augmentation, combines train/val sets,
    encodes labels, and prepares feature matrices.

    Implements caching for the final processed numpy arrays.

    Args:
        load_cached_data (bool): If True, attempts to load processed arrays from cache.

    Returns:
        tuple: (X_train, y_train, X_test, test_ids, classes)
            - X_train (np.ndarray): Combined training and validation features.
            - y_train (np.ndarray): Encoded labels for X_train.
            - X_test (np.ndarray): Test set features.
            - test_ids (np.ndarray): IDs corresponding to X_test rows.
            - classes (np.ndarray): Array of class names corresponding to encoded labels.
    """
    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    x_train_path = os.path.join(cache_dir, "X_train.npy")
    y_train_path = os.path.join(cache_dir, "y_train.npy")
    x_test_path = os.path.join(cache_dir, "X_test.npy")
    test_ids_path = os.path.join(cache_dir, "test_ids.npy")
    classes_path = os.path.join(cache_dir, "classes.npy")

    # Check if all cache files exist
    cache_exists = all(
        os.path.exists(p)
        for p in [x_train_path, y_train_path, x_test_path, test_ids_path, classes_path]
    )

    if load_cached_data and cache_exists:
        # Load from cache
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        test_ids = np.load(test_ids_path)
        classes = np.load(classes_path, allow_pickle=True)
        return X_train, y_train, X_test, test_ids, classes

    # If cache miss or force reload, process from scratch

    # 1. Load Metadata
    df_train_meta = pd.read_csv(config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(config.VAL_META_PATH)
    df_test_meta = pd.read_csv(config.TEST_META_PATH)

    # 2. Augment Data with Morphological Features
    # The augment_dataframe function handles its own caching of the intermediate parquet files
    df_train_aug = image_features.augment_dataframe(
        df_train_meta, load_cached_data=load_cached_data, cache_name="train_augmented"
    )
    df_val_aug = image_features.augment_dataframe(
        df_val_meta, load_cached_data=load_cached_data, cache_name="val_augmented"
    )
    df_test_aug = image_features.augment_dataframe(
        df_test_meta, load_cached_data=load_cached_data, cache_name="test_augmented"
    )

    # 3. Combine Training and Validation Sets
    # As per strategy, we train on the full available labeled data
    df_train_full = pd.concat([df_train_aug, df_val_aug], axis=0, ignore_index=True)

    # 4. Prepare Features and Targets
    # Identify feature columns: all columns except metadata and target
    non_feature_cols = ["id", "species", "image_path"]
    feature_cols = [c for c in df_train_full.columns if c not in non_feature_cols]

    # Sort feature columns to ensure consistent order between train and test
    feature_cols = sorted(feature_cols)

    X_train = df_train_full[feature_cols].values.astype(np.float32)
    y_train_raw = df_train_full["species"].values

    # 5. Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    classes = le.classes_

    # 6. Prepare Test Data
    test_ids = df_test_aug["id"].values
    X_test = df_test_aug[feature_cols].values.astype(np.float32)

    # 7. Save to Cache
    np.save(x_train_path, X_train)
    np.save(y_train_path, y_train)
    np.save(x_test_path, X_test)
    np.save(test_ids_path, test_ids)
    np.save(classes_path, classes)

    return X_train, y_train, X_test, test_ids, classes
