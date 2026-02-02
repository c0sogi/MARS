import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import library.config as conf
import library.image_processing as img_proc


def load_data(load_cached_data=True):
    """
    Loads the dataset, processes features (tabular + morphometric), and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays
                                 from the cache directory.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes)
            - X_train (np.ndarray): Training features (float64).
            - y_train (np.ndarray): Training labels (int).
            - X_val (np.ndarray): Validation features (float64).
            - y_val (np.ndarray): Validation labels (int).
            - X_test (np.ndarray): Test features (float64).
            - test_ids (np.ndarray): Test image IDs.
            - classes (np.ndarray): Array of class names (strings) corresponding to label indices.
    """
    # Ensure cache directory exists
    os.makedirs(conf.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(conf.CACHE_DIR, "data_cache.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                X_train = data["X_train"]
                y_train = data["y_train"]
                X_val = data["X_val"]
                y_val = data["y_val"]
                X_test = data["X_test"]
                test_ids = data["test_ids"]
                classes = data["classes"]
            return X_train, y_train, X_val, y_val, X_test, test_ids, classes
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing from scratch...")

    # 2. Load Metadata
    print("Loading metadata CSVs...")
    df_train = pd.read_csv(conf.TRAIN_DATA_PATH)
    df_val = pd.read_csv(conf.VAL_DATA_PATH)
    df_test = pd.read_csv(conf.TEST_DATA_PATH)

    # 3. Process Splits
    # We define a helper to process each split consistently
    def process_split(df, split_name):
        # A. Extract Tabular Features (Margin, Shape, Texture)
        # These are already in the CSV
        X_tabular = df[conf.ALL_TABULAR_COLS].values.astype(conf.FLOAT_PRECISION)

        # B. Extract Morphometric Features (Image-based)
        # This function handles its own caching of the intermediate parquet file
        df_morph = img_proc.process_dataset_morphometrics(
            df, dataset_name=split_name, load_cached_data=load_cached_data
        )
        X_morph = df_morph.values.astype(conf.FLOAT_PRECISION)

        # C. Combine Features
        # Concatenate tabular and morphometric features horizontally
        X_combined = np.hstack([X_tabular, X_morph])

        return X_combined

    print("Processing Training Data...")
    X_train = process_split(df_train, "train")

    print("Processing Validation Data...")
    X_val = process_split(df_val, "val")

    print("Processing Test Data...")
    X_test = process_split(df_test, "test")
    test_ids = df_test[conf.ID_COL].values

    # 4. Process Labels
    print("Encoding Labels...")
    le = LabelEncoder()
    # Fit on training species.
    # Note: Validation set is stratified, so it should contain same classes.
    # However, to be safe and consistent with competition format, we rely on train classes.
    y_train = le.fit_transform(df_train[conf.TARGET_COL])
    y_val = le.transform(df_val[conf.TARGET_COL])
    classes = le.classes_

    # 5. Save to Cache
    print(f"Saving processed data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        test_ids=test_ids,
        classes=classes,
    )

    print("Data loading complete.")
    print(f"Feature Matrix Shape: {X_train.shape} (Rows, Cols)")

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
