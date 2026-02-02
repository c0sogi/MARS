import os
import numpy as np
import pandas as pd
from library import config, utils, image_features


def get_feature_subsets(columns):
    """
    Identifies and groups feature columns into subsets based on their prefixes.

    Args:
        columns: List or Index of column names.

    Returns:
        dict: A dictionary mapping subset names ('global', 'margin', 'shape',
              'texture', 'morphometrics') to lists of column names.
    """
    # Define prefixes from config
    p_margin = config.FEATURE_PREFIXES["margin"]
    p_shape = config.FEATURE_PREFIXES["shape"]
    p_texture = config.FEATURE_PREFIXES["texture"]

    # Identify columns by prefix
    margin_cols = [c for c in columns if c.startswith(p_margin)]
    shape_cols = [c for c in columns if c.startswith(p_shape)]
    texture_cols = [c for c in columns if c.startswith(p_texture)]

    # Global view is defined as the original 192 features (Margin + Shape + Texture)
    global_cols = margin_cols + shape_cols + texture_cols

    # Morphometrics are the columns that are not in the global set
    # We assume 'id', 'species', 'image_path' are already removed from the input 'columns'
    # or we explicitly filter them out if 'columns' comes from a raw df.
    # Here we assume 'columns' are the feature columns of the processed X dataframe.
    morph_cols = [c for c in columns if c not in global_cols]

    return {
        "global": global_cols,
        "margin": margin_cols,
        "shape": shape_cols,
        "texture": texture_cols,
        "morphometrics": morph_cols,
    }


def load_datasets(load_cached_data=True):
    """
    Loads the training, validation, and test datasets.
    Merges provided tabular features with extracted morphometric features.
    Implements caching using Parquet (for DataFrames) and NPY (for arrays).

    Args:
        load_cached_data (bool): If True, attempts to load processed data from disk.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes, feature_subsets)
            - X_train, X_val, X_test: pd.DataFrame of features (float64).
            - y_train, y_val: np.ndarray of class labels (strings).
            - test_ids: np.ndarray of test image IDs (integers).
            - classes: np.ndarray of unique class names.
            - feature_subsets: dict mapping view names to column lists.
    """
    # Define cache file paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_X_train = os.path.join(cache_dir, "X_train.parquet")
    path_y_train = os.path.join(cache_dir, "y_train.npy")
    path_X_val = os.path.join(cache_dir, "X_val.parquet")
    path_y_val = os.path.join(cache_dir, "y_val.npy")
    path_X_test = os.path.join(cache_dir, "X_test.parquet")
    path_test_ids = os.path.join(cache_dir, "test_ids.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(path_X_train)
            and os.path.exists(path_y_train)
            and os.path.exists(path_X_val)
            and os.path.exists(path_y_val)
            and os.path.exists(path_X_test)
            and os.path.exists(path_test_ids)
        ):

            print("Loading datasets from cache...")
            X_train = pd.read_parquet(path_X_train)
            y_train = np.load(path_y_train, allow_pickle=True)
            X_val = pd.read_parquet(path_X_val)
            y_val = np.load(path_y_val, allow_pickle=True)
            X_test = pd.read_parquet(path_X_test)
            test_ids = np.load(path_test_ids, allow_pickle=True)

            classes = np.unique(y_train)
            feature_subsets = get_feature_subsets(X_train.columns)

            return (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                test_ids,
                classes,
                feature_subsets,
            )
        else:
            print("Cache missing or incomplete. Processing from scratch...")

    # 2. Load Metadata
    print("Loading metadata CSVs...")
    df_train_meta = pd.read_csv(config.TRAIN_PATH)
    df_val_meta = pd.read_csv(config.VAL_PATH)
    df_test_meta = pd.read_csv(config.TEST_PATH)

    # 3. Extract Morphometrics (Handles its own caching)
    print("Processing morphometrics...")
    df_train_morph = image_features.process_all_images(
        df_train_meta, "train", load_cached_data
    )
    df_val_morph = image_features.process_all_images(
        df_val_meta, "val", load_cached_data
    )
    df_test_morph = image_features.process_all_images(
        df_test_meta, "test", load_cached_data
    )

    # 4. Merge Data
    # Inner join on 'id' to combine tabular features with morphometrics
    print("Merging features...")
    df_train_full = pd.merge(df_train_meta, df_train_morph, on="id", how="inner")
    df_val_full = pd.merge(df_val_meta, df_val_morph, on="id", how="inner")
    df_test_full = pd.merge(df_test_meta, df_test_morph, on="id", how="inner")

    # 5. Prepare X and y
    def process_split(df, is_test=False):
        # Drop non-feature columns
        drop_cols = ["id", "image_path"]
        if "species" in df.columns:
            drop_cols.append("species")

        # Extract features
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Enforce float64
        X = utils.enforce_float64(X)

        if is_test:
            return X, df["id"].values
        else:
            return X, df["species"].values

    X_train, y_train = process_split(df_train_full, is_test=False)
    X_val, y_val = process_split(df_val_full, is_test=False)
    X_test, test_ids = process_split(df_test_full, is_test=True)

    # 6. Save to Cache
    print("Saving processed datasets to cache...")
    X_train.to_parquet(path_X_train, index=False)
    np.save(path_y_train, y_train)

    X_val.to_parquet(path_X_val, index=False)
    np.save(path_y_val, y_val)

    X_test.to_parquet(path_X_test, index=False)
    np.save(path_test_ids, test_ids)

    # 7. Finalize
    classes = np.unique(y_train)
    feature_subsets = get_feature_subsets(X_train.columns)

    print(f"Data loading complete.")
    print(
        f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}"
    )
    print(f"Number of classes: {len(classes)}")

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes, feature_subsets
