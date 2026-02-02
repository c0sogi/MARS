import os
import pandas as pd
import numpy as np
from library.utils import set_seed
from library.image_features import process_images

# Define working directory for caching
CACHE_DIR = "./working/idea_49/"
os.makedirs(CACHE_DIR, exist_ok=True)


def load_and_merge_data(mode: str, load_cached_data: bool = True):
    """
    Loads data for a specific mode ('train', 'val', 'test'), merges provided
    tabular features with extracted image morphometrics, and returns formatted arrays.

    Args:
        mode (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load pre-processed numpy arrays from cache.

    Returns:
        tuple:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features) in float64.
            y (np.ndarray or None): Target labels array of shape (n_samples,) or None for test.
            ids (np.ndarray): Image IDs array of shape (n_samples,).
    """
    set_seed(42)

    # Define cache file paths
    cache_X = os.path.join(CACHE_DIR, f"X_{mode}.npy")
    cache_y = os.path.join(CACHE_DIR, f"y_{mode}.npy")
    cache_ids = os.path.join(CACHE_DIR, f"ids_{mode}.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if os.path.exists(cache_X) and os.path.exists(cache_ids):
            # Check if y exists (it won't for test, or might be None)
            # For train/val, y must exist. For test, it might not.
            if mode in ["train", "val"] and not os.path.exists(cache_y):
                pass  # Cache incomplete, proceed to processing
            else:
                print(f"Loading merged {mode} data from cache...")
                X = np.load(cache_X)
                ids = np.load(cache_ids)
                if mode in ["train", "val"]:
                    y = np.load(cache_y, allow_pickle=True)
                else:
                    y = None
                return X, y, ids

    print(f"Processing and merging data for {mode}...")

    # 2. Load Metadata
    metadata_path = f"./metadata/{mode}.csv"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # 3. Extract IDs and Targets
    ids = df["id"].values
    y = df["species"].values if "species" in df.columns else None

    # 4. Extract Provided Tabular Features
    # Columns: margin_1..64, shape_1..64, texture_1..64
    # We identify them by name pattern
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("margin_") or c.startswith("shape_") or c.startswith("texture_")
    ]
    # Sort to ensure consistent order (though metadata generation usually preserves it)
    # The prompt implies specific blocks, let's trust the column order in CSV or sort strictly if needed.
    # Given the column names have indices (e.g. margin_1, margin_2), simple sorting might fail (margin_10 < margin_2).
    # However, the provided features are just blocks. We take them as they appear in the file which is standard.
    X_tabular = df[feature_cols].values.astype(np.float64)

    # 5. Extract Image Features (Morphometrics)
    # Get relative image paths
    image_paths = df["image_path"].tolist()

    # Call image_features.process_images
    # This handles its own caching for the image processing part
    df_morph = process_images(
        image_paths, cache_name=f"{mode}_morph", load_cached_data=load_cached_data
    )
    X_morph = df_morph.values.astype(np.float64)

    # 6. Merge Features
    # Concatenate tabular and morphometric features horizontally
    X = np.hstack([X_tabular, X_morph])

    # 7. Save to Cache
    print(f"Saving merged {mode} data to cache...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)
    if y is not None:
        np.save(cache_y, y)

    return X, y, ids


def get_train_val_split(load_cached_data: bool = True):
    """
    Retrieves the stratified training and validation sets used for the Selection Phase.

    Returns:
        X_train, y_train, X_val, y_val
    """
    X_train, y_train, _ = load_and_merge_data(
        "train", load_cached_data=load_cached_data
    )
    X_val, y_val, _ = load_and_merge_data("val", load_cached_data=load_cached_data)

    return X_train, y_train, X_val, y_val


def get_full_train_data(load_cached_data: bool = True):
    """
    Retrieves the combined training and validation data used for the Final Retraining Phase.

    Returns:
        X_full, y_full
    """
    X_train, y_train, _ = load_and_merge_data(
        "train", load_cached_data=load_cached_data
    )
    X_val, y_val, _ = load_and_merge_data("val", load_cached_data=load_cached_data)

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    return X_full, y_full


def get_test_data(load_cached_data: bool = True):
    """
    Retrieves the test set features and IDs.

    Returns:
        X_test, ids_test
    """
    X_test, _, ids_test = load_and_merge_data("test", load_cached_data=load_cached_data)
    return X_test, ids_test
