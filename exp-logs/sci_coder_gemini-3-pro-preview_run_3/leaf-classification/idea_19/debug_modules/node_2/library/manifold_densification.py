import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import load_numpy, save_numpy, ensure_directory
from library.feature_extraction import extract_dataset_features


def compute_orthogonal_centroids(features: np.ndarray) -> np.ndarray:
    """
    Transforms (N, 36, D) features into (N, 9, D) centroids by averaging
    mutually exclusive sets of 4 orthogonal views.

    The 36 views correspond to angles 0, 10, ..., 350.
    Centroid k (0..8) is the average of views {k, k+9, k+18, k+27}
    which correspond to angles {10k, 10k+90, 10k+180, 10k+270}.

    Args:
        features: Input features of shape (N, 36, D).

    Returns:
        Centroids of shape (N, 9, D).
    """
    if features.shape[1] != Config.NUM_ROTATIONS:
        raise ValueError(
            f"Expected {Config.NUM_ROTATIONS} rotations, got {features.shape[1]}"
        )

    N, R, D = features.shape

    # Reshape to (N, 4, 9, D)
    # Dimension 1 (size 4) represents the orthogonal group (0, 90, 180, 270 offsets)
    # Dimension 2 (size 9) represents the base angle index (0..8)
    reshaped = features.reshape(N, 4, 9, D)

    # Average over the orthogonal views (axis 1)
    centroids = reshaped.mean(axis=1)  # Result: (N, 9, D)

    return centroids


def load_tabular_data(metadata_path: str, ids: np.ndarray) -> np.ndarray:
    """
    Loads tabular features from metadata CSV, aligned with the provided IDs.
    """
    df = pd.read_csv(metadata_path)
    df = df.set_index("id")

    # Ensure alignment
    try:
        df_sorted = df.loc[ids]
    except KeyError as e:
        raise KeyError(f"Some IDs in the provided array are missing from metadata: {e}")

    # Extract feature columns
    cols = []
    for prefix in Config.TABULAR_PREFIXES:
        cols.extend([c for c in df_sorted.columns if c.startswith(prefix)])

    if not cols:
        raise ValueError("No tabular feature columns found in metadata.")

    return df_sorted[cols].values.astype(np.float32)


def load_labels(metadata_path: str, ids: np.ndarray) -> np.ndarray:
    """
    Loads labels from metadata CSV, aligned with the provided IDs.
    """
    df = pd.read_csv(metadata_path)
    df = df.set_index("id")

    try:
        df_sorted = df.loc[ids]
    except KeyError:
        raise KeyError("IDs mismatch in label loading.")

    if "species" not in df_sorted.columns:
        raise ValueError("Column 'species' not found in metadata.")

    return df_sorted["species"].values


def prepare_training_data(
    img_features: np.ndarray,
    tab_features: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
) -> tuple:
    """
    Prepares hyper-densified training data by flattening centroids.
    Expands dataset size by factor of 9.

    Returns:
        X_img: (N*9, D_img)
        X_tab: (N*9, D_tab)
        y: (N*9,)
        ids: (N*9,)
    """
    # 1. Compute Centroids -> (N, 9, D_img)
    centroids = compute_orthogonal_centroids(img_features)
    N, C, D_img = centroids.shape

    # 2. Flatten Image Features -> (N*9, D_img)
    X_img = centroids.reshape(N * C, D_img)

    # 3. Expand Tabular Features -> (N*9, D_tab)
    X_tab = np.repeat(tab_features, C, axis=0)

    # 4. Expand Labels and IDs
    y = np.repeat(labels, C, axis=0)
    ids_expanded = np.repeat(ids, C, axis=0)

    return X_img, X_tab, y, ids_expanded


def prepare_inference_data(
    img_features: np.ndarray, tab_features: np.ndarray, ids: np.ndarray
) -> tuple:
    """
    Prepares structured data for inference.
    Retains (N, 9, ...) structure for Test-Time Aggregation.

    Returns:
        X_img: (N, 9, D_img)
        X_tab: (N, 9, D_tab)
        ids: (N,)
    """
    # 1. Compute Centroids -> (N, 9, D_img)
    X_img = compute_orthogonal_centroids(img_features)
    N, C, _ = X_img.shape

    # 2. Expand Tabular Features -> (N, 9, D_tab)
    # Replicate tabular features for each centroid
    X_tab = np.tile(tab_features[:, np.newaxis, :], (1, C, 1))

    return X_img, X_tab, ids


def get_densified_train_data(load_cached_data: bool = True):
    """
    Orchestrates the creation of the densified training set with caching.
    """
    cache_img = os.path.join(Config.WORKING_DIR, "train_densified_img.npy")
    cache_tab = os.path.join(Config.WORKING_DIR, "train_densified_tab.npy")
    cache_y = os.path.join(Config.WORKING_DIR, "train_densified_y.npy")
    cache_ids = os.path.join(Config.WORKING_DIR, "train_densified_ids.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(cache_img)
            and os.path.exists(cache_tab)
            and os.path.exists(cache_y)
        ):
            print("Loading densified training data from cache...")
            return (
                load_numpy(cache_img),
                load_numpy(cache_tab),
                load_numpy(cache_y),
                load_numpy(cache_ids),
            )

    print("Generating densified training data...")
    # 1. Get Raw Features (Cached internally by feature_extraction)
    raw_img, raw_ids = extract_dataset_features(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMG_FEATURES,
        Config.CACHE_TRAIN_IDS,
        load_cached_data,
    )

    # 2. Get Tabular & Labels
    raw_tab = load_tabular_data(Config.TRAIN_METADATA_PATH, raw_ids)
    raw_y = load_labels(Config.TRAIN_METADATA_PATH, raw_ids)

    # 3. Densify
    X_img, X_tab, y, ids = prepare_training_data(raw_img, raw_tab, raw_y, raw_ids)

    # 4. Save
    save_numpy(X_img, cache_img)
    save_numpy(X_tab, cache_tab)
    save_numpy(y, cache_y)
    save_numpy(ids, cache_ids)

    return X_img, X_tab, y, ids


def get_densified_val_data(load_cached_data: bool = True):
    """
    Orchestrates the creation of the structured validation set with caching.
    Returns (X_img, X_tab, ids, y) where X is (N, 9, D) and y is (N,).
    """
    cache_img = os.path.join(Config.WORKING_DIR, "val_densified_img.npy")
    cache_tab = os.path.join(Config.WORKING_DIR, "val_densified_tab.npy")
    cache_y = os.path.join(Config.WORKING_DIR, "val_densified_y.npy")
    cache_ids = os.path.join(Config.WORKING_DIR, "val_densified_ids.npy")

    # Local cache paths for raw val features (not in Config)
    raw_cache_img = os.path.join(Config.WORKING_DIR, "val_raw_img.npy")
    raw_cache_ids = os.path.join(Config.WORKING_DIR, "val_raw_ids.npy")

    if load_cached_data:
        if (
            os.path.exists(cache_img)
            and os.path.exists(cache_tab)
            and os.path.exists(cache_y)
        ):
            print("Loading densified validation data from cache...")
            return (
                load_numpy(cache_img),
                load_numpy(cache_tab),
                load_numpy(cache_ids),
                load_numpy(cache_y),
            )

    print("Generating densified validation data...")
    raw_img, raw_ids = extract_dataset_features(
        Config.VAL_METADATA_PATH, raw_cache_img, raw_cache_ids, load_cached_data
    )
    raw_tab = load_tabular_data(Config.VAL_METADATA_PATH, raw_ids)
    y = load_labels(Config.VAL_METADATA_PATH, raw_ids)

    X_img, X_tab, ids = prepare_inference_data(raw_img, raw_tab, raw_ids)

    save_numpy(X_img, cache_img)
    save_numpy(X_tab, cache_tab)
    save_numpy(y, cache_y)
    save_numpy(ids, cache_ids)

    return X_img, X_tab, ids, y


def get_densified_test_data(load_cached_data: bool = True):
    """
    Orchestrates the creation of the structured test set with caching.
    Returns (X_img, X_tab, ids).
    """
    cache_img = os.path.join(Config.WORKING_DIR, "test_densified_img.npy")
    cache_tab = os.path.join(Config.WORKING_DIR, "test_densified_tab.npy")
    cache_ids = os.path.join(Config.WORKING_DIR, "test_densified_ids.npy")

    if load_cached_data:
        if os.path.exists(cache_img) and os.path.exists(cache_tab):
            print("Loading densified test data from cache...")
            return (
                load_numpy(cache_img),
                load_numpy(cache_tab),
                load_numpy(cache_ids),
            )

    print("Generating densified test data...")
    raw_img, raw_ids = extract_dataset_features(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_IMG_FEATURES,
        Config.CACHE_TEST_IDS,
        load_cached_data,
    )
    raw_tab = load_tabular_data(Config.TEST_METADATA_PATH, raw_ids)

    X_img, X_tab, ids = prepare_inference_data(raw_img, raw_tab, raw_ids)

    save_numpy(X_img, cache_img)
    save_numpy(X_tab, cache_tab)
    save_numpy(ids, cache_ids)

    return X_img, X_tab, ids
