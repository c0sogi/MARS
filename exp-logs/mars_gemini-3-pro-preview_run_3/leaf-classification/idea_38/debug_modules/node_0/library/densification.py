import os
import numpy as np
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger("densification")


def compute_centroids(img_features):
    """
    Computes 3 orthogonal centroids from 12 views.

    Args:
        img_features: (N, 12, D) numpy array

    Returns:
        primary_centroids: (N, 3, D) numpy array
    """
    # Indices corresponding to orthogonal views based on 30 degree steps
    # Set A: 0, 90, 180, 270 -> indices [0, 3, 6, 9]
    # Set B: 30, 120, 210, 300 -> indices [1, 4, 7, 10]
    # Set C: 60, 150, 240, 330 -> indices [2, 5, 8, 11]

    idx_a = [0, 3, 6, 9]
    idx_b = [1, 4, 7, 10]
    idx_c = [2, 5, 8, 11]

    # Compute means along the view dimension (axis 1)
    c_a = np.mean(img_features[:, idx_a, :], axis=1)  # (N, D)
    c_b = np.mean(img_features[:, idx_b, :], axis=1)  # (N, D)
    c_c = np.mean(img_features[:, idx_c, :], axis=1)  # (N, D)

    # Stack to (N, 3, D)
    primary_centroids = np.stack([c_a, c_b, c_c], axis=1)

    return primary_centroids


def interpolate_convex_hull(primary_centroids):
    """
    Generates 3 synthetic centroids via linear interpolation (MixUp).

    Args:
        primary_centroids: (N, 3, D) numpy array [C_A, C_B, C_C]

    Returns:
        synthetic_centroids: (N, 3, D) numpy array
    """
    c_a = primary_centroids[:, 0, :]
    c_b = primary_centroids[:, 1, :]
    c_c = primary_centroids[:, 2, :]

    # Linear interpolation (0.5/0.5)
    c_ab = 0.5 * c_a + 0.5 * c_b
    c_bc = 0.5 * c_b + 0.5 * c_c
    c_ca = 0.5 * c_c + 0.5 * c_a

    synthetic_centroids = np.stack([c_ab, c_bc, c_ca], axis=1)

    return synthetic_centroids


def prepare_training_data(
    img_features, tab_features, ids, labels, cache_suffix="train", load_cached_data=True
):
    """
    Prepares densified training data (6x augmentation: 3 primary + 3 synthetic centroids).

    Args:
        img_features: (N, 12, D)
        tab_features: (N, T)
        ids: (N,)
        labels: (N,)
        cache_suffix: str, suffix for cache files (e.g., 'train')
        load_cached_data: bool

    Returns:
        X_img: (N*6, D)
        X_tab: (N*6, T)
        y: (N*6,)
        ids_expanded: (N*6,)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_x_img = os.path.join(cache_dir, f"densified_{cache_suffix}_img.npy")
    path_x_tab = os.path.join(cache_dir, f"densified_{cache_suffix}_tab.npy")
    path_y = os.path.join(cache_dir, f"densified_{cache_suffix}_y.npy")
    path_ids = os.path.join(cache_dir, f"densified_{cache_suffix}_ids.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(path_x_img)
            and os.path.exists(path_x_tab)
            and os.path.exists(path_y)
            and os.path.exists(path_ids)
        ):
            logger.info(f"Loading cached densified training data for {cache_suffix}...")
            X_img = np.load(path_x_img)
            X_tab = np.load(path_x_tab)
            y = np.load(path_y)
            ids_expanded = np.load(path_ids)
            return X_img, X_tab, y, ids_expanded
        else:
            logger.info(
                f"Cache missing for densified {cache_suffix} data. Processing..."
            )
    else:
        logger.info(f"Ignoring cache. Processing densified {cache_suffix} data...")

    # 1. Compute Primary Centroids (N, 3, D)
    primary = compute_centroids(img_features)

    # 2. Compute Synthetic Centroids (N, 3, D)
    synthetic = interpolate_convex_hull(primary)

    # 3. Concatenate (N, 6, D)
    # Order: A, B, C, AB, BC, CA
    combined_img = np.concatenate([primary, synthetic], axis=1)

    # 4. Flatten Image Features (N*6, D)
    N, K, D = combined_img.shape
    X_img = combined_img.reshape(N * K, D)

    # 5. Replicate Tabular Features, Labels, IDs
    # Repeat each row K times
    X_tab = np.repeat(tab_features, K, axis=0)
    y = np.repeat(labels, K, axis=0)
    ids_expanded = np.repeat(ids, K, axis=0)

    # Save to cache
    logger.info(f"Saving densified training data to {cache_dir}...")
    np.save(path_x_img, X_img)
    np.save(path_x_tab, X_tab)
    np.save(path_y, y)
    np.save(path_ids, ids_expanded)

    return X_img, X_tab, y, ids_expanded


def prepare_inference_data(
    img_features,
    tab_features,
    ids,
    labels=None,
    cache_suffix="val",
    load_cached_data=True,
):
    """
    Prepares inference data (3x augmentation: 3 primary centroids).
    Used for Validation and Test sets.

    Args:
        img_features: (N, 12, D)
        tab_features: (N, T)
        ids: (N,)
        labels: (N,) or None
        cache_suffix: str, suffix for cache files (e.g., 'val', 'test')
        load_cached_data: bool

    Returns:
        X_img: (N*3, D)
        X_tab: (N*3, T)
        ids_expanded: (N*3,)
        y_expanded: (N*3,) or None
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_x_img = os.path.join(cache_dir, f"canonical_{cache_suffix}_img.npy")
    path_x_tab = os.path.join(cache_dir, f"canonical_{cache_suffix}_tab.npy")
    path_ids = os.path.join(cache_dir, f"canonical_{cache_suffix}_ids.npy")
    path_y = os.path.join(cache_dir, f"canonical_{cache_suffix}_y.npy")

    # Check cache
    if load_cached_data:
        files_exist = (
            os.path.exists(path_x_img)
            and os.path.exists(path_x_tab)
            and os.path.exists(path_ids)
        )
        if labels is not None:
            files_exist = files_exist and os.path.exists(path_y)

        if files_exist:
            logger.info(f"Loading cached inference data for {cache_suffix}...")
            X_img = np.load(path_x_img)
            X_tab = np.load(path_x_tab)
            ids_expanded = np.load(path_ids)
            y_expanded = np.load(path_y) if labels is not None else None
            return X_img, X_tab, ids_expanded, y_expanded
        else:
            logger.info(
                f"Cache missing for inference {cache_suffix} data. Processing..."
            )
    else:
        logger.info(f"Ignoring cache. Processing inference {cache_suffix} data...")

    # 1. Compute Primary Centroids (N, 3, D)
    # We do NOT use synthetic centroids for inference.
    primary = compute_centroids(img_features)

    # 2. Flatten Image Features (N*3, D)
    N, K, D = primary.shape
    X_img = primary.reshape(N * K, D)

    # 3. Replicate Tabular Features, IDs
    X_tab = np.repeat(tab_features, K, axis=0)
    ids_expanded = np.repeat(ids, K, axis=0)

    # 4. Replicate Labels if provided
    if labels is not None:
        y_expanded = np.repeat(labels, K, axis=0)
    else:
        y_expanded = None

    # Save to cache
    logger.info(f"Saving inference data to {cache_dir}...")
    np.save(path_x_img, X_img)
    np.save(path_x_tab, X_tab)
    np.save(path_ids, ids_expanded)

    if y_expanded is not None:
        np.save(path_y, y_expanded)

    return X_img, X_tab, ids_expanded, y_expanded
