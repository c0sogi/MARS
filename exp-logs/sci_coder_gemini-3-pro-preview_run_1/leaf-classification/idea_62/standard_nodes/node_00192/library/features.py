import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHE_DIR, GEOMETRIC_FEATURES, FLOAT_PRECISION
from library.utils import (
    get_logger,
    save_parquet,
    load_parquet,
    save_npy,
    load_npy,
    ensure_dir,
)

logger = get_logger("features")


def compute_image_metrics(image_full_path: str) -> dict:
    """
    Computes specific geometric features for a single binary leaf image.

    Args:
        image_full_path (str): The absolute or relative path to the image file.

    Returns:
        dict: A dictionary containing the 6 scalar geometric features.
    """
    # Initialize default values to ensure dictionary structure is consistent
    metrics = {k: 0.0 for k in GEOMETRIC_FEATURES}

    if not os.path.exists(image_full_path):
        return metrics

    # Load image in grayscale
    img = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return metrics

    # Apply polarity correction:
    # Dataset description: "binary black leaves against white backgrounds".
    # We need: White leaves (255) on Black background (0) for standard CV operations.
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 1. Contour-based Shape Descriptors
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return metrics

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    metrics["Area"] = float(area)

    # If area is effectively zero, return defaults to avoid division by zero
    if area <= 1e-9:
        return metrics

    # 3. Solidity: Ratio of Contour Area to Convex Hull Area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        metrics["Solidity"] = float(area / hull_area)
    else:
        metrics["Solidity"] = 0.0

    # 4. Extent: Ratio of Contour Area to Bounding Rectangle Area
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        metrics["Extent"] = float(area / rect_area)
    else:
        metrics["Extent"] = 0.0

    # 5. Aspect Ratio: Width to Height ratio of Bounding Rectangle
    if h > 0:
        metrics["Aspect_Ratio"] = float(w / h)
    else:
        metrics["Aspect_Ratio"] = 0.0

    # 6. Eccentricity: Measures how much the shape deviates from a circle
    # Requires at least 5 points to fit an ellipse
    if len(cnt) >= 5:
        try:
            # fitEllipse returns ((center), (axes), angle)
            # axes are (minorAxisLength, majorAxisLength) or vice versa depending on angle
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)

            # Semi-axes
            a = max(d1, d2) / 2.0
            b = min(d1, d2) / 2.0

            if a > 0:
                # e = sqrt(1 - (b/a)^2)
                eccentricity = np.sqrt(1 - (b / a) ** 2)
                metrics["Eccentricity"] = float(eccentricity)
            else:
                metrics["Eccentricity"] = 0.0
        except Exception:
            # Fallback if ellipse fitting fails numerically
            metrics["Eccentricity"] = 0.0
    else:
        metrics["Eccentricity"] = 0.0

    return metrics


def generate_feature_set(
    metadata_path: str, dataset_name: str, load_cached_data: bool = True
):
    """
    Generates the complete feature matrix X, target array y, and id array.
    Combines existing tabular features from metadata with computed geometric features.

    Args:
        metadata_path (str): Path to the metadata CSV file (train, val, or test).
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): The feature matrix (float64).
            y (np.ndarray or None): The target labels (if available).
            ids (np.ndarray): The image IDs.
    """
    # Define cache file paths
    cache_X_path = os.path.join(CACHE_DIR, f"X_{dataset_name}.parquet")
    cache_y_path = os.path.join(CACHE_DIR, f"y_{dataset_name}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"ids_{dataset_name}.npy")

    # 1. Attempt to Load from Cache
    if load_cached_data:
        # We check for X and ids. y is optional (might not exist for test set)
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            logger.info(
                f"Loading cached features for {dataset_name} from {CACHE_DIR}..."
            )
            X = load_parquet(cache_X_path)
            ids = load_npy(cache_ids_path)

            y = None
            if os.path.exists(cache_y_path):
                y = load_npy(cache_y_path)

            return X, y, ids
        else:
            logger.info(
                f"Cache miss for {dataset_name}. Starting feature generation..."
            )
    else:
        logger.info(f"Forcing feature regeneration for {dataset_name}...")

    # 2. Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata for {dataset_name}: {df_meta.shape}")

    # 3. Extract IDs and Targets
    ids = df_meta["id"].values
    y = None
    if "species" in df_meta.columns:
        y = df_meta["species"].values

    # 4. Extract Tabular Features
    # Select columns starting with margin, shape, or texture
    tabular_cols = [
        c
        for c in df_meta.columns
        if any(c.startswith(p) for p in ["margin", "shape", "texture"])
    ]
    # Sort to ensure deterministic column order
    tabular_cols.sort()
    X_tabular = df_meta[tabular_cols].copy()

    # 5. Compute Geometric Features
    logger.info(f"Computing geometric features for {len(df_meta)} images...")
    geo_features_list = []

    for _, row in df_meta.iterrows():
        # Construct full image path
        # Metadata contains relative path (e.g., "images/123.jpg")
        # INPUT_DIR is "./input"
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Compute metrics
        feats = compute_image_metrics(full_path)
        geo_features_list.append(feats)

    X_geo = pd.DataFrame(geo_features_list)

    # 6. Combine Features
    # Concatenate tabular and geometric features
    X_combined = pd.concat(
        [X_tabular.reset_index(drop=True), X_geo.reset_index(drop=True)], axis=1
    )

    # Enforce float64 precision
    X_combined = X_combined.astype(np.float64)

    # 7. Save to Cache
    logger.info(f"Saving generated features for {dataset_name} to cache...")
    save_parquet(X_combined, cache_X_path)
    save_npy(ids, cache_ids_path)
    if y is not None:
        save_npy(y, cache_y_path)

    return X_combined, y, ids
