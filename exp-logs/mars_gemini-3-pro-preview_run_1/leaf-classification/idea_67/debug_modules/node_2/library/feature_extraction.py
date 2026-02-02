import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_DIR,
    IMAGES_DIR,
    TABULAR_FEATURE_PREFIXES,
    GEOMETRIC_FEATURES,
    FLOAT_PRECISION,
    get_full_image_path,
)
from library.utils import set_seed


def extract_geometric_features_from_image(image_path):
    """
    Extracts the 6 geometric features from a binary leaf image.
    Features: Area, Major_Axis_Length, Eccentricity, Solidity, Extent, Aspect_Ratio.

    Args:
        image_path (str): Absolute path to the image file.

    Returns:
        np.ndarray: A 1D array of shape (6,) containing the features in float64.
    """
    # Initialize default values (zeros) in case of processing failure
    default_features = np.zeros(len(GEOMETRIC_FEATURES), dtype=FLOAT_PRECISION)

    if not os.path.exists(image_path):
        return default_features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_features

    # Polarity Correction:
    # The dataset description states "binary black leaves against white backgrounds".
    # We apply THRESH_BINARY_INV to make the leaf white (foreground, 255) and background black (0).
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    # Use CHAIN_APPROX_NONE to keep all boundary points for maximum fidelity
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return default_features

    # Implicit Denoising: Select the largest contour by Area to ignore artifacts
    c = max(contours, key=cv2.contourArea)

    # 1. Absolute Mass: Area
    area = cv2.contourArea(c)
    if area == 0:
        return default_features

    # Pre-calculations for derived features
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    x, y, w, h = cv2.boundingRect(c)
    rect_area = w * h

    # Ellipse Fitting for Major Axis and Eccentricity
    # cv2.fitEllipse requires at least 5 points
    major_axis_length = 0.0
    eccentricity = 0.0

    if len(c) >= 5:
        try:
            # fitEllipse returns ((center_x, center_y), (width, height), angle)
            # width and height correspond to the axes lengths (diameters)
            (center, (axis1, axis2), angle) = cv2.fitEllipse(c)

            ma = max(axis1, axis2)
            mi = min(axis1, axis2)

            major_axis_length = ma

            if ma > 0:
                # Eccentricity = sqrt(1 - (minor_axis / major_axis)^2)
                eccentricity = np.sqrt(1 - (mi / ma) ** 2)
        except Exception:
            # Fallback if ellipse fitting fails numerically
            major_axis_length = float(max(w, h))
            eccentricity = 0.0
    else:
        # Fallback for very small contours
        major_axis_length = float(max(w, h))

    # 4. Roughness: Solidity (Area / Convex Hull Area)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 5. Rectangularity: Extent (Area / Bounding Rect Area)
    extent = area / rect_area if rect_area > 0 else 0.0

    # 6. Orientation: Aspect Ratio (Bounding Width / Bounding Height)
    aspect_ratio = w / h if h > 0 else 0.0

    # Assemble vector in strict order defined in config
    # ["area", "major_axis_length", "eccentricity", "solidity", "extent", "aspect_ratio"]
    features = np.array(
        [area, major_axis_length, eccentricity, solidity, extent, aspect_ratio],
        dtype=FLOAT_PRECISION,
    )

    return features


def load_data(split="train", load_cached_data=True):
    """
    Loads data for a specific split (train, val, or test).
    Handles caching of the processed feature matrix and labels.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from ./working/idea_67/ cache.

    Returns:
        tuple: (X, y, ids)
            X (pd.DataFrame): Combined tabular and geometric features.
            y (np.ndarray or None): Target labels (None for 'test' split).
            ids (np.ndarray): Image IDs.
    """

    # Define cache file paths
    cache_X_path = os.path.join(CACHE_DIR, f"X_{split}.parquet")
    cache_y_path = os.path.join(CACHE_DIR, f"y_{split}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"ids_{split}.npy")

    # Attempt to load from cache
    if load_cached_data:
        # Check if X and ids exist
        if os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
            # For train/val, y must also exist
            if split in ["train", "val"] and not os.path.exists(cache_y_path):
                pass  # Cache incomplete, proceed to compute
            else:
                print(f"Loading {split} data from cache: {cache_X_path}")
                X = pd.read_parquet(cache_X_path)
                ids = np.load(cache_ids_path)
                y = (
                    np.load(cache_y_path, allow_pickle=True)
                    if split in ["train", "val"]
                    else None
                )
                return X, y, ids

    print(f"Processing {split} data from scratch...")

    # Determine metadata source
    if split == "train":
        meta_path = TRAIN_META_PATH
    elif split == "val":
        meta_path = VAL_META_PATH
    elif split == "test":
        meta_path = TEST_META_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    df_meta = pd.read_csv(meta_path)

    # Extract Identifiers and Targets
    ids = df_meta["id"].values
    y = df_meta["species"].values if "species" in df_meta.columns else None

    # 1. Select Tabular Features
    # Filter columns that start with defined prefixes (margin, shape, texture)
    tabular_cols = [
        c
        for c in df_meta.columns
        if any(c.startswith(p) for p in TABULAR_FEATURE_PREFIXES)
    ]
    # Sort alphanumerically to ensure deterministic column order
    tabular_cols.sort()
    X_tabular = df_meta[tabular_cols].copy().astype(FLOAT_PRECISION)

    # 2. Extract Geometric Features from Images
    geo_features_list = []

    # Process each image
    for image_id in ids:
        full_path = get_full_image_path(image_id)
        feats = extract_geometric_features_from_image(full_path)
        geo_features_list.append(feats)

    # Create DataFrame for geometric features
    X_geometric = pd.DataFrame(
        np.stack(geo_features_list), columns=GEOMETRIC_FEATURES, index=X_tabular.index
    )

    # 3. Combine Features (Axis-Augmented Fusion)
    X = pd.concat([X_tabular, X_geometric], axis=1)

    # Save to cache
    print(f"Saving {split} data to cache...")
    X.to_parquet(cache_X_path)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    return X, y, ids
