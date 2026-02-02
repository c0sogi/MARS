import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def extract_single_image_features(image_path):
    """
    Extracts deterministic morphometric features from a single binary leaf image.

    Features extracted (11 total):
    - Hu Moments (7 features): Invariant to translation, scale, and rotation.
    - Aspect Ratio (1 feature): Bounding rect width / height.
    - Solidity (1 feature): Contour area / Convex hull area.
    - Extent (1 feature): Contour area / Bounding rect area.
    - Eccentricity (1 feature): Calculated from fitted ellipse.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        np.ndarray: A 1D array of shape (11,) containing float64 features.
                    Returns zeros if processing fails.
    """
    # Initialize feature vector (7 Hu + 4 Scalars = 11)
    features = np.zeros(11, dtype=Config.FLOAT_PRECISION)

    if not os.path.exists(image_path):
        return features

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert image: The dataset has black leaves on white background.
    # We want white object on black background for contour finding.
    # Threshold to ensure binary (0 or 255)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Compute Moments
    M = cv2.moments(cnt)
    area = M["m00"]

    # Avoid division by zero for very small noise contours
    if area == 0:
        return features

    # 1. Hu Moments (7 features)
    hu = cv2.HuMoments(M).flatten()
    features[0:7] = hu

    # 2. Geometric Scalars
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect Ratio
    if h > 0:
        features[7] = float(w) / h

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features[8] = area / hull_area

    # Extent
    rect_area = w * h
    if rect_area > 0:
        features[9] = area / rect_area

    # Eccentricity
    # Requires at least 5 points to fit ellipse
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2)
                features[10] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if ellipse fitting fails (e.g. collinear points)
            features[10] = 0.0

    return features


def get_morphometric_features(df, dataset_name, load_cached_data=True):
    """
    Generates or loads the morphometric feature matrix for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing the 'image_path' column.
                           Paths should be relative to Config.INPUT_DIR.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, 11).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{dataset_name}_morphometrics.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometric features from {cache_path}...")
        try:
            features = np.load(cache_path)
            if features.shape[0] == len(df):
                return features.astype(Config.FLOAT_PRECISION)
            else:
                print(
                    f"Cache shape mismatch ({features.shape[0]} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Computing morphometric features for {dataset_name} ({len(df)} images)...")

    feature_list = []
    # Construct full paths
    # The metadata 'image_path' is relative, e.g., 'images/1.jpg'
    # Config.INPUT_DIR is './input'
    full_paths = [
        os.path.join(Config.INPUT_DIR, rel_path)
        for rel_path in df[Config.IMAGE_PATH_COL]
    ]

    for path in full_paths:
        feats = extract_single_image_features(path)
        feature_list.append(feats)

    features = np.array(feature_list, dtype=Config.FLOAT_PRECISION)

    # 3. Save Cache
    print(f"Saving features to {cache_path}...")
    np.save(cache_path, features)

    return features
