import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR, FLOAT_PRECISION


def extract_morphometrics(image_rel_path):
    """
    Extracts morphometric features from a binary leaf image.

    Features extracted:
    1. Hu Moments (7 features): Scale, rotation, and translation invariant shape descriptors.
    2. Aspect Ratio (1 feature): Width / Height of the bounding rectangle.
    3. Solidity (1 feature): Contour Area / Convex Hull Area.
    4. Extent (1 feature): Contour Area / Bounding Rectangle Area.

    Total: 10 features.

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/10.jpg').

    Returns:
        np.ndarray: A 1D array of size 10 containing the extracted features.
                    Returns zeros if extraction fails.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Initialize default feature vector (zeros)
    # 7 Hu moments + 3 geometric scalars = 10 features
    feature_vector = np.zeros(10, dtype=FLOAT_PRECISION)

    if not os.path.exists(full_path):
        return feature_vector

    # Load image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return feature_vector

    # The dataset description says "binary black leaves against white backgrounds".
    # cv2.findContours looks for white objects on black background.
    # We apply an inverse threshold to flip the polarity.
    # Using 127 as threshold is safe for binary images.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return feature_vector

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # --- Feature 1: Hu Moments (7 dims) ---
    moments = cv2.moments(cnt)
    # HuMoments returns a 7x1 array, flatten to 1D
    hu_moments = cv2.HuMoments(moments).flatten()
    feature_vector[0:7] = hu_moments

    # --- Geometric Scalars (3 dims) ---
    area = moments["m00"]

    # Avoid division by zero if area is 0 (degenerate contour)
    if area > 0:
        # Aspect Ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        feature_vector[7] = aspect_ratio

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0
        feature_vector[8] = solidity

        # Extent
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0.0
        feature_vector[9] = extent

    return feature_vector.astype(FLOAT_PRECISION)


def process_image_batch(metadata_df, dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a dataframe of images with caching support.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path' column.
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: A (N, 10) array of extracted features.
    """
    # Ensure working directory exists (redundant if config handles it, but safe)
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_morphometrics.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        try:
            features = np.load(cache_path)
            # Verify shape matches current metadata length
            if features.shape[0] == len(metadata_df):
                return features.astype(FLOAT_PRECISION)
            else:
                print(
                    f"Cache shape mismatch ({features.shape[0]} vs {len(metadata_df)}). Recomputing."
                )
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing.")

    # 2. Compute Features
    print(f"Extracting morphometrics for {dataset_name} ({len(metadata_df)} images)...")

    features_list = []
    image_paths = metadata_df["image_path"].values

    for rel_path in image_paths:
        feat = extract_morphometrics(rel_path)
        features_list.append(feat)

    features = np.array(features_list, dtype=FLOAT_PRECISION)

    # 3. Save Cache
    try:
        np.save(cache_path, features)
        print(f"Saved morphometrics to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return features
