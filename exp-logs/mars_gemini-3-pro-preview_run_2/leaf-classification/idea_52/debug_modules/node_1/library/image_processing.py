import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    IMAGES_DIR,
    CACHE_DIR,
    POLARITY_CHECK_THRESHOLD,
    FLOAT_PRECISION,
)


def get_morphometric_features(image_rel_path):
    """
    Reads an image, corrects polarity, and extracts shape descriptors.

    Features extracted (11 total):
    - Hu Moments (7)
    - Aspect Ratio
    - Solidity
    - Extent
    - Eccentricity

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/10.jpg').

    Returns:
        np.ndarray: A 1D array of shape (11,) containing features in float64.
    """
    full_path = os.path.join(os.path.dirname(IMAGES_DIR), image_rel_path)

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        # Return zeros if image cannot be read
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # 1. Polarity Correction
    # Check corner pixels to determine background color
    # We sample 5x5 corners
    h, w = img.shape
    corners = [
        img[0:5, 0:5],
        img[0:5, w - 5 : w],
        img[h - 5 : h, 0:5],
        img[h - 5 : h, w - 5 : w],
    ]

    # Calculate mean intensity of corners (0-255 scale)
    corner_mean = np.mean([np.mean(c) for c in corners])

    # Normalize to 0-1 for threshold comparison
    corner_mean_norm = corner_mean / 255.0

    # If background is white (high intensity), invert so leaf is white (foreground)
    if corner_mean_norm > POLARITY_CHECK_THRESHOLD:
        img = cv2.bitwise_not(img)

    # 2. Contour Extraction
    # Find contours on the binary image
    # Use a binary threshold to ensure clean mask (though input is likely binary)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 3. Feature Extraction

    # A. Hu Moments (7 invariants)
    moments = cv2.moments(cnt)
    # Add small epsilon to avoid log(0) if we were doing log transform,
    # but here we return raw Hu moments as per plan.
    hu_moments = cv2.HuMoments(moments).flatten()

    # B. Geometric Scalars

    # Aspect Ratio & Extent
    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0
    rect_area = w_rect * h_rect
    contour_area = moments["m00"]  # Same as cv2.contourArea(cnt)
    extent = contour_area / rect_area if rect_area > 0 else 0.0

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    # Needs at least 5 points to fit ellipse
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2)
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except:
            eccentricity = 0.0

    # Combine features
    geometric_features = np.array(
        [aspect_ratio, solidity, extent, eccentricity], dtype=FLOAT_PRECISION
    )

    features = np.concatenate([hu_moments, geometric_features])

    return features.astype(FLOAT_PRECISION)


def extract_morphometrics(metadata_df, dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a given dataset, with caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path' column.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, 11).
    """
    cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_morphometrics.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}")
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    print(
        f"Extracting morphometric features for {dataset_name} ({len(metadata_df)} images)..."
    )

    features_list = []

    # Iterate through dataframe
    # Note: image_path in metadata is like 'images/12.jpg'
    for _, row in metadata_df.iterrows():
        feat = get_morphometric_features(row["image_path"])
        features_list.append(feat)

    features_matrix = np.vstack(features_list).astype(FLOAT_PRECISION)

    # 3. Save Cache
    try:
        np.save(cache_path, features_matrix)
        print(f"Saved morphometrics to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return features_matrix
