import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    FLOAT_PRECISION,
    IMG_THRESHOLD_CORNER_MEAN,
)


def extract_morphometrics(image_full_path):
    """
    Performs polarity correction (inverting images if background is white),
    calculates 7 Hu Moments, and computes 4 geometric scalars (Aspect Ratio,
    Solidity, Extent, Eccentricity) for a single binary leaf image.

    Args:
        image_full_path (str): The absolute path to the image file.

    Returns:
        np.ndarray: A 1D array of 11 float64 features.
                    [hu_1...hu_7, aspect_ratio, solidity, extent, eccentricity]
    """
    # Initialize default feature vector (11 zeros)
    # 7 Hu moments + 4 Geometric scalars
    n_features = 11
    default_features = np.zeros(n_features, dtype=FLOAT_PRECISION)

    if not os.path.exists(image_full_path):
        return default_features

    # Read image as grayscale (0-255)
    img = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_features

    # =========================================================================
    # 1. Polarity Correction
    # =========================================================================
    # Check corners to determine background color.
    # If the background is white (high intensity), the leaf is black.
    # We need the leaf to be white (foreground) for contour detection.
    h, w = img.shape
    corner_size = 10
    corners = []

    # Extract corners safely handling small images
    if h > corner_size and w > corner_size:
        corners.append(img[0:corner_size, 0:corner_size])  # Top-Left
        corners.append(img[0:corner_size, w - corner_size : w])  # Top-Right
        corners.append(img[h - corner_size : h, 0:corner_size])  # Bottom-Left
        corners.append(img[h - corner_size : h, w - corner_size : w])  # Bottom-Right
    else:
        corners.append(img)

    # Calculate mean intensity of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # Threshold logic: 0.5 normalized -> 127.5 in 8-bit
    threshold_val = IMG_THRESHOLD_CORNER_MEAN * 255.0

    if corner_mean > threshold_val:
        # Background is white; invert image so leaf becomes white (foreground)
        img = cv2.bitwise_not(img)

    # =========================================================================
    # 2. Contour Extraction
    # =========================================================================
    # Use RETR_EXTERNAL to get the outer boundary of the leaf
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return default_features

    # Assume the largest contour by area is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Calculate moments
    moments = cv2.moments(cnt)
    area = moments["m00"]

    if area == 0:
        return default_features

    # =========================================================================
    # 3. Feature Calculation
    # =========================================================================

    # A. Hu Moments (7 invariants)
    # cv2.HuMoments returns a 7x1 array, we flatten to 1D
    hu_moments = cv2.HuMoments(moments).flatten()

    # B. Geometric Scalars

    # Aspect Ratio (Width / Height of bounding rect)
    x, y, rect_w, rect_h = cv2.boundingRect(cnt)
    aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0.0

    # Solidity (Contour Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Extent (Contour Area / Bounding Rect Area)
    rect_area = rect_w * rect_h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Eccentricity (sqrt(1 - (minor_axis/major_axis)^2))
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            # returns (center, (MA, ma), angle)
            # Note: fitEllipse returns diameters (axes lengths)
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                eccentricity = 0.0
        except Exception:
            # Fallback if ellipse fitting fails numerically
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    # Combine all features
    geometric_features = np.array(
        [aspect_ratio, solidity, extent, eccentricity], dtype=FLOAT_PRECISION
    )
    features = np.concatenate([hu_moments, geometric_features])

    return features.astype(FLOAT_PRECISION)


def load_image_features(df, dataset_name, load_cached_data=True):
    """
    Applies morphometric extraction to the dataset dataframe with caching.

    Args:
        df (pd.DataFrame): DataFrame containing 'image_path' column.
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test') used for cache naming.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_filename = f"morphometrics_{dataset_name}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            # Simple validation: check if length matches input df
            if len(cached_df) == len(df):
                return cached_df
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute features from scratch
    feature_list = []

    # Define column names
    # Hu moments 1-7
    hu_cols = [f"hu_{i}" for i in range(1, 8)]
    geo_cols = ["aspect_ratio", "solidity", "extent", "eccentricity"]
    feature_cols = hu_cols + geo_cols

    for idx, row in df.iterrows():
        # Construct full path from relative path in metadata
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Extract features
        feats = extract_morphometrics(full_path)
        feature_list.append(feats)

    # Create DataFrame
    features_df = pd.DataFrame(feature_list, columns=feature_cols)

    # Enforce double precision
    features_df = features_df.astype(FLOAT_PRECISION)

    # 3. Save to cache
    features_df.to_parquet(cache_path, index=False)

    return features_df
