import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def extract_morphometric_features(image_path):
    """
    Extracts invariant geometric descriptors from a binary leaf image.

    Features extracted:
    1. Hu Moments (7 features): Invariant to scale, rotation, and translation.
    2. Geometric Scalars (4 features): Aspect Ratio, Solidity, Extent, Eccentricity.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        np.ndarray: A 1D array of shape (11,) containing float64 features.
                    Returns zeros if image load fails or no contour found.
    """
    # Initialize result array (7 Hu moments + 4 scalars)
    # Order: Hu[0]..Hu[6], AspectRatio, Solidity, Extent, Eccentricity
    result = np.zeros(11, dtype=np.float64)

    if not os.path.exists(image_path):
        return result

    # Load image in grayscale (binary images are effectively grayscale)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return result

    # Ensure binary (0 or 255) just in case, though dataset is binary
    # If image has multiple channels, convert to gray
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find contours
    # RETR_EXTERNAL: We only care about the outer boundary of the leaf
    # CHAIN_APPROX_SIMPLE: Compress horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return result

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Store Hu Moments (indices 0-6)
    result[0:7] = hu_moments

    # 2. Geometric Scalars
    area = moments["m00"]

    # Avoid division by zero for very small contours
    if area == 0:
        return result

    # Bounding Rectangle
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect Ratio
    if h > 0:
        aspect_ratio = float(w) / h
    else:
        aspect_ratio = 0.0

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = float(area) / hull_area
    else:
        solidity = 0.0

    # Extent
    rect_area = w * h
    if rect_area > 0:
        extent = float(area) / rect_area
    else:
        extent = 0.0

    # Eccentricity
    # Needs at least 5 points to fit ellipse
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are diameters (major/minor axes)
            # Sort so a is major (larger), b is minor (smaller)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                # e = sqrt(1 - (b^2 / a^2))
                eccentricity = np.sqrt(1 - (minor_axis**2) / (major_axis**2))
        except:
            # Fallback if ellipse fitting fails numerically
            eccentricity = 0.0

    # Store Scalars (indices 7-10)
    result[7] = aspect_ratio
    result[8] = solidity
    result[9] = extent
    result[10] = eccentricity

    return result


def process_images(image_paths, load_cached_data=True, cache_path=None):
    """
    Batch processes images to extract morphometric features with caching.

    Args:
        image_paths (list): List of relative image paths (e.g., 'images/1.jpg').
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_path (str): Path to the parquet file for caching.

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Determine cache path if not provided (though usually provided by caller via Config)
    if cache_path is None:
        # Fallback, though Config paths should be used
        cache_path = os.path.join(Config.WORKING_DIR, "morphometric_features.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometric features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            # Verify length matches
            if len(df) == len(image_paths):
                return df
            else:
                print("Cache length mismatch. Recomputing features...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing features...")

    # 2. Compute from scratch
    print("Extracting morphometric features from images...")

    features_list = []

    # Define column names
    cols = Config.HU_MOMENTS_COLS + Config.GEOMETRIC_SCALARS_COLS

    for rel_path in image_paths:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        feat_vector = extract_morphometric_features(full_path)
        features_list.append(feat_vector)

    # Create DataFrame
    df = pd.DataFrame(features_list, columns=cols)

    # Enforce Double Precision as per Idea
    df = df.astype(np.float64)

    # 3. Save to cache
    if cache_path:
        print(f"Saving morphometric features to {cache_path}...")
        df.to_parquet(cache_path, index=False)

    return df
