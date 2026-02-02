import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    FLOAT_PRECISION,
    POLARITY_CHECK_THRESHOLD,
    WORKING_DIR,
)


def extract_morphometrics(image_rel_path):
    """
    Extracts physical morphometric features from a binary leaf image.

    Features extracted:
    - 7 Hu Moments (invariant to scale, rotation, translation)
    - Geometric Scalars:
        - Aspect Ratio
        - Solidity
        - Extent
        - Eccentricity
        - Orientation
        - Roundness (4 * pi * Area / Perimeter^2)

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/10.jpg')

    Returns:
        np.array: A 1D float64 array of extracted features.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Initialize feature vector with zeros (13 features total: 7 Hu + 6 Geometric)
    # 7 Hu moments + AspectRatio, Solidity, Extent, Eccentricity, Orientation, Roundness
    num_features = 13
    feature_vector = np.zeros(num_features, dtype=FLOAT_PRECISION)

    if not os.path.exists(full_path):
        return feature_vector

    # Load image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return feature_vector

    # --- Polarity Correction ---
    # Check corners to determine background color
    h, w = img.shape
    corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
    corner_mean = np.mean(corners) / 255.0

    # If background is white (high intensity), invert image so leaf is foreground (white)
    if corner_mean > POLARITY_CHECK_THRESHOLD:
        img = cv2.bitwise_not(img)

    # --- Contour Extraction ---
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return feature_vector

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return feature_vector

    # --- Feature Calculation ---

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Log transform is common for Hu moments to handle scale, but we stick to raw
    # values here as the downstream PowerTransformer will handle distribution.
    feature_vector[0:7] = hu_moments

    # 2. Geometric Scalars

    # Aspect Ratio
    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0
    feature_vector[7] = aspect_ratio

    # Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    feature_vector[8] = solidity

    # Extent (Area / Bounding Rect Area)
    rect_area = w_rect * h_rect
    extent = float(area) / rect_area if rect_area > 0 else 0.0
    feature_vector[9] = extent

    # Eccentricity and Orientation (via fitEllipse)
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
        major_axis = max(axis1, axis2)
        minor_axis = min(axis1, axis2)

        eccentricity = (
            np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0
        )
        feature_vector[10] = eccentricity
        feature_vector[11] = angle  # Orientation
    else:
        feature_vector[10] = 0.0
        feature_vector[11] = 0.0

    # Roundness / Compactness (4 * pi * Area / Perimeter^2)
    perimeter = cv2.arcLength(cnt, True)
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    feature_vector[12] = roundness

    return feature_vector


def generate_physical_dataset(df, cache_path, load_cached_data=True):
    """
    Generates or loads the physical feature dataset for a given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing 'image_path' column.
        cache_path (str): Path to save/load the .npy file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.array: Matrix of shape (N_samples, N_features) with float64 precision.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading physical features from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            # Verify shape matches dataframe length
            if data.shape[0] == len(df):
                return data.astype(FLOAT_PRECISION)
            else:
                print(
                    f"Cache shape mismatch ({data.shape[0]} vs {len(df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print(f"Computing physical features for {len(df)} images...")

    features_list = []
    image_paths = df["image_path"].values

    for path in image_paths:
        feats = extract_morphometrics(path)
        features_list.append(feats)

    data = np.array(features_list, dtype=FLOAT_PRECISION)

    # Save to cache
    print(f"Saving physical features to cache: {cache_path}")
    np.save(cache_path, data)

    return data
