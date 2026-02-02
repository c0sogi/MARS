import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, META_FEATURES, WORKING_DIR


def extract_single_image_features(image_rel_path):
    """
    Extracts morphological meta-features from a binary leaf image.

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        dict: Dictionary containing 'aspect_ratio', 'solidity', 'extent', 'eccentricity'.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Default values in case of failure
    features = {k: 0.0 for k in META_FEATURES}

    if not os.path.exists(full_path):
        return features

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Invert image: Dataset is black leaf on white background.
    # We need white object on black background for contour detection.
    # 255 - img is equivalent to bitwise_not for 8-bit images
    img_inv = 255 - img

    # Threshold to ensure binary (though dataset is already binary)
    _, thresh = cv2.threshold(img_inv, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # 1. Aspect Ratio & 3. Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    if h > 0:
        features["aspect_ratio"] = float(w) / h

    if rect_area > 0:
        features["extent"] = float(area) / rect_area

    # 2. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        features["solidity"] = float(area) / hull_area

    # 4. Eccentricity
    # Requires at least 5 points to fit an ellipse
    if len(cnt) >= 5:
        try:
            # fitEllipse returns ((center_x, center_y), (width, height), angle)
            # Note: width and height here refer to the lengths of the axes
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)

            # MA and ma are axis lengths. We need semi-major (a) and semi-minor (b)
            # or just use ratio of axes.
            # Eccentricity e = sqrt(1 - (b^2 / a^2))

            a = max(MA, ma)
            b = min(MA, ma)

            if a > 0:
                features["eccentricity"] = np.sqrt(1 - (b / a) ** 2)
        except:
            # Fallback if ellipse fitting fails numerically
            features["eccentricity"] = 0.0

    return features


def get_augmented_dataset(metadata_path, cache_path, load_cached_data=True):
    """
    Loads dataset, extracts meta-features, and merges them. Implements caching.

    Args:
        metadata_path (str): Path to the source CSV (train/val/test).
        cache_path (str): Path where the processed parquet file should be stored.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing original columns plus meta-features.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing features...")

    # 2. Compute from scratch
    print(f"Processing features for {os.path.basename(metadata_path)}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Extract features for each image
    # Using a list comprehension for efficiency
    extracted_data = []
    image_paths = df["image_path"].tolist()

    for path in image_paths:
        feats = extract_single_image_features(path)
        extracted_data.append(feats)

    # Create DataFrame from extracted features
    meta_df = pd.DataFrame(extracted_data)

    # Concatenate with original dataframe
    # We align on index since order is preserved
    df_augmented = pd.concat([df, meta_df], axis=1)

    # 3. Save to cache
    print(f"Saving augmented dataset to {cache_path}...")
    df_augmented.to_parquet(cache_path, index=False)

    return df_augmented
