import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_morphometrics(image_full_path):
    """
    Extracts deterministic morphometric features from a binary leaf image.

    Features extracted:
    - Hu Moments (7 descriptors)
    - Aspect Ratio
    - Solidity
    - Extent
    - Eccentricity

    Args:
        image_full_path (str): Absolute or relative path to the image file.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    # Initialize default feature vector
    features = {
        "hu_1": 0.0,
        "hu_2": 0.0,
        "hu_3": 0.0,
        "hu_4": 0.0,
        "hu_5": 0.0,
        "hu_6": 0.0,
        "hu_7": 0.0,
        "aspect_ratio": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
        "eccentricity": 0.0,
    }

    if not os.path.exists(image_full_path):
        return features

    # Read image as grayscale (dataset is binary)
    img = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Ensure binary thresholding
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, hu in enumerate(hu_moments):
        features[f"hu_{i+1}"] = hu

    # 2. Geometric Scalars
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect Ratio
    if h > 0:
        features["aspect_ratio"] = float(w) / h

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["solidity"] = float(area) / hull_area

    # Extent
    rect_area = w * h
    if rect_area > 0:
        features["extent"] = float(area) / rect_area

    # Eccentricity
    # Requires at least 5 points to fit ellipse
    if len(cnt) >= 5:
        try:
            (center, (ma, MA), angle) = cv2.fitEllipse(cnt)
            # ma is minor axis, MA is major axis in OpenCV return convention sometimes varies,
            # but usually second tuple is (width, height).
            # We sort them to ensure a is major, b is minor
            a = max(ma, MA) / 2.0
            b = min(ma, MA) / 2.0
            if a > 0:
                features["eccentricity"] = np.sqrt(1 - (b**2 / a**2))
        except Exception:
            features["eccentricity"] = 0.0

    return features


def get_macro_features(metadata_df, cache_path, load_cached_data=True):
    """
    Orchestrates the extraction of macro features for a dataset.
    Handles caching to parquet files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path' column.
        cache_path (str): Path to save/load the parquet file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features aligned with metadata_df.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached macro features from {cache_path}")
        try:
            df_features = pd.read_parquet(cache_path)
            # Basic validation to ensure length matches
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print("Cached file length mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting macro features for {len(metadata_df)} samples...")

    feature_list = []

    for _, row in metadata_df.iterrows():
        # metadata 'image_path' is relative to input dir (e.g., 'images/12.jpg')
        # We need to construct the full path relative to the script execution location
        # config.INPUT_DIR is './input'
        full_path = os.path.join(config.INPUT_DIR, row["image_path"])

        feats = extract_morphometrics(full_path)
        feature_list.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(feature_list)

    # Enforce float64 precision as per config
    df_features = df_features.astype(config.FLOAT_PRECISION)

    # 3. Save to cache
    print(f"Saving macro features to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
