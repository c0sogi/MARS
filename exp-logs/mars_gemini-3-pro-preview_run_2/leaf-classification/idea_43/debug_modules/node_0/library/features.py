import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    USE_POLARITY_CORRECTION,
    MORPHOMETRIC_FEATURES,
    FLOAT_PRECISION,
)


def correct_polarity(image):
    """
    Ensures the leaf is the foreground (white/255) and background is black (0).
    Assumes the corners of the image are background.
    """
    # Get image dimensions
    h, w = image.shape[:2]

    # Extract corner regions (3x3 to be robust to noise)
    # Handle small images gracefully
    if h < 6 or w < 6:
        corners = [image]
    else:
        corners = [
            image[0:3, 0:3],
            image[0:3, w - 3 : w],
            image[h - 3 : h, 0:3],
            image[h - 3 : h, w - 3 : w],
        ]

    # Calculate mean intensity of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If background is white (high intensity), invert
    # We assume binary images, so threshold around 127 is safe
    if corner_mean > 127:
        image = cv2.bitwise_not(image)

    return image


def extract_single_image_features(image_path):
    """
    Extracts morphometric features from a single image file.
    """
    # Initialize features with 0.0
    features = {k: 0.0 for k in MORPHOMETRIC_FEATURES}

    # Load image
    if not os.path.exists(image_path):
        return features

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Polarity Correction
    if USE_POLARITY_CORRECTION:
        img = correct_polarity(img)

    # Threshold to ensure strict binary (0, 255)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Basic Moments
    M = cv2.moments(cnt)
    area = M["m00"]

    # Avoid division by zero for very small contours
    if area < 1e-6:
        return features

    features["area"] = area

    # Perimeter
    perimeter = cv2.arcLength(cnt, True)
    features["perimeter"] = perimeter

    # Hu Moments
    # We store raw Hu moments. Downstream PowerTransformer will handle distribution.
    hu = cv2.HuMoments(M).flatten()
    for i in range(7):
        features[f"hu_moment_{i+1}"] = hu[i]

    # Geometric Scalars
    x, y, w, h_rect = cv2.boundingRect(cnt)
    rect_area = w * h_rect
    features["aspect_ratio"] = float(w) / h_rect if h_rect > 0 else 0
    features["extent"] = area / rect_area if rect_area > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    features["solidity"] = area / hull_area if hull_area > 0 else 0

    # Eccentricity
    # Requires at least 5 points to fit an ellipse
    if len(cnt) >= 5:
        try:
            # fitEllipse returns ((x,y), (width, height), angle)
            (x_e, y_e), (d1, d2), angle = cv2.fitEllipse(cnt)
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            if major_axis > 0:
                a = major_axis / 2
                b = minor_axis / 2
                # eccentricity = sqrt(1 - (b/a)^2)
                ratio = (b / a) ** 2
                features["eccentricity"] = np.sqrt(max(0, 1 - ratio))
        except Exception:
            features["eccentricity"] = 0.0
    else:
        features["eccentricity"] = 0.0

    return features


def extract_morphometric_features(metadata_df, dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a given dataset (train/val/test).
    Handles caching to parquet.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'image_path'.
        dataset_name (str): Name of the dataset split (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with 'id' and extracted morphometric features.
    """
    cache_path = os.path.join(WORKING_DIR, f"morphometrics_{dataset_name}.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # Ensure precision
            float_cols = [c for c in df.columns if c != "id"]
            df[float_cols] = df[float_cols].astype(FLOAT_PRECISION)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting morphometrics for {dataset_name}...")

    results = []

    # Iterate through metadata
    # metadata_df contains 'id' and 'image_path' (relative to input dir)
    for idx, row in metadata_df.iterrows():
        image_id = row["id"]
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        feats = extract_single_image_features(full_path)
        feats["id"] = image_id
        results.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(results)

    # Reorder columns to have id first
    cols = ["id"] + MORPHOMETRIC_FEATURES

    # Ensure all columns exist (in case of empty results or missing keys)
    for c in cols:
        if c not in df_features.columns:
            df_features[c] = 0.0

    df_features = df_features[cols]

    # Cast to specified precision
    float_cols = [c for c in df_features.columns if c != "id"]
    df_features[float_cols] = df_features[float_cols].astype(FLOAT_PRECISION)

    # 3. Save to cache
    try:
        df_features.to_parquet(cache_path, index=False)
        print(f"Saved morphometrics to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df_features
