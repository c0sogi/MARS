import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR


def extract_single_image_features(image_rel_path):
    """
    Extracts Hu Moments and Geometric Scalars from a single image.
    Performs polarity correction to ensure the leaf is foreground (white).
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Initialize default features (11 dimensions: 7 Hu + 4 Geometric)
    default_feats = np.zeros(11, dtype=np.float64)

    if not os.path.exists(full_path):
        return default_feats

    # Read as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return default_feats

    # Polarity Correction
    # Check corners to determine if background is white
    h, w = img.shape
    # Define corners (top-left, top-right, bottom-left, bottom-right)
    # Using a 5x5 window for robustness
    corners = [
        img[0:5, 0:5],
        img[0:5, w - 5 : w],
        img[h - 5 : h, 0:5],
        img[h - 5 : h, w - 5 : w],
    ]

    corner_mean = np.mean([np.mean(c) for c in corners])

    # Assuming 8-bit image (0-255). 0.5 threshold in normalized space is 127.5
    if corner_mean > 127:
        # Invert image: Background becomes black (0), Leaf becomes white (255)
        img = cv2.bitwise_not(img)

    # Binarize to ensure clean shapes (though dataset is binary, this cleans compression artifacts)
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 1. Hu Moments
    moments = cv2.moments(img_bin)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. Geometric Scalars
    # Find contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return default_feats

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return default_feats

    # Aspect Ratio & Extent
    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
    rect_area = w_rect * h_rect
    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0
    extent = area / rect_area if rect_area > 0 else 0.0

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    # fitEllipse requires at least 5 points
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except:
            eccentricity = 0.0

    # Combine features
    # hu_moments is shape (7,)
    geo_features = np.array(
        [aspect_ratio, solidity, extent, eccentricity], dtype=np.float64
    )

    return np.concatenate([hu_moments, geo_features])


def extract_morphometrics(df, dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a given dataframe containing image paths.

    Args:
        df (pd.DataFrame): Dataframe containing 'image_path' column.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Dataframe containing the extracted features.
    """
    cache_path = os.path.join(WORKING_DIR, f"morphometrics_{dataset_name}.parquet")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Extracting morphometrics for {dataset_name}...")

    feature_list = []
    image_paths = df["image_path"].values

    for path in image_paths:
        feats = extract_single_image_features(path)
        feature_list.append(feats)

    feature_matrix = np.array(feature_list)

    # Create DataFrame
    col_names = [f"hu_{i}" for i in range(7)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]
    features_df = pd.DataFrame(feature_matrix, columns=col_names)

    # Add ID column for alignment safety
    if "id" in df.columns:
        features_df["id"] = df["id"].values

    # Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    features_df.to_parquet(cache_path, index=False)
    print(f"Saved morphometrics for {dataset_name} to {cache_path}")

    return features_df
