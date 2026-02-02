import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR, IMAGE_THRESHOLD


def extract_single_image_features(image_path):
    """
    Extracts 5 geometric features from a single image path using the
    Implicit Filtering strategy (Cite Lesson 00197).

    Features: Area, Eccentricity, Solidity, Extent, Aspect_Ratio.
    """
    # Initialize feature vector (5 features)
    features = np.zeros(5, dtype=np.float64)

    if not os.path.exists(image_path):
        return features

    # 1. Load Image in Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # 2. Polarity Correction + Fixed Thresholding
    # Leaf is foreground (white), background is black
    # Threshold 127 cuts through JPEG ringing (Lesson 00195)
    # cv2.THRESH_BINARY_INV ensures leaf is white (255) if input was black-on-white
    _, binary = cv2.threshold(img, IMAGE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # 3. Lossless Contour Extraction (Implicit Filtering - Cite Lesson 00197)
    # We skip explicit denoising (CCA) as it introduces overhead and variance.
    # CHAIN_APPROX_NONE preserves all boundary points for accurate perimeter/shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return features

    # Select largest contour
    cnt = max(contours, key=cv2.contourArea)

    # 4. Compute Parsimonious Geometric Descriptors
    area = cv2.contourArea(cnt)

    # If area is effectively zero, return empty features
    if area <= 0:
        return features

    features[0] = area  # Absolute Scale

    # Eccentricity (Elongation)
    # Requires at least 5 points to fit an ellipse
    if len(cnt) >= 5:
        try:
            # fitEllipse returns (center, size, angle) where size is (width, height)
            (x, y), (d1, d2), angle = cv2.fitEllipse(cnt)
            a = max(d1, d2) / 2.0  # Semi-major axis
            b = min(d1, d2) / 2.0  # Semi-minor axis
            if a > 0:
                # e = sqrt(1 - (b/a)^2)
                features[1] = np.sqrt(1 - (b**2 / a**2))
            else:
                features[1] = 0.0
        except:
            features[1] = 0.0
    else:
        features[1] = 0.0

    # Solidity (Roughness)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features[2] = area / hull_area

    # Extent (Rectangularity) & Aspect Ratio (Orientation)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    if rect_area > 0:
        features[3] = area / rect_area

    if h > 0:
        features[4] = float(w) / h

    return features


def process_images(metadata_df, dataset_name, load_cached_data=True):
    """
    Batch processes images listed in metadata_df to extract geometric features.
    Handles caching to ./working/idea_64/.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'file_path'.
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'test').
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        np.ndarray: (N, 6) array of geometric features.
    """
    cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_geometric.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached geometric features for {dataset_name}...")
            features = np.load(cache_file)
            if features.shape[0] == len(metadata_df):
                return features
            else:
                print(
                    f"Cache size mismatch ({features.shape[0]} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(
        f"Extracting geometric features for {dataset_name} ({len(metadata_df)} images)..."
    )

    feature_list = []
    file_paths = metadata_df["file_path"].values

    for i, rel_path in enumerate(file_paths):
        full_path = os.path.join(INPUT_DIR, rel_path)
        feats = extract_single_image_features(full_path)
        feature_list.append(feats)

    features = np.array(feature_list, dtype=np.float64)

    # 3. Save to Cache
    try:
        np.save(cache_file, features)
        print(f"Saved features to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_file}: {e}")

    return features
