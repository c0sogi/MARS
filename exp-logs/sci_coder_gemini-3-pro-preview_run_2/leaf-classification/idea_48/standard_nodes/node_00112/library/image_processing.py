import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR, FLOAT_PRECISION


def process_image(image_path: str) -> np.ndarray:
    """
    Loads an image, corrects its polarity, and extracts Hu Moments and Geometric Scalars.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        np.ndarray: A 1D array of shape (11,) containing:
                    [Hu1...Hu7, Solidity, Extent, Eccentricity, AspectRatio].
                    Returns a zero vector if the image cannot be processed.
    """
    # 1. Load Image
    if not os.path.exists(image_path):
        return np.zeros(11, dtype=FLOAT_PRECISION)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # 2. Polarity Correction
    # Check corners to determine if background is white
    h, w = img.shape
    corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
    avg_corner = np.mean(corners)

    # If background is white (high intensity), invert so leaf is foreground (white)
    if avg_corner > 127:
        img = cv2.bitwise_not(img)

    # 3. Hu Moments (7 features)
    # Calculate moments on the grayscale image (now foreground is bright)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 4. Geometric Scalars (4 features)
    # Threshold to strict binary for contour finding
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    solidity = 0.0
    extent = 0.0
    eccentricity = 0.0
    aspect_ratio = 0.0

    if contours:
        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 0:
            # Solidity: Area / Hull Area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0.0

            # Extent: Area / Bounding Rect Area
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            rect_area = w_rect * h_rect
            if rect_area > 0:
                extent = area / rect_area
            else:
                extent = 0.0

            # Eccentricity and Aspect Ratio: Fit Ellipse
            # fitEllipse requires at least 5 points
            if len(cnt) >= 5:
                try:
                    (xc, yc), (d1, d2), angle = cv2.fitEllipse(cnt)
                    major_axis = max(d1, d2)
                    minor_axis = min(d1, d2)

                    if minor_axis > 0:
                        aspect_ratio = major_axis / minor_axis
                        # Eccentricity: sqrt(1 - (b/a)^2)
                        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    else:
                        aspect_ratio = 0.0
                        eccentricity = 0.0
                except Exception:
                    # Fallback to bounding rect if ellipse fitting fails
                    if w_rect > 0 and h_rect > 0:
                        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                    eccentricity = 0.0
            else:
                # Fallback for small contours
                if w_rect > 0 and h_rect > 0:
                    aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                eccentricity = 0.0

    # Combine features
    geometric_scalars = np.array(
        [solidity, extent, eccentricity, aspect_ratio], dtype=FLOAT_PRECISION
    )
    features = np.concatenate([hu_moments, geometric_scalars])

    return features.astype(FLOAT_PRECISION)


def extract_morphometric_features(
    df: pd.DataFrame, dataset_name: str, load_cached_data: bool = True
) -> np.ndarray:
    """
    Extracts morphometric features for a given dataset, utilizing caching.

    Args:
        df (pd.DataFrame): DataFrame containing an 'image_path' column.
                           Paths should be relative to INPUT_DIR (e.g., 'images/1.jpg').
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching key.
        load_cached_data (bool): If True, attempts to load from disk before computing.

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, 11).
    """
    cache_path = os.path.join(WORKING_DIR, f"morphometrics_{dataset_name}.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached morphometric features from {cache_path}...")
            features = np.load(cache_path)
            if features.shape[0] == len(df):
                return features
            else:
                print("Cached data shape mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Extracting morphometric features for {dataset_name} ({len(df)} images)...")

    feature_list = []

    # Iterate through DataFrame
    # Using simple loop to avoid tqdm dependency if not desired, or for silent execution
    for _, row in df.iterrows():
        # Construct full path: ./input/images/id.jpg
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        feats = process_image(full_path)
        feature_list.append(feats)

    features = np.vstack(feature_list).astype(FLOAT_PRECISION)

    # 3. Save Cache
    try:
        np.save(cache_path, features)
        print(f"Saved morphometric features to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return features
