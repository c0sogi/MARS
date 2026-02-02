import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import ensure_directory, save_cache, load_cache


def check_polarity(img):
    """
    Checks if the image background is white (mean corner intensity > threshold).
    If so, inverts the image so the leaf becomes the foreground (white) and background black.

    Args:
        img (np.ndarray): Grayscale image (0-255).

    Returns:
        np.ndarray: Polarity-corrected image.
    """
    h, w = img.shape
    c_size = 10  # Size of corner patch

    # Handle very small images
    if h < c_size * 2 or w < c_size * 2:
        c_size = min(h, w) // 2

    # Extract corners
    corners = [
        img[0:c_size, 0:c_size],
        img[0:c_size, w - c_size : w],
        img[h - c_size : h, 0:c_size],
        img[h - c_size : h, w - c_size : w],
    ]

    # Calculate mean intensity of corners (normalized 0-1)
    # Flatten list of arrays into one array for mean calculation
    all_corners = np.concatenate([c.flatten() for c in corners])
    mean_intensity = np.mean(all_corners) / 255.0

    if mean_intensity > Config.IMG_POLARITY_THRESHOLD:
        # Invert image
        return cv2.bitwise_not(img)

    return img


def extract_morphometrics(img):
    """
    Extracts Hu Moments and Geometric Scalars from a binary image.

    Args:
        img (np.ndarray): Binary image where object is foreground (white).

    Returns:
        np.ndarray: 11-dimensional feature vector.
    """
    # Initialize feature vector
    # 7 Hu moments + 4 Geometric scalars = 11 features
    features = np.zeros(11, dtype=Config.FLOAT_TYPE)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # --- Hu Moments (7 features) ---
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    features[0:7] = hu_moments

    # --- Geometric Scalars (4 features) ---

    # 1. Aspect Ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # 2. Extent (Object Area / Bounding Rect Area)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # 3. Solidity (Object Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 4. Eccentricity
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            # fitEllipse returns (center, (MA, ma), angle)
            # Note: The order of MA, ma depends on the implementation/angle,
            # so we explicitly sort them to get major and minor axes.
            (center, (d1, d2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            if major_axis > 0:
                # e = sqrt(1 - (b^2 / a^2))
                # where a = major/2, b = minor/2.
                # Ratio (b/a)^2 is same as (minor/major)^2
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if fitEllipse fails numerically
            eccentricity = 0.0

    features[7] = aspect_ratio
    features[8] = extent
    features[9] = solidity
    features[10] = eccentricity

    return features


def process_images(metadata_df, dataset_name, load_cached_data=True):
    """
    Main driver to process a list of images defined in metadata.
    Handles caching, polarity correction, and feature extraction.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Matrix of extracted features (N_samples, 11).
    """
    # Define cache path
    cache_filename = f"morphometrics_{dataset_name}.npy"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try Load Cache
    if load_cached_data:
        cached_data = load_cache(cache_path)
        if cached_data is not None:
            print(f"Loaded cached morphometrics for '{dataset_name}' from {cache_path}")
            return cached_data.astype(Config.FLOAT_TYPE)

    # 2. Process from Scratch
    print(f"Processing images for '{dataset_name}' (Cache miss or force reload)...")

    features_list = []
    missing_count = 0

    for idx, row in metadata_df.iterrows():
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Initialize zero vector
        feats = np.zeros(11, dtype=Config.FLOAT_TYPE)

        if os.path.exists(full_path):
            # Read image as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Correct polarity (ensure leaf is white)
                img_corrected = check_polarity(img)

                # Extract features
                feats = extract_morphometrics(img_corrected)
            else:
                missing_count += 1
        else:
            missing_count += 1

        features_list.append(feats)

    if missing_count > 0:
        print(
            f"Warning: {missing_count} images were missing or unreadable in '{dataset_name}'."
        )

    # Convert to numpy array
    result = np.array(features_list, dtype=Config.FLOAT_TYPE)

    # 3. Save to Cache
    save_cache(result, cache_path)
    print(f"Saved morphometrics for '{dataset_name}' to {cache_path}")

    return result
