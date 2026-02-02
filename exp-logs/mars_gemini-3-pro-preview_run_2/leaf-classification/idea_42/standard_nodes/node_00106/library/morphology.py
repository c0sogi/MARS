import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def correct_polarity(image):
    """
    Checks the corners of the image to determine if the background is white.
    If so, inverts the image so the leaf (foreground) is white and background is black.
    """
    if image is None:
        return None

    h, w = image.shape

    # Define corner regions (5x5 pixels)
    # Handle very small images gracefully
    margin = 5
    if h <= margin * 2 or w <= margin * 2:
        margin = 0  # Just check the whole image or single pixels if tiny

    corners = []
    if margin > 0:
        corners.append(image[0:margin, 0:margin])  # Top-left
        corners.append(image[0:margin, w - margin : w])  # Top-right
        corners.append(image[h - margin : h, 0:margin])  # Bottom-left
        corners.append(image[h - margin : h, w - margin : w])  # Bottom-right
    else:
        corners.append(image)

    # Calculate mean intensity of the corners
    corner_means = [np.mean(c) for c in corners]
    avg_corner_intensity = np.mean(corner_means)

    # Threshold scaling (Config threshold is 0-1 float, image is 0-255 uint8)
    threshold_val = Config.POLARITY_CHECK_THRESHOLD * 255.0

    if avg_corner_intensity > threshold_val:
        # Background is white, invert it
        return cv2.bitwise_not(image)

    return image


def extract_hu_moments(image):
    """
    Extracts the 7 Hu Moments from the binary image.
    """
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments


def extract_geometric_props(image):
    """
    Extracts geometric scalar properties: Aspect Ratio, Solidity, Extent, Eccentricity.
    Returns a numpy array of shape (4,).
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(4, dtype=Config.FLOAT_PRECISION)

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return np.zeros(4, dtype=Config.FLOAT_PRECISION)

    # 1. Aspect Ratio and 3. Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = area / rect_area if rect_area > 0 else 0.0

    # 2. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 4. Eccentricity
    # Needs at least 5 points to fit an ellipse
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are diameters. Sort to get minor and major.
            axes = sorted([axis1, axis2])
            minor_axis, major_axis = axes[0], axes[1]

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2)
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if fitEllipse fails
            eccentricity = 0.0

    return np.array(
        [aspect_ratio, solidity, extent, eccentricity], dtype=Config.FLOAT_PRECISION
    )


def process_single_image(rel_path):
    """
    Loads an image, corrects polarity, and extracts all morphometric features.
    Returns a concatenated vector of shape (11,).
    """
    full_path = os.path.join(Config.INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        return np.zeros(11, dtype=Config.FLOAT_PRECISION)

    # Read as Grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return np.zeros(11, dtype=Config.FLOAT_PRECISION)

    # Correct Polarity
    img_corrected = correct_polarity(img)

    # Extract Features
    hu = extract_hu_moments(img_corrected)
    geo = extract_geometric_props(img_corrected)

    return np.concatenate([hu, geo])


def get_morphometric_features(metadata_path, dataset_key, load_cached_data=True):
    """
    Main function to get morphometric features for a dataset.
    Handles caching mechanism.

    Args:
        metadata_path (str): Path to the metadata csv (train/val/test).
        dataset_key (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, 11).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_filename = f"morphometric_features_{dataset_key}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(
            f"Loading cached morphometric features for '{dataset_key}' from {cache_path}..."
        )
        try:
            features = np.load(cache_path)
            return features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Computing morphometric features for '{dataset_key}'...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Debugging subsample
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    features_list = []

    # Iterate and process
    for idx, row in df.iterrows():
        # image_path column contains relative path like "images/123.jpg"
        feat_vector = process_single_image(row["image_path"])
        features_list.append(feat_vector)

    features = np.array(features_list, dtype=Config.FLOAT_PRECISION)

    # 3. Save to Cache
    print(f"Saving features to {cache_path}...")
    np.save(cache_path, features)

    return features
