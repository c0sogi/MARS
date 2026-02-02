import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Constants
CACHE_DIR = "./working/idea_64"
INPUT_DIR = "./input"


def correct_polarity(img):
    """
    Detects if the background is white and inverts the image if necessary.
    Assumes binary image (0 or 255).

    Args:
        img (np.ndarray): Input binary image (grayscale).

    Returns:
        np.ndarray: Polarity-corrected image where foreground is white (255)
                    and background is black (0).
    """
    h, w = img.shape
    # Sample 5x5 corners to determine background color
    corners = [
        img[0:5, 0:5],
        img[0:5, w - 5 : w],
        img[h - 5 : h, 0:5],
        img[h - 5 : h, w - 5 : w],
    ]

    # Calculate mean pixel value of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If corners are bright (white background), invert
    # Assuming 8-bit image: 0 is black, 255 is white
    if corner_mean > 127:
        return cv2.bitwise_not(img)
    return img


def get_morphometrics(img):
    """
    Calculates Hu Moments and geometric scalars from a binary image.

    Args:
        img (np.ndarray): Input binary image (foreground=255).

    Returns:
        np.ndarray: A 1D array containing 11 features:
                    [Hu1...Hu7, AspectRatio, Extent, Solidity, Eccentricity]
    """
    # Ensure image is strictly binary for contour finding
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return zeros if no contour found (empty image)
    if not contours:
        return np.zeros(11)

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Calculate Moments
    M = cv2.moments(cnt)
    area = M["m00"]

    # Avoid division by zero for very small/noise contours
    if area == 0:
        return np.zeros(11)

    # 1. Hu Moments (7 invariants)
    # We return raw Hu moments; downstream models (LDA) handle scaling/transform.
    hu = cv2.HuMoments(M).flatten()

    # 2. Aspect Ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # 3. Extent (Ratio of contour area to bounding rect area)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # 4. Solidity (Ratio of contour area to convex hull area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 5. Eccentricity (Derived from central moments)
    # Measures how much the shape deviates from a circle (0=circle, 1=line)
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]

    # Eigenvalues of the covariance matrix of the image distribution
    term1 = mu20 + mu02
    term2 = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)

    lambda1 = (term1 + term2) / 2
    lambda2 = (term1 - term2) / 2

    if lambda1 == 0:
        eccentricity = 0.0
    else:
        # e = sqrt(1 - (minor_axis / major_axis)^2) ~ sqrt(1 - lambda2/lambda1)
        eccentricity = np.sqrt(1 - lambda2 / lambda1)

    features = np.concatenate([hu, [aspect_ratio, extent, solidity, eccentricity]])

    return features


def process_image_batch(image_paths, cache_name, load_cached_data=True):
    """
    Processes a list of image paths to extract morphometric features.
    Implements caching to ./working/idea_64/ to avoid re-computation.

    Args:
        image_paths (list): List of relative paths to images (e.g., 'images/1.jpg').
        cache_name (str): Unique identifier for the cache file (e.g., 'train_morph').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing extracted features and 'id'.
    """
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing...")

    # 2. Compute Features
    print(f"Extracting morphometrics for {len(image_paths)} images...")

    feature_list = []
    ids = []

    # Define column names
    cols = [f"hu_{i}" for i in range(1, 8)] + [
        "aspect_ratio",
        "extent",
        "solidity",
        "eccentricity",
    ]

    for rel_path in image_paths:
        # Extract ID from path for alignment (e.g., 'images/123.jpg' -> 123)
        try:
            img_id = int(os.path.splitext(os.path.basename(rel_path))[0])
        except ValueError:
            img_id = -1

        full_path = os.path.join(INPUT_DIR, rel_path)

        # Default to zeros if file missing or read fails
        feats = np.zeros(11)

        if os.path.exists(full_path):
            # Read as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = correct_polarity(img)
                feats = get_morphometrics(img)

        feature_list.append(feats)
        ids.append(img_id)

    # Create DataFrame
    df_features = pd.DataFrame(feature_list, columns=cols)
    df_features["id"] = ids

    # 3. Save to Cache
    print(f"Saving morphometrics to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
