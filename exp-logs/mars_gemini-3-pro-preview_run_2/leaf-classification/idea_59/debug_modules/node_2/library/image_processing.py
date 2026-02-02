import os
import cv2
import numpy as np
import pandas as pd
from library.utils import ensure_float64

# Constants
CACHE_DIR = "./working/idea_59"
INPUT_DIR = "./input"


def get_polarity_corrected_image(image):
    """
    Inverts the image if the background appears to be white (high intensity).
    The task description states 'black leaves against white backgrounds',
    but OpenCV contour detection assumes white objects on black backgrounds.

    Args:
        image (np.ndarray): Input image (grayscale).

    Returns:
        np.ndarray: Polarity-corrected image (Object is white, Background is black).
    """
    # Ensure grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape

    # Sample 5x5 regions from the four corners
    corners = [
        image[0:5, 0:5],
        image[0:5, w - 5 : w],
        image[h - 5 : h, 0:5],
        image[h - 5 : h, w - 5 : w],
    ]

    # Calculate global mean of corner regions
    # If corners are white (255), mean will be high.
    corner_mean = np.mean([np.mean(c) for c in corners])

    # Threshold: if mean > 127 (midpoint of 0-255), assume white background
    if corner_mean > 127:
        # Invert to make leaf white (255) and background black (0)
        return cv2.bitwise_not(image)

    return image


def extract_morphometrics(image):
    """
    Extracts Hu Moments (7) and Geometric Scalars (4) from a binary image.

    Args:
        image (np.ndarray): Binary image (white object, black background).

    Returns:
        np.ndarray: 1D array of 11 float64 features.
    """
    # Ensure binary thresholding
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # --- 1. Hu Moments ---
    # Compute moments
    moments = cv2.moments(thresh)
    # Compute Hu moments (7 invariants)
    hu_moments = cv2.HuMoments(moments).flatten()

    # --- 2. Geometric Scalars ---
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    aspect_ratio = 0.0
    solidity = 0.0
    extent = 0.0
    eccentricity = 0.0

    if contours:
        # Assume the largest contour corresponds to the leaf
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Bounding Rectangle -> Aspect Ratio & Extent
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        if h > 0:
            aspect_ratio = float(w) / float(h)

        if rect_area > 0:
            extent = area / rect_area

        # Convex Hull -> Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = area / hull_area

        # Fit Ellipse -> Eccentricity
        # fitEllipse requires at least 5 points
        if len(cnt) >= 5:
            try:
                # (x, y), (major_axis, minor_axis), angle
                # Note: OpenCV does not guarantee which axis is returned first in (MA, ma)
                (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)

                # Semi-axes
                a = max(d1, d2) / 2.0
                b = min(d1, d2) / 2.0

                if a > 0:
                    # Eccentricity formula: sqrt(1 - (b/a)^2)
                    eccentricity = np.sqrt(1 - (b / a) ** 2)
            except Exception:
                # Fallback if ellipse fitting fails numerically
                eccentricity = 0.0

    # Concatenate features
    geometric_scalars = np.array([aspect_ratio, solidity, extent, eccentricity])
    features = np.concatenate([hu_moments, geometric_scalars])

    return ensure_float64(features)


def process_dataset(metadata_path, dataset_name, load_cached_data=True):
    """
    Loads metadata, processes images to extract morphometrics, and returns a DataFrame.
    Implements caching to disk using Parquet.

    Args:
        metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
        dataset_name (str): Identifier for the dataset (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and extracted features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"morphometrics_{dataset_name}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for '{dataset_name}' from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Process from Scratch
    print(f"Processing images for '{dataset_name}' from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Feature column names
    hu_cols = [f"hu_{i}" for i in range(7)]
    geo_cols = ["aspect_ratio", "solidity", "extent", "eccentricity"]
    feature_cols = hu_cols + geo_cols

    ids = []
    features_list = []

    for idx, row in df_meta.iterrows():
        image_id = row["id"]
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Default zero vector if image load fails
        img_features = np.zeros(len(feature_cols))

        if os.path.exists(full_path):
            # Read image as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Correct polarity (Leaf should be white)
                img_corr = get_polarity_corrected_image(img)
                # Extract features
                img_features = extract_morphometrics(img_corr)

        ids.append(image_id)
        features_list.append(img_features)

    # Create DataFrame
    df_features = pd.DataFrame(features_list, columns=feature_cols)
    df_features.insert(0, "id", ids)

    # Enforce float64 precision
    for col in feature_cols:
        df_features[col] = ensure_float64(df_features[col].values)

    # 3. Save Cache
    print(f"Saving morphometrics for '{dataset_name}' to {cache_path}")
    df_features.to_parquet(cache_path, index=False)

    return df_features
