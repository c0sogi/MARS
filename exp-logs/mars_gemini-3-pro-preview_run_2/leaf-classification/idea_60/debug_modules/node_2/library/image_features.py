import os
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from library import config, utils


def check_polarity(image, threshold=config.MORPH_INVERT_THRESHOLD):
    """
    Determines if the image needs inversion based on corner pixel intensity.

    Args:
        image: Numpy array of the image (uint8).
        threshold: Threshold ratio (0-1) for determining background color.
                   If corner mean > threshold * 255, it assumes white background.

    Returns:
        bool: True if image should be inverted, False otherwise.
    """
    if image is None:
        return False

    h, w = image.shape
    # Define corner sample size (e.g., 10x10 pixels)
    c_w, c_h = min(10, w), min(10, h)

    # Extract corners
    tl = image[0:c_h, 0:c_w]
    tr = image[0:c_h, w - c_w : w]
    bl = image[h - c_h : h, 0:c_w]
    br = image[h - c_h : h, w - c_w : w]

    # Calculate mean intensity of corners
    corner_mean = (np.mean(tl) + np.mean(tr) + np.mean(bl) + np.mean(br)) / 4.0

    # If corners are bright (white background), we need to invert
    return corner_mean > (threshold * 255)


def extract_morphometrics(image_path):
    """
    Extracts Hu Moments and Geometric Scalars from a binary leaf image.

    Args:
        image_path: Full path to the image file.

    Returns:
        dict: Dictionary containing extracted features.
    """
    # Define feature names for consistency
    feature_names = [f"hu_{i}" for i in range(7)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]
    defaults = {k: 0.0 for k in feature_names}

    if not os.path.exists(image_path):
        return defaults

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return defaults

    # Check polarity and invert if needed (Leaf should be white, Background black)
    if check_polarity(img):
        img = cv2.bitwise_not(img)

    # Binarize to ensure clean mask
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return defaults

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return defaults

    # 1. Hu Moments (7 invariants)
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. Geometric Scalars
    x, y, w, h = cv2.boundingRect(cnt)

    # Aspect Ratio
    aspect_ratio = float(w) / h if h > 0 else 0

    # Solidity (Area / Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Extent (Area / Bounding Rect Area)
    extent = area / (w * h) if (w * h) > 0 else 0

    # Eccentricity
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            # fitEllipse returns (center, (width, height), angle)
            (e_x, e_y), (d1, d2), angle = cv2.fitEllipse(cnt)
            # Semi-axes
            a = max(d1, d2) / 2.0
            b = min(d1, d2) / 2.0
            if a > 0:
                eccentricity = np.sqrt(1 - (b / a) ** 2)
        except Exception:
            eccentricity = 0.0

    # Pack results
    features = {}
    for i in range(7):
        features[f"hu_{i}"] = hu_moments[i]

    features["aspect_ratio"] = aspect_ratio
    features["solidity"] = solidity
    features["extent"] = extent
    features["eccentricity"] = eccentricity

    return features


def process_all_images(metadata_df, cache_name, load_cached_data=True):
    """
    Extracts morphometrics for all images in the provided DataFrame.
    Implements caching to avoid re-computation.

    Args:
        metadata_df: DataFrame containing 'id' and 'image_path' columns.
        cache_name: String identifier for the cache file (e.g., 'train', 'test').
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and extracted features.
    """
    cache_path = os.path.join(config.WORKING_DIR, f"morphometrics_{cache_name}.parquet")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing...")

    print(f"Extracting morphometrics for {len(metadata_df)} images ({cache_name})...")

    # 2. Prepare Paths
    # metadata_df['image_path'] is relative (e.g., "images/123.jpg")
    # We construct full absolute paths
    paths = [
        os.path.join(config.INPUT_DIR, rel_path)
        for rel_path in metadata_df["image_path"]
    ]
    ids = metadata_df["id"].values

    # 3. Parallel Execution
    results = Parallel(n_jobs=config.N_JOBS)(
        delayed(extract_morphometrics)(p) for p in paths
    )

    # 4. Construct DataFrame
    features_df = pd.DataFrame(results)
    features_df.insert(0, "id", ids)

    # Ensure float64 precision for features
    feature_cols = [c for c in features_df.columns if c != "id"]
    features_df[feature_cols] = utils.enforce_float64(features_df[feature_cols])

    # Ensure ID is integer
    features_df["id"] = features_df["id"].astype(int)

    # 5. Save Cache
    print(f"Saving morphometrics to {cache_path}")
    features_df.to_parquet(cache_path, index=False)

    return features_df
