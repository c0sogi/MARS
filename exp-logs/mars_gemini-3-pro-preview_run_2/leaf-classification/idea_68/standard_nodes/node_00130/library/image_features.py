import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    DTYPE,
)
from library.utils import set_seed


def check_polarity(img):
    """
    Checks if the background is white (pixel value > 127).
    If so, inverts the image so the object is foreground (white) and background is black.

    Args:
        img (np.ndarray): Grayscale image (uint8).

    Returns:
        np.ndarray: Polarity-corrected image.
    """
    if img is None:
        return None

    # Check corners to determine background color
    h, w = img.shape
    patch_size = 5

    # Handle very small images
    if h < patch_size or w < patch_size:
        patch_size = min(h, w)

    # Define corners
    corners = [
        img[0:patch_size, 0:patch_size],
        img[0:patch_size, w - patch_size : w],
        img[h - patch_size : h, 0:patch_size],
        img[h - patch_size : h, w - patch_size : w],
    ]

    # Calculate mean of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If background is bright (> 127), invert to make background dark
    if corner_mean > 127:
        return cv2.bitwise_not(img)
    return img


def extract_morphometrics(img):
    """
    Extracts Hu Moments and Geometric Scalars from a binary image.

    Args:
        img (np.ndarray): Polarity-corrected binary image (object is bright).

    Returns:
        dict: Dictionary of features.
    """
    # Default zero features
    default_feats = {f"hu_{i}": 0.0 for i in range(7)}
    default_feats.update(
        {"aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0, "eccentricity": 0.0}
    )

    if img is None:
        return default_feats

    # Ensure binary
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return default_feats

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return default_feats

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    feats = {f"hu_{i}": float(hu_moments[i]) for i in range(7)}

    # 2. Geometric Scalars
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    # Aspect Ratio
    aspect_ratio = float(w) / h if h > 0 else 0.0
    feats["aspect_ratio"] = aspect_ratio

    # Extent
    extent = float(area) / rect_area if rect_area > 0 else 0.0
    feats["extent"] = extent

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0
    feats["solidity"] = solidity

    # Eccentricity
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            # fitEllipse returns ((x,y), (MA, ma), angle)
            # Note: OpenCV docs are sometimes ambiguous, but typically returns (width, height) of bounding box
            # We take max/min to ensure we have major/minor axes
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            a = major_axis / 2
            b = minor_axis / 2

            if a > 0:
                # e = sqrt(1 - b^2/a^2)
                eccentricity = np.sqrt(1 - (b**2 / a**2))
            else:
                eccentricity = 0.0
        except Exception:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    feats["eccentricity"] = eccentricity

    return feats


def process_dataset(metadata_path, dataset_name, load_cached_data=True):
    """
    Loads metadata, processes images to extract morphometric features,
    and returns a DataFrame. Handles caching.

    Args:
        metadata_path (str): Path to the metadata CSV.
        dataset_name (str): Name tag for cache file (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and extracted features.
    """
    cache_file = os.path.join(CACHE_DIR, f"morphometrics_{dataset_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Extracting morphometrics for {dataset_name}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)

    results = []

    for idx, row in df_meta.iterrows():
        image_id = row["id"]
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            # Fallback if image missing
            feats = extract_morphometrics(None)
        else:
            # Read image
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            # Correct polarity
            img_corr = check_polarity(img)
            # Extract features
            feats = extract_morphometrics(img_corr)

        feats["id"] = image_id
        results.append(feats)

    # Create DataFrame
    df_features = pd.DataFrame(results)

    # Ensure ID is first column and cast to correct types
    cols = ["id"] + [c for c in df_features.columns if c != "id"]
    df_features = df_features[cols]

    # Save to cache
    print(f"Saving morphometrics to {cache_file}...")
    df_features.to_parquet(cache_file, index=False)

    return df_features


def get_morphometric_features(load_cached_data=True):
    """
    Main entry point to get all morphometric features for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (df_train, df_val, df_test) containing IDs and morphometric features.
    """
    set_seed()

    df_train = process_dataset(TRAIN_DATA_PATH, "train", load_cached_data)
    df_val = process_dataset(VAL_DATA_PATH, "val", load_cached_data)
    df_test = process_dataset(TEST_DATA_PATH, "test", load_cached_data)

    return df_train, df_val, df_test
