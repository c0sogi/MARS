import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_63"


def get_eccentricity(contour):
    """
    Calculates eccentricity from a contour using ellipse fitting.
    Eccentricity e = sqrt(1 - (b/a)^2) where a is semi-major axis and b is semi-minor axis.
    """
    if len(contour) < 5:
        return 0.0
    try:
        # fitEllipse returns ((x,y), (width, height), angle)
        (x, y), (d1, d2), angle = cv2.fitEllipse(contour)

        major_axis = max(d1, d2)
        minor_axis = min(d1, d2)

        if major_axis == 0:
            return 0.0

        # a = major/2, b = minor/2
        # e = sqrt(1 - (minor/2 / major/2)^2) = sqrt(1 - (minor/major)^2)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        return eccentricity
    except Exception:
        return 0.0


def process_single_image(image_rel_path):
    """
    Extracts 11 morphometric features from a single image.
    Features: 7 Hu Moments + Aspect Ratio, Solidity, Extent, Eccentricity.

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        np.ndarray: A 1D array of 11 float64 features.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Initialize default vector (11 features)
    features = np.zeros(11, dtype=np.float64)

    if not os.path.exists(full_path):
        return features

    # Read as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Normalize to 0-1 for processing
    img_norm = img.astype(np.float64) / 255.0

    # Polarity Correction: Check corners to determine background color
    h, w = img_norm.shape
    c_w, c_h = max(1, w // 10), max(1, h // 10)

    corners = [
        img_norm[0:c_h, 0:c_w],  # Top-left
        img_norm[0:c_h, w - c_w : w],  # Top-right
        img_norm[h - c_h : h, 0:c_w],  # Bottom-left
        img_norm[h - c_h : h, w - c_w : w],  # Bottom-right
    ]

    corner_mean = np.mean([np.mean(c) for c in corners])

    # If background is white (high intensity), invert so leaf is white (1.0)
    if corner_mean > 0.5:
        img_binary = 1.0 - img_norm
    else:
        img_binary = img_norm

    # Threshold to get strict binary mask for contour finding
    # Convert back to uint8: 0 or 255
    _, thresh = cv2.threshold(
        (img_binary * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Take largest contour by area as the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # --- Feature 1-7: Hu Moments ---
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    features[0:7] = hu

    # --- Geometric Scalars ---
    x, y, cw, ch = cv2.boundingRect(cnt)

    # Feature 8: Aspect Ratio
    if ch > 0:
        aspect_ratio = float(cw) / ch
    else:
        aspect_ratio = 0.0
    features[7] = aspect_ratio

    # Feature 9: Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0.0
    features[8] = solidity

    # Feature 10: Extent (Area / Bounding Rect Area)
    rect_area = cw * ch
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0.0
    features[9] = extent

    # Feature 11: Eccentricity
    features[10] = get_eccentricity(cnt)

    return features


def extract_morphometrics(metadata_df, load_cached_data=True, cache_name="train"):
    """
    Extracts morphometric features for the given dataframe.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path'.
        load_cached_data (bool): Whether to load from cache if available.
        cache_name (str): Identifier for the cache file (e.g., 'train', 'val', 'test').

    Returns:
        np.ndarray: Feature matrix of shape (N, 11).
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"morphometrics_{cache_name}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        try:
            data = np.load(cache_path)
            if data.shape[0] == len(metadata_df):
                return data
            else:
                print(
                    f"Cache shape mismatch ({data.shape[0]} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(
        f"Extracting morphometrics for {cache_name} set ({len(metadata_df)} images)..."
    )

    feature_list = []
    image_paths = metadata_df["image_path"].values

    for i, path in enumerate(image_paths):
        feats = process_single_image(path)
        feature_list.append(feats)

    X_morph = np.array(feature_list, dtype=np.float64)

    # 3. Save to cache
    print(f"Saving morphometrics to {cache_path}")
    np.save(cache_path, X_morph)

    return X_morph
