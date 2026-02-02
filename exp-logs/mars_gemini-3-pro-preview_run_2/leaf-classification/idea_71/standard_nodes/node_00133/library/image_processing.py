import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHE_DIR

# =============================================================================
# CORE IMAGE PROCESSING FUNCTIONS
# =============================================================================


def check_polarity_and_invert(img):
    """
    Checks the polarity of the binary image. The dataset consists of leaves
    against a background. We want the leaf to be the foreground (high intensity, 1.0)
    and the background to be low intensity (0.0).

    If the corners of the image are high intensity (white), it implies a white
    background. In this case, we invert the image.

    Args:
        img (np.ndarray): Input grayscale image normalized to [0, 1].

    Returns:
        np.ndarray: Polarity-corrected image.
    """
    h, w = img.shape
    # Define corner region size (5% of dimension, min 1 pixel)
    c_w, c_h = max(1, w // 20), max(1, h // 20)

    # Extract corners
    corners = [
        img[0:c_h, 0:c_w],  # Top-left
        img[0:c_h, w - c_w : w],  # Top-right
        img[h - c_h : h, 0:c_w],  # Bottom-left
        img[h - c_h : h, w - c_w : w],  # Bottom-right
    ]

    # Calculate mean intensity of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If background is white (> 0.5), invert to make leaf foreground
    if corner_mean > 0.5:
        return 1.0 - img

    return img


def get_eccentricity(moments):
    """
    Calculates eccentricity from image moments.
    Eccentricity is the ratio of the focal distance to the major axis length.

    Args:
        moments (dict): OpenCV moments dictionary.

    Returns:
        float: Eccentricity value [0, 1).
    """
    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    # Eigenvalues of the covariance matrix of the image distribution
    delta = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
    lambda1 = (mu20 + mu02 + delta) / 2
    lambda2 = (mu20 + mu02 - delta) / 2

    if lambda1 == 0:
        return 0.0

    # Eccentricity = sqrt(1 - (minor_axis/major_axis)^2)
    # lambda1 proportional to major axis^2, lambda2 to minor axis^2
    ratio = lambda2 / lambda1
    # Clip ratio to [0, 1] to avoid numerical errors
    ratio = max(0.0, min(1.0, ratio))

    return np.sqrt(1.0 - ratio)


def extract_single_image_features(full_path):
    """
    Loads an image and extracts Hu Moments and Geometric Scalars.

    Args:
        full_path (str): Absolute path to the image file.

    Returns:
        np.ndarray: Array of 11 features (7 Hu Moments + 4 Geometric Scalars).
                    Returns zeros if processing fails.
    """
    # 11 Features: Hu[0-6], AspectRatio, Solidity, Extent, Eccentricity
    zeros_result = np.zeros(11, dtype=np.float64)

    if not os.path.exists(full_path):
        return zeros_result

    # Read image as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return zeros_result

    # Normalize to [0, 1]
    img = img.astype(np.float64) / 255.0

    # Correct Polarity
    img = check_polarity_and_invert(img)

    # Convert to binary uint8 for contour extraction
    # Leaf is now ~1.0, Background ~0.0
    _, thresh = cv2.threshold((img * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return zeros_result

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # Calculate Moments
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return zeros_result

    # --- Feature 1: Hu Moments (7 invariants) ---
    hu = cv2.HuMoments(M).flatten()

    # --- Feature 2: Geometric Scalars ---
    area = M["m00"]

    # Aspect Ratio & Extent (Bounding Rect)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = area / rect_area if rect_area > 0 else 0.0

    # Solidity (Convex Hull)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    eccentricity = get_eccentricity(M)

    # Combine all features
    features = np.concatenate([hu, [aspect_ratio, solidity, extent, eccentricity]])
    return features


# =============================================================================
# BATCH PROCESSING
# =============================================================================


def process_all_images(metadata_df, load_cached_data=True, debug_limit=None):
    """
    Iterates through image IDs in the dataframe, processes each image to extract
    morphometric features, and returns a DataFrame. Implements caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'image_path'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug_limit (int, optional): If set, limits processing to N images for debugging.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the 11 extracted features.
    """
    # Apply debug limit if specified
    if debug_limit is not None:
        metadata_df = metadata_df.head(debug_limit)

    # Generate a unique cache filename based on the hash of the IDs
    # This ensures that different splits (train/val/test) get their own cache files
    ids_hash = pd.util.hash_pandas_object(metadata_df["id"], index=False).sum()
    cache_filename = f"morphometrics_{ids_hash}.parquet"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Extracting morphometrics for {len(metadata_df)} images...")

    # 2. Process Images
    feature_names = [f"hu_{i}" for i in range(1, 8)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]

    data_matrix = []
    ids = metadata_df["id"].values

    for idx, row in metadata_df.iterrows():
        # Resolve full image path
        # Metadata 'image_path' is relative to INPUT_DIR (e.g., "images/12.jpg")
        if "image_path" in row:
            rel_path = row["image_path"]
        else:
            # Fallback if column missing
            rel_path = os.path.join("images", f"{row['id']}.jpg")

        full_path = os.path.join(INPUT_DIR, rel_path)

        # Extract features
        features = extract_single_image_features(full_path)
        data_matrix.append(features)

    # 3. Create DataFrame
    df_features = pd.DataFrame(data_matrix, columns=feature_names)
    df_features.insert(0, "id", ids)

    # 4. Save Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_features.to_parquet(cache_path, index=False)
    print(f"Saved morphometrics to {cache_path}")

    return df_features
