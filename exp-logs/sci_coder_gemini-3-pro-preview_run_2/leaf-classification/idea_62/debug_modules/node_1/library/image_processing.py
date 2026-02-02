import os
import cv2
import numpy as np
import pandas as pd
import library.config as conf


def load_and_correct_image(image_path):
    """
    Loads a binary leaf image and ensures the leaf is the foreground (white)
    and background is black.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        numpy.ndarray: A binary image (0 and 255) where the leaf is 255.
                       Returns None if image cannot be loaded.
    """
    if not os.path.exists(image_path):
        return None

    # Load image (handle potential grayscale or color input)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # Ensure single channel
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize to ensure clean 0/255 values (handling potential JPG artifacts)
    # Using Otsu's binarization for robustness
    _, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Polarity Correction
    # Check the 4 corners to determine background color
    h, w = img_binary.shape
    corners = [
        img_binary[0, 0],
        img_binary[0, w - 1],
        img_binary[h - 1, 0],
        img_binary[h - 1, w - 1],
    ]
    corner_mean = np.mean(corners)

    # If corners are white (high value), background is white.
    # We want leaf to be white (foreground). So we invert.
    # Threshold is 0.5 * 255 = 127.5
    threshold_val = conf.POLARITY_THRESHOLD * 255

    if corner_mean > threshold_val:
        img_binary = cv2.bitwise_not(img_binary)

    return img_binary


def extract_morphometrics(img):
    """
    Extracts Hu Moments and Geometric Scalars from a binary leaf image.

    Args:
        img (numpy.ndarray): Binary image (leaf=255, bg=0).

    Returns:
        numpy.ndarray: A 1D array of float64 features (len=11).
                       [Hu0..Hu6, AspectRatio, Solidity, Extent, Eccentricity]
    """
    # Initialize feature vector
    # 7 Hu moments + 4 Geometric scalars = 11 features
    features = np.zeros(11, dtype=conf.FLOAT_PRECISION)

    if img is None:
        return features

    # 1. Hu Moments
    # Calculate moments on the binary image directly
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    features[0:7] = hu_moments

    # 2. Geometric Scalars
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    if area == 0:
        return features

    # Bounding Rectangle
    x, y, w, h = cv2.boundingRect(cnt)

    # Convex Hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    # --- Scalar Calculations ---

    # Aspect Ratio (Width / Height)
    if h > 0:
        features[7] = float(w) / h
    else:
        features[7] = 0.0

    # Solidity (Contour Area / Hull Area)
    if hull_area > 0:
        features[8] = area / hull_area
    else:
        features[8] = 0.0

    # Extent (Contour Area / Bounding Rect Area)
    rect_area = w * h
    if rect_area > 0:
        features[9] = area / rect_area
    else:
        features[9] = 0.0

    # Eccentricity
    # Requires fitting an ellipse. Needs at least 5 points.
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2) where a is semi-major, b is semi-minor
                # (b/a)^2 = (minor/major)^2
                ratio_sq = (minor_axis / major_axis) ** 2
                features[10] = np.sqrt(1 - ratio_sq)
            else:
                features[10] = 0.0
        except:
            features[10] = 0.0
    else:
        features[10] = 0.0

    return features


def process_dataset_morphometrics(df, dataset_name, load_cached_data=True):
    """
    Orchestrates the extraction of morphometric features for a dataset.
    Handles caching to parquet files.

    Args:
        df (pd.DataFrame): Dataframe containing 'image_path' column.
        dataset_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        pd.DataFrame: DataFrame containing the extracted features.
    """
    # Ensure cache directory exists
    os.makedirs(conf.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(conf.CACHE_DIR, f"morphometrics_{dataset_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}...")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting morphometrics for {dataset_name} ({len(df)} images)...")

    feature_list = []

    # Define column names
    col_names = [f"hu_{i}" for i in range(7)] + [
        "aspect_ratio",
        "solidity",
        "extent",
        "eccentricity",
    ]

    for idx, row in df.iterrows():
        # Construct full path
        # Metadata image_path is relative to input dir (e.g., "images/1.jpg")
        full_path = os.path.join(conf.INPUT_DIR, row[conf.IMAGE_PATH_COL])

        # Process
        img = load_and_correct_image(full_path)
        feats = extract_morphometrics(img)

        feature_list.append(feats)

    # Create DataFrame
    features_df = pd.DataFrame(feature_list, columns=col_names, index=df.index)

    # Ensure float64 precision
    features_df = features_df.astype(conf.FLOAT_PRECISION)

    # 3. Save to cache
    print(f"Saving morphometrics to {cache_path}...")
    features_df.to_parquet(cache_path)

    return features_df
