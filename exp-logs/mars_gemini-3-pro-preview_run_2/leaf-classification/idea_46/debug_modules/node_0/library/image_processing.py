import os
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from library import config


def correct_polarity(img):
    """
    Inverts the image if the background is white (mean > threshold).
    Ensures the leaf is the foreground (white) against a black background.

    Args:
        img (np.ndarray): Input image (grayscale).

    Returns:
        np.ndarray: Polarity-corrected image.
    """
    # Calculate mean pixel intensity
    mean_intensity = np.mean(img)

    # Threshold calculation based on config
    # config.POLARITY_THRESHOLD is typically 0.5 (float)
    # Image is uint8 (0-255)
    threshold_val = config.POLARITY_THRESHOLD * 255.0

    if mean_intensity > threshold_val:
        # Invert image: 255 - pixel_value
        return cv2.bitwise_not(img)

    return img


def extract_morphometrics(img):
    """
    Extracts Hu Moments and Geometric Scalars from a binary image.

    Features extracted:
    1. Hu Moments (7 invariants)
    2. Aspect Ratio
    3. Solidity
    4. Extent
    5. Eccentricity

    Args:
        img (np.ndarray): Input binary/grayscale image.

    Returns:
        np.ndarray: 1D array of 11 features.
    """
    # Ensure image is binary for contour detection
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize feature vector (11 features)
    # 7 Hu moments + 4 Geometric Scalars
    features = np.zeros(11, dtype=config.FLOAT_PRECISION)

    if not contours:
        return features

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # --- 1. Hu Moments (7 features) ---
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    features[0:7] = hu

    # --- 2. Geometric Scalars (4 features) ---

    # Bounding Rectangle -> Aspect Ratio, Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = area / rect_area if rect_area > 0 else 0.0

    # Convex Hull -> Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Ellipse Fit -> Eccentricity
    eccentricity = 0.0
    if len(cnt) >= 5:  # fitEllipse requires at least 5 points
        try:
            # fitEllipse returns ((center_x, center_y), (width, height), angle)
            (x_e, y_e), (MA, ma), angle = cv2.fitEllipse(cnt)

            # MA and ma are the lengths of the axes (diameters)
            axis_lengths = sorted([MA, ma])
            minor_axis = axis_lengths[0]
            major_axis = axis_lengths[1]

            if major_axis > 0:
                # Eccentricity e = sqrt(1 - (b/a)^2)
                ratio_sq = (minor_axis / major_axis) ** 2
                eccentricity = np.sqrt(1 - ratio_sq)
        except Exception:
            # Fallback if ellipse fitting fails numerically
            eccentricity = 0.0

    features[7] = aspect_ratio
    features[8] = solidity
    features[9] = extent
    features[10] = eccentricity

    return features


def _process_single_image(image_path_rel):
    """
    Internal helper to process a single image path.
    Used for parallel execution.
    """
    full_path = os.path.join(config.INPUT_DIR, image_path_rel)

    # Load image as grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        # Return zero vector if image cannot be loaded
        return np.zeros(11, dtype=config.FLOAT_PRECISION)

    # Correct polarity (ensure leaf is white)
    img = correct_polarity(img)

    # Extract features
    features = extract_morphometrics(img)

    return features


def get_morphometric_features(metadata_df, load_cached_data=True):
    """
    Generates or loads polarity-corrected morphometric features for the given metadata.

    Args:
        metadata_df (pd.DataFrame): Dataframe containing 'id' and 'image_path' columns.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, 11).
    """
    # Generate a robust cache filename based on the IDs in the dataframe
    ids_hash = pd.util.hash_pandas_object(metadata_df["id"], index=False).sum()
    cache_filename = f"morphometrics_{ids_hash}.npy"
    cache_path = os.path.join(config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometric features from {cache_path}")
        try:
            features = np.load(cache_path)
            if features.shape[0] == len(metadata_df):
                return features
            else:
                print("Cached file dimension mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting morphometric features for {len(metadata_df)} images...")

    image_paths = metadata_df["image_path"].tolist()

    # Execute in parallel
    results = Parallel(n_jobs=config.N_JOBS)(
        delayed(_process_single_image)(path) for path in image_paths
    )

    features = np.array(results, dtype=config.FLOAT_PRECISION)

    # 3. Save to cache
    try:
        np.save(cache_path, features)
        print(f"Saved morphometric features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return features
