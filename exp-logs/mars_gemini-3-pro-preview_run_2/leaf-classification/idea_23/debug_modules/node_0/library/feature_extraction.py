import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR, BINARY_THRESHOLD


def process_image(image_rel_path):
    """
    Extracts orthogonal morphometric features from a single binary image.

    Features extracted (11 total):
    - Hu Moments (7 features): Scale, rotation, and translation invariant descriptors.
    - Aspect Ratio: Ratio of width to height of the bounding rectangle.
    - Solidity: Ratio of contour area to its convex hull area.
    - Extent: Ratio of contour area to bounding rectangle area.
    - Eccentricity: Measure of how much the shape deviates from a circle (0=circle, 1=line).

    Args:
        image_rel_path (str): Relative path to the image from the input directory.

    Returns:
        np.ndarray: A 1D array of shape (11,) containing float64 features.
                    Returns zeros if image cannot be processed or no contour is found.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Read image
    # The images are binary black leaves against white backgrounds, or vice versa.
    # We read as grayscale.
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return np.zeros(11, dtype=np.float64)

    # Threshold to ensure strict binary
    # Assuming leaves are black on white or white on black.
    # Usually, object of interest should be white for contour detection in OpenCV.
    # Let's check the corner pixel to determine background color.
    # If corner is white (255), we invert.
    if img[0, 0] > 127:
        img = cv2.bitwise_not(img)

    _, thresh = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(11, dtype=np.float64)

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()  # 7 values

    # Avoid log transform here (log(0) issues), we rely on PowerTransformer later in the pipeline
    # to handle the distribution of these raw moments.

    # 2. Geometric Scalars
    area = moments["m00"]

    # Safety check for area to avoid division by zero
    if area == 0:
        return np.zeros(11, dtype=np.float64)

    # Bounding Rect -> Aspect Ratio, Extent
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Convex Hull -> Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Ellipse Fit -> Eccentricity
    # Requires at least 5 points to fit an ellipse
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are diameters (major/minor axes)
            # Ensure a is major, b is minor
            major = max(axis1, axis2)
            minor = min(axis1, axis2)
            if major > 0:
                # e = sqrt(1 - (b/a)^2)
                eccentricity = np.sqrt(1 - (minor / major) ** 2)
        except:
            # Fallback if fitEllipse fails
            eccentricity = 0.0

    # Combine features
    # [Hu1...Hu7, AspectRatio, Solidity, Extent, Eccentricity]
    features = np.concatenate(
        [hu_moments, np.array([aspect_ratio, solidity, extent, eccentricity])]
    )

    return features.astype(np.float64)


def extract_morphometrics(metadata_df, dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a given dataset dataframe.
    Implements caching to ./working/idea_23/.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path' column.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, 11).
    """
    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_morphometrics.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}...")
        try:
            features = np.load(cache_path)
            if features.shape[0] == len(metadata_df):
                return features
            else:
                print(
                    f"Cache shape mismatch ({features.shape[0]} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Extracting morphometrics for {dataset_name} ({len(metadata_df)} images)...")

    feature_list = []
    image_paths = metadata_df["image_path"].values

    for i, rel_path in enumerate(image_paths):
        feats = process_image(rel_path)
        feature_list.append(feats)

        # Optional: Print progress every 100 images
        if (i + 1) % 100 == 0:
            pass  # Silent execution as per instructions

    features = np.array(feature_list, dtype=np.float64)

    # 3. Save Cache
    try:
        np.save(cache_path, features)
        print(f"Saved morphometrics to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return features
