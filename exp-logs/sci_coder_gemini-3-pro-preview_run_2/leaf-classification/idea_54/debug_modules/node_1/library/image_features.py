import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    INVERT_THRESHOLD,
    CORNER_MARGIN,
    FLOAT_PRECISION,
)


def process_single_image(image_rel_path):
    """
    Reads an image, corrects polarity, and extracts Hu Moments and Geometric features.

    Args:
        image_rel_path (str): Relative path to the image file.

    Returns:
        np.ndarray: A 1D array containing 7 Hu moments and 4 geometric scalars.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Return zeros if path doesn't exist (safety check)
    if not os.path.exists(full_path):
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(11, dtype=FLOAT_PRECISION)

    # --- Polarity Correction ---
    # Check corners to see if background is white (high pixel values)
    h, w = img.shape
    corners = [
        img[0:CORNER_MARGIN, 0:CORNER_MARGIN],
        img[0:CORNER_MARGIN, w - CORNER_MARGIN : w],
        img[h - CORNER_MARGIN : h, 0:CORNER_MARGIN],
        img[h - CORNER_MARGIN : h, w - CORNER_MARGIN : w],
    ]

    # Calculate mean of corner pixels
    corner_pixels = np.concatenate([c.flatten() for c in corners])
    mean_val = np.mean(corner_pixels) if corner_pixels.size > 0 else 0

    # Threshold for inversion (assuming 0-255 image)
    # If corners are bright, background is white -> Invert to make leaf white on black
    if mean_val > (INVERT_THRESHOLD * 255.0):
        img = cv2.bitwise_not(img)

    # --- Feature Extraction ---

    # 1. Hu Moments (Shape Invariants)
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. Geometric Scalars via Contours
    # Binary threshold to ensure clean contours (leaf is white, bg is black)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    geo_features = np.zeros(
        4, dtype=FLOAT_PRECISION
    )  # [aspect_ratio, solidity, extent, eccentricity]

    if contours:
        # Assume largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area > 0:
            # Bounding Rect -> Aspect Ratio, Extent
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            rect_area = w_rect * h_rect

            aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0
            extent = area / rect_area if rect_area > 0 else 0.0

            # Convex Hull -> Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            # Ellipse Fit -> Eccentricity
            eccentricity = 0.0
            if len(cnt) >= 5:
                try:
                    # fitEllipse returns (center, (MA, ma), angle)
                    # Note: OpenCV returns axes lengths (diameters), not radii
                    (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)

                    # Sort axes to identify major and minor
                    axes = sorted([d1, d2])
                    minor_axis, major_axis = axes[0], axes[1]

                    if major_axis > 0:
                        # e = sqrt(1 - (b/a)^2)
                        eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                except Exception:
                    # Fallback if ellipse fitting fails
                    eccentricity = 0.0

            geo_features = np.array(
                [aspect_ratio, solidity, extent, eccentricity], dtype=FLOAT_PRECISION
            )

    # Combine features
    return np.concatenate([hu_moments, geo_features]).astype(FLOAT_PRECISION)


def get_morphometric_features(df, dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a given dataframe of images.
    Handles caching to disk.

    Args:
        df (pd.DataFrame): Dataframe containing 'image_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, 11).
    """
    cache_file = os.path.join(CACHE_DIR, f"{dataset_name}_morphometrics.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached morphometrics for '{dataset_name}' from {cache_file}")
        return np.load(cache_file)

    # Compute features
    print(f"Extracting morphometrics for '{dataset_name}'...")
    features_list = []

    # Ensure image_path column exists
    if "image_path" not in df.columns:
        raise ValueError("Dataframe must contain 'image_path' column.")

    paths = df["image_path"].values

    for i, path in enumerate(paths):
        feat_vector = process_single_image(path)
        features_list.append(feat_vector)

    X = np.array(features_list, dtype=FLOAT_PRECISION)

    # Save to cache
    np.save(cache_file, X)
    print(f"Saved morphometrics for '{dataset_name}' to {cache_file}")

    return X
