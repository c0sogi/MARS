import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_57"


def extract_morphometrics(image_paths, cache_key, load_cached_data=True):
    """
    Extracts morphometric features (Hu Moments and Geometric Scalars) from a list of images.

    Args:
        image_paths (list): List of relative image paths (e.g., ['images/1.jpg', ...]).
        cache_key (str): Unique identifier for this dataset split (e.g., 'train', 'test').
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        np.ndarray: A float64 array of shape (N, 11) containing the extracted features.
    """
    set_seed(42)

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"morphometrics_{cache_key}.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached morphometrics from {cache_file}...")
        try:
            features = np.load(cache_file)
            if features.shape[0] == len(image_paths):
                return features.astype(np.float64)
            else:
                print("Cached data shape mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print(f"Extracting morphometrics for {len(image_paths)} images...")

    features_list = []

    for rel_path in image_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Initialize feature vector (11 features: 7 Hu + 4 Geometric)
        # Default to 0 if extraction fails
        sample_features = np.zeros(11, dtype=np.float64)

        if os.path.exists(full_path):
            try:
                # Read Image
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    # Polarity Correction
                    # Check corners to determine if background is white
                    h, w = img.shape
                    corners = [
                        img[0:5, 0:5],  # Top-left
                        img[0:5, w - 5 : w],  # Top-right
                        img[h - 5 : h, 0:5],  # Bottom-left
                        img[h - 5 : h, w - 5 : w],  # Bottom-right
                    ]
                    corner_mean = np.mean([np.mean(c) for c in corners])

                    # If background is white (high intensity), invert so object is white
                    if corner_mean > 127:
                        img = cv2.bitwise_not(img)

                    # Find Contours
                    contours, _ = cv2.findContours(
                        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Assume the largest contour is the leaf
                        cnt = max(contours, key=cv2.contourArea)

                        # 1. Hu Moments (7 features)
                        moments = cv2.moments(cnt)
                        hu = cv2.HuMoments(moments).flatten()
                        sample_features[0:7] = hu

                        # 2. Geometric Scalars
                        area = moments["m00"]

                        # Bounding Rect
                        x, y, rect_w, rect_h = cv2.boundingRect(cnt)
                        rect_area = rect_w * rect_h

                        # Convex Hull
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)

                        # Aspect Ratio
                        if rect_h > 0:
                            aspect_ratio = float(rect_w) / rect_h
                        else:
                            aspect_ratio = 0.0

                        # Extent
                        if rect_area > 0:
                            extent = float(area) / rect_area
                        else:
                            extent = 0.0

                        # Solidity
                        if hull_area > 0:
                            solidity = float(area) / hull_area
                        else:
                            solidity = 0.0

                        # Eccentricity
                        # Needs at least 5 points to fit ellipse
                        eccentricity = 0.0
                        if len(cnt) >= 5:
                            try:
                                (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
                                # axis1 and axis2 are diameters (major/minor axes depending on orientation)
                                major = max(axis1, axis2)
                                minor = min(axis1, axis2)
                                if major > 0:
                                    # e = sqrt(1 - (b/a)^2) where a is semi-major, b is semi-minor
                                    # (b/a)^2 is same as (minor_axis/major_axis)^2
                                    eccentricity = np.sqrt(1 - (minor / major) ** 2)
                            except:
                                pass  # Keep 0 if fitting fails

                        sample_features[7] = aspect_ratio
                        sample_features[8] = solidity
                        sample_features[9] = extent
                        sample_features[10] = eccentricity

            except Exception as e:
                # In case of any processing error, we keep the zero vector
                pass

        features_list.append(sample_features)

    # Convert to numpy array
    features_array = np.array(features_list, dtype=np.float64)

    # Save to cache
    print(f"Saving morphometrics to {cache_file}...")
    np.save(cache_file, features_array)

    return features_array
