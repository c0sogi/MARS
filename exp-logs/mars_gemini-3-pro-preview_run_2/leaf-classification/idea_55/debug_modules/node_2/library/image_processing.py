import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    POLARITY_THRESHOLD,
    FLOAT_PRECISION,
    RANDOM_SEED,
)
from library.utils import set_seed


def process_single_image(image_path):
    """
    Reads a binary leaf image, applies polarity correction, and extracts
    morphometric features (Hu Moments + Geometric Scalars).

    Args:
        image_path (str): Full path to the image file.

    Returns:
        np.ndarray: A 1D array of extracted features (float64).
                    Returns a zero vector if image is invalid.
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 12 features: 7 Hu moments + 5 Geometric Scalars
    # If image load fails, return zeros
    if img is None:
        return np.zeros(12, dtype=FLOAT_PRECISION)

    # Polarity Correction
    # Check the mean intensity of the 4 corners (5x5 patches)
    h, w = img.shape
    corner_size = 5
    # Ensure image is large enough for corner check
    if h > corner_size * 2 and w > corner_size * 2:
        corners = [
            img[0:corner_size, 0:corner_size],
            img[0:corner_size, w - corner_size : w],
            img[h - corner_size : h, 0:corner_size],
            img[h - corner_size : h, w - corner_size : w],
        ]
        avg_corner_intensity = np.mean([c.mean() for c in corners])

        # If background is white (high intensity), invert so leaf is foreground (white)
        # Image is 0-255, so we normalize to 0-1 for comparison with POLARITY_THRESHOLD
        if (avg_corner_intensity / 255.0) > POLARITY_THRESHOLD:
            img = cv2.bitwise_not(img)

    # Binarize to ensure strict 0/255 values
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(12, dtype=FLOAT_PRECISION)

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)

    # --- Feature Extraction ---

    # 1. Hu Moments (7 features)
    # These are translation, scale, and rotation invariant
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. Geometric Scalars (5 features)
    area = moments["m00"]

    # If area is effectively zero, return zeros to avoid division by zero
    if area <= 1e-6:
        return np.zeros(12, dtype=FLOAT_PRECISION)

    # Aspect Ratio
    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0

    # Extent (Ratio of contour area to bounding rectangle area)
    rect_area = w_rect * h_rect
    extent = area / rect_area if rect_area > 0 else 0.0

    # Solidity (Ratio of contour area to its convex hull area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Eccentricity (Measure of how much the shape deviates from a circle)
    # Requires fitting an ellipse, needs at least 5 points
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (x_e, y_e), (MA, ma), angle = cv2.fitEllipse(cnt)
            # fitEllipse returns (MajorAxis, MinorAxis) or vice versa depending on orientation
            # We sort them so 'a' is semi-major axis, 'b' is semi-minor
            axes = sorted([MA, ma], reverse=True)
            major_axis = axes[0]
            minor_axis = axes[1]

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2)
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if ellipse fit fails
            eccentricity = 0.0

    # Compactness (Isoperimetric Quotient variant: 4*pi*A / P^2)
    perimeter = cv2.arcLength(cnt, True)
    compactness = 0.0
    if perimeter > 0:
        compactness = (4 * np.pi * area) / (perimeter**2)

    # Combine all features
    scalars = np.array(
        [aspect_ratio, extent, solidity, eccentricity, compactness],
        dtype=FLOAT_PRECISION,
    )
    features = np.concatenate([hu_moments, scalars])

    return features.astype(FLOAT_PRECISION)


def extract_morphometrics(dataset_name, load_cached_data=True):
    """
    Extracts morphometric features for a specified dataset split.
    Manages caching to avoid re-processing images.

    Args:
        dataset_name (str): One of 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, 12).
    """
    set_seed(RANDOM_SEED)

    # Validate dataset name
    if dataset_name not in ["train", "val", "test"]:
        raise ValueError(
            f"Invalid dataset_name: {dataset_name}. Must be 'train', 'val', or 'test'."
        )

    # Define cache path
    cache_file = os.path.join(WORKING_DIR, f"morphometrics_{dataset_name}.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_file}...")
        try:
            features = np.load(cache_file)
            return features
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Extracting morphometrics for {dataset_name} set...")

    # Load metadata
    metadata_path = os.path.join(METADATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    feature_list = []

    # Iterate over images
    # Using simple loop as per instruction to avoid progress bars
    for _, row in df.iterrows():
        # Construct full image path
        # Metadata 'image_path' is relative to INPUT_DIR (e.g., "images/12.jpg")
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            feats = process_single_image(full_path)
        else:
            # If file is missing (unlikely given EDA), return zeros
            feats = np.zeros(12, dtype=FLOAT_PRECISION)

        feature_list.append(feats)

    # Convert to numpy array
    features = np.array(feature_list, dtype=FLOAT_PRECISION)

    # 3. Save to Cache
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, features)
        print(f"Saved morphometrics to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_file}: {e}")

    return features
