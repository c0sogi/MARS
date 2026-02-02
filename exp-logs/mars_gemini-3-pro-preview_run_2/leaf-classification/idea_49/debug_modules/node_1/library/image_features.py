import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Ensure cache directory exists
CACHE_DIR = "./working/idea_49/"
os.makedirs(CACHE_DIR, exist_ok=True)


def check_polarity(img: np.ndarray) -> np.ndarray:
    """
    Checks if the image background is white (high intensity) and inverts it
    if necessary so that the leaf is foreground (white) and background is black.

    Strategy: Check the mean intensity of the four corners. If > 127 (for uint8),
    assume white background.
    """
    if img is None:
        return img

    h, w = img.shape
    # Define corners (10x10 patches)
    corners = [
        img[0:10, 0:10],
        img[0:10, w - 10 : w],
        img[h - 10 : h, 0:10],
        img[h - 10 : h, w - 10 : w],
    ]

    # Calculate mean of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If background is bright (white), invert
    if corner_mean > 127:
        return cv2.bitwise_not(img)

    return img


def extract_hu_moments(img: np.ndarray) -> list:
    """
    Calculates the 7 Hu Moments invariants for the image.
    Returns a list of 7 floats.
    """
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Log transform is often used to handle scale, but we return raw values
    # to let the downstream scaler handle it, as per 'Polynomial Physical Experts' design.
    # We cast to float64 explicitly.
    return [float(x) for x in hu_moments]


def extract_geometric_props(img: np.ndarray) -> dict:
    """
    Calculates scalar geometric properties: Aspect Ratio, Solidity, Extent, Eccentricity.
    """
    props = {"aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0, "eccentricity": 0.0}

    # Find contours
    # Use RETR_EXTERNAL to get the outer boundary
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return props

    # Assume the leaf is the largest contour
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return props

    # 1. Aspect Ratio and Extent (Bounding Rect)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    if w > 0 and h > 0:
        props["aspect_ratio"] = float(w) / h
        props["extent"] = float(area) / rect_area

    # 2. Solidity (Convex Hull)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        props["solidity"] = float(area) / hull_area

    # 3. Eccentricity (Ellipse Fit)
    # fitEllipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                # e = sqrt(1 - (b^2/a^2))
                props["eccentricity"] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if fitEllipse fails
            props["eccentricity"] = 0.0

    return props


def process_images(
    image_paths: list, cache_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Batch processes a list of image paths to extract morphometric features.

    Args:
        image_paths: List of relative file paths (e.g., ['images/1.jpg', ...])
        cache_name: Identifier for the cache file (e.g., 'train_morph', 'test_morph')
        load_cached_data: If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: DataFrame containing extracted features (float64).
    """
    set_seed(42)

    cache_file = os.path.join(CACHE_DIR, f"{cache_name}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached image features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print(f"Processing {len(image_paths)} images for {cache_name}...")

    # 2. Process Images
    data_list = []
    input_dir = "./input"

    for rel_path in image_paths:
        full_path = os.path.join(input_dir, rel_path)

        # Initialize row with defaults
        row_data = {}

        if os.path.exists(full_path):
            # Read as grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Preprocessing
                img = check_polarity(img)

                # Feature Extraction
                hu = extract_hu_moments(img)
                geo = extract_geometric_props(img)

                # Populate row
                for i, val in enumerate(hu):
                    row_data[f"hu_moment_{i+1}"] = val

                row_data.update(geo)
            else:
                # Image read failed
                row_data = {f"hu_moment_{i+1}": 0.0 for i in range(7)}
                row_data.update(
                    {
                        "aspect_ratio": 0.0,
                        "solidity": 0.0,
                        "extent": 0.0,
                        "eccentricity": 0.0,
                    }
                )
        else:
            # File missing
            row_data = {f"hu_moment_{i+1}": 0.0 for i in range(7)}
            row_data.update(
                {
                    "aspect_ratio": 0.0,
                    "solidity": 0.0,
                    "extent": 0.0,
                    "eccentricity": 0.0,
                }
            )

        data_list.append(row_data)

    # 3. Create DataFrame
    df = pd.DataFrame(data_list)

    # Ensure float64 precision
    df = df.astype(np.float64)

    # 4. Save Cache
    print(f"Saving image features to {cache_file}...")
    df.to_parquet(cache_file)

    return df
