import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR


def correct_polarity(img):
    """
    Checks the corners of the image to determine if the background is white.
    If the background is white (mean corner intensity > 127), inverts the image
    so the leaf becomes the foreground (white) on a black background.
    """
    h, w = img.shape
    # Sample 10x10 corners
    corners = [
        img[0:10, 0:10],
        img[0:10, w - 10 : w],
        img[h - 10 : h, 0:10],
        img[h - 10 : h, w - 10 : w],
    ]

    # Calculate mean intensity of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # If background is white (high intensity), invert
    if corner_mean > 127:
        return cv2.bitwise_not(img)
    return img


def extract_morphometrics(img):
    """
    Extracts Hu Moments (7 invariants) and Geometric Scalars (Aspect Ratio,
    Solidity, Extent, Eccentricity) from a binary image.
    """
    features = {
        "hu_1": 0.0,
        "hu_2": 0.0,
        "hu_3": 0.0,
        "hu_4": 0.0,
        "hu_5": 0.0,
        "hu_6": 0.0,
        "hu_7": 0.0,
        "aspect_ratio": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
        "eccentricity": 0.0,
    }

    # Ensure binary
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # Hu Moments
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        features[f"hu_{i+1}"] = hu[i]

    # Geometric Scalars
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    # Aspect Ratio
    if h > 0:
        features["aspect_ratio"] = float(w) / h

    # Extent
    if rect_area > 0:
        features["extent"] = area / rect_area

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["solidity"] = area / hull_area

    # Eccentricity
    # Needs at least 5 points to fit ellipse
    if len(cnt) >= 5:
        try:
            (center, (width, height), angle) = cv2.fitEllipse(cnt)
            major_axis = max(width, height)
            minor_axis = min(width, height)

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2) where b is semi-minor, a is semi-major
                # equivalent to sqrt(1 - (minor_axis/major_axis)^2)
                features["eccentricity"] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if ellipse fitting fails
            pass

    return features


def process_images(df, dataset_name, load_cached_data=True):
    """
    Iterates over the dataframe, loads images, corrects polarity, and extracts
    morphometric features. Supports caching to disk.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"morphometrics_{dataset_name}.parquet")

    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Extracting morphometrics for {dataset_name}...")

    results = []
    ids = []

    for idx, row in df.iterrows():
        # Construct full image path
        # metadata 'image_path' is relative to input dir (e.g. "images/1.jpg")
        img_path = os.path.join(INPUT_DIR, row["image_path"])

        # Handle missing files gracefully
        if not os.path.exists(img_path):
            # Return zeroed features
            results.append(extract_morphometrics(np.zeros((100, 100), dtype=np.uint8)))
            ids.append(row["id"])
            continue

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            results.append(extract_morphometrics(np.zeros((100, 100), dtype=np.uint8)))
            ids.append(row["id"])
            continue

        # Process
        img_corrected = correct_polarity(img)
        feats = extract_morphometrics(img_corrected)

        results.append(feats)
        ids.append(row["id"])

    # Create DataFrame
    result_df = pd.DataFrame(results)
    result_df["id"] = ids

    # Save to cache
    result_df.to_parquet(cache_path, index=False)

    return result_df
