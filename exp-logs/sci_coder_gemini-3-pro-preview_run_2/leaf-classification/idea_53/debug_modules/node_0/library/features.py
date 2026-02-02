import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    IMAGES_BASE_DIR,
    WORKING_DIR,
    POLARITY_THRESHOLD,
    ID_COL,
    IMAGE_PATH_COL,
)


def check_polarity(img, threshold=POLARITY_THRESHOLD):
    """
    Determines if the image background is bright (white) and needs inversion.
    Assumes binary image (0 or 255).
    Checks 4 corners (5x5 patches).
    Returns True if the background appears to be white (indicating inversion is needed).
    """
    h, w = img.shape
    # Sample corners
    corners = [
        img[0:5, 0:5],
        img[0:5, w - 5 : w],
        img[h - 5 : h, 0:5],
        img[h - 5 : h, w - 5 : w],
    ]

    # Calculate mean intensity of corners
    corner_mean = np.mean([np.mean(c) for c in corners])

    # Normalize to 0-1
    corner_mean_norm = corner_mean / 255.0

    # If corners are bright (white background), we need to invert
    # so that the object (leaf) becomes bright (foreground)
    return corner_mean_norm > threshold


def extract_morphometrics_single(img_path):
    """
    Extracts Hu Moments and Geometric Scalars from a single image.
    Returns a dictionary of features.
    """
    full_path = os.path.join(IMAGES_BASE_DIR, img_path)

    # Initialize features with NaNs
    features = {f"hu_{i+1}": np.nan for i in range(7)}
    features.update(
        {
            "aspect_ratio": np.nan,
            "solidity": np.nan,
            "extent": np.nan,
            "eccentricity": np.nan,
        }
    )

    if not os.path.exists(full_path):
        return features

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Binarize (ensure strict 0/255) using Otsu's thresholding
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Polarity Check and Correction
    # Ensure leaf is White (255) and Background is Black (0)
    if check_polarity(img_bin):
        img_bin = cv2.bitwise_not(img_bin)

    # Find Contours
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # 1. Hu Moments
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, hu in enumerate(hu_moments):
        features[f"hu_{i+1}"] = hu

    # 2. Geometric Scalars

    # Bounding Rect -> Aspect Ratio, Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    if h > 0:
        features["aspect_ratio"] = float(w) / h
    else:
        features["aspect_ratio"] = 0.0

    if rect_area > 0:
        features["extent"] = area / rect_area
    else:
        features["extent"] = 0.0

    # Convex Hull -> Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["solidity"] = area / hull_area
    else:
        features["solidity"] = 0.0

    # Fit Ellipse -> Eccentricity
    # Requires at least 5 points
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are diameters (major/minor axes lengths)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2)
                # (b/a)^2 = (minor/major)^2
                features["eccentricity"] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            else:
                features["eccentricity"] = 0.0
        except Exception:
            features["eccentricity"] = 0.0
    else:
        features["eccentricity"] = 0.0

    return features


def process_images(df, load_cached_data=True, cache_name="morphometrics.parquet"):
    """
    Batch processes images listed in the dataframe to extract morphometric features.
    Implements caching to avoid re-computation.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'image_path'.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_name (str): Filename for the cache in WORKING_DIR.

    Returns:
        pd.DataFrame: DataFrame with 'id' and extracted features.
    """
    cache_path = os.path.join(WORKING_DIR, cache_name)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Load from cache if requested and available
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        try:
            cached_df = pd.read_parquet(cache_path)

            # Check if all requested IDs are in cache
            required_ids = set(df[ID_COL])
            cached_ids = set(cached_df[ID_COL])

            if required_ids.issubset(cached_ids):
                # Return only the requested rows
                return cached_df[cached_df[ID_COL].isin(required_ids)].reset_index(
                    drop=True
                )
            else:
                print("Cache found but incomplete. Merging and updating...")
                # We will process missing IDs below
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
            cached_df = pd.DataFrame()
    else:
        cached_df = pd.DataFrame()

    # 2. Identify missing IDs
    if not cached_df.empty:
        existing_ids = set(cached_df[ID_COL])
        df_to_process = df[~df[ID_COL].isin(existing_ids)].copy()
    else:
        df_to_process = df.copy()

    # 3. Compute features for missing IDs
    if not df_to_process.empty:
        print(f"Extracting morphometric features for {len(df_to_process)} images...")
        results = []

        for idx, row in df_to_process.iterrows():
            img_id = row[ID_COL]
            img_path = row[IMAGE_PATH_COL]

            feats = extract_morphometrics_single(img_path)
            feats[ID_COL] = img_id
            results.append(feats)

        new_results_df = pd.DataFrame(results)
        # Ensure ID is correct type
        new_results_df[ID_COL] = new_results_df[ID_COL].astype(df[ID_COL].dtype)

        # Combine with existing cache
        if not cached_df.empty:
            final_df = pd.concat([cached_df, new_results_df], ignore_index=True)
        else:
            final_df = new_results_df

        # Update cache
        try:
            final_df.to_parquet(cache_path)
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_path}: {e}")

    else:
        final_df = cached_df

    # Return only the requested rows in the correct order (optional, but good practice)
    # We filter by the input df IDs
    return final_df[final_df[ID_COL].isin(df[ID_COL])].reset_index(drop=True)
