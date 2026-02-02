import os
import cv2
import numpy as np
import pandas as pd

from library import config


def extract_morphological_props(image_path):
    """
    Extracts morphological scalar descriptors from a binary leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing 'aspect_ratio', 'solidity', 'extent', 'eccentricity'.
              Returns zeros if image processing fails.
    """
    # Default values in case of failure or empty image
    default_props = {
        "aspect_ratio": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
        "eccentricity": 0.0,
    }

    if not os.path.exists(image_path):
        return default_props

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return default_props

    # The dataset description says "binary black leaves against white backgrounds".
    # regionprops expects the object of interest to be non-zero (foreground).
    # Therefore, we invert the image: Leaf becomes White (255), Background becomes Black (0).
    img_inverted = cv2.bitwise_not(img)

    # Threshold to ensure strict binary (0 or 1/255)
    _, binary = cv2.threshold(img_inverted, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    # RETR_EXTERNAL retrieves only the extreme outer contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return default_props

    # If multiple regions exist (noise), take the largest one by area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return default_props

    # 1. Aspect Ratio & Extent
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # 2. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # 3. Eccentricity
    # Eccentricity is the ratio of the focal distance to the major axis length.
    # e = sqrt(1 - (b^2 / a^2)) where a is semi-major axis, b is semi-minor axis.
    # cv2.fitEllipse returns ((x,y), (MA, ma), angle).
    # Note: fitEllipse requires at least 5 points.
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are the lengths of the axes (diameters)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)

            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
        except Exception:
            # Fallback if ellipse fitting fails
            eccentricity = 0.0

    return {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
    }


def augment_dataframe(df, load_cached_data=True, cache_name="augmented_data"):
    """
    Augments the input DataFrame with morphological features extracted from images.
    Implements caching using Parquet files.

    Args:
        df (pd.DataFrame): Input dataframe containing 'image_path' column.
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_name (str): Identifier for the cache file (e.g., 'train', 'test').

    Returns:
        pd.DataFrame: DataFrame with added morphological feature columns.
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(config.WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached augmented data from {cache_path}...")
            cached_df = pd.read_parquet(cache_path)
            # Verify length matches (simple integrity check)
            if len(cached_df) == len(df):
                return cached_df
            else:
                print("Cached data length mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Extracting morphological features for {cache_name}...")

    features_list = []

    # Iterate over the dataframe
    # Using df.to_dict('records') is often faster than iterrows for iteration
    records = df.to_dict("records")

    for row in records:
        # Construct full path. Metadata 'image_path' is relative (e.g., 'images/1.jpg')
        # config.INPUT_DIR is './input'
        full_path = os.path.join(config.INPUT_DIR, row["image_path"])

        props = extract_morphological_props(full_path)
        features_list.append(props)

    # Create a DataFrame from the new features
    features_df = pd.DataFrame(features_list)

    # Concatenate with original dataframe (reset index to ensure alignment)
    # We assume the order is preserved which is true for list construction
    df_augmented = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    # 3. Save to cache
    try:
        print(f"Saving augmented data to {cache_path}...")
        df_augmented.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return df_augmented
