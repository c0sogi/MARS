import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    IMAGES_REL_DIR,
    MORPHO_COLS,
    POLARITY_CHECK_THRESHOLD,
    FLOAT_PRECISION,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
)
from library.utils import set_seed


def _process_single_image(rel_path):
    """
    Reads an image, corrects polarity, and extracts morphometric features.

    Returns:
        np.array: A 1D array of shape (11,) containing 7 Hu moments + 4 scalars.
                  Returns zeros if image processing fails.
    """
    full_path = os.path.join(INPUT_DIR, rel_path)

    # Initialize result vector (7 Hu + 4 Scalars)
    # Order: hu_1..hu_7, aspect_ratio, solidity, extent, eccentricity
    result = np.zeros(11, dtype=FLOAT_PRECISION)

    if not os.path.exists(full_path):
        return result

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return result

    # --- Polarity Correction ---
    # Check corners to determine background color
    h, w = img.shape
    corner_size = 5
    corners = []

    # Top-left
    corners.append(img[0:corner_size, 0:corner_size])
    # Top-right
    corners.append(img[0:corner_size, w - corner_size : w])
    # Bottom-left
    corners.append(img[h - corner_size : h, 0:corner_size])
    # Bottom-right
    corners.append(img[h - corner_size : h, w - corner_size : w])

    # Calculate mean intensity of corners (normalized 0-1)
    corner_mean = np.mean([np.mean(c) for c in corners]) / 255.0

    # If background is bright (white), invert image so leaf is foreground (white on black)
    if corner_mean > POLARITY_CHECK_THRESHOLD:
        img = cv2.bitwise_not(img)

    # --- Contour Extraction ---
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return result

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return result

    # --- Hu Moments ---
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Assign Hu moments (indices 0-6)
    result[0:7] = hu_moments

    # --- Geometric Scalars ---
    # 1. Aspect Ratio
    x, y, w_rect, h_rect = cv2.boundingRect(cnt)
    aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0

    # 2. Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0

    # 3. Extent
    rect_area = w_rect * h_rect
    extent = float(area) / rect_area if rect_area > 0 else 0.0

    # 4. Eccentricity
    # Needs at least 5 points to fit ellipse
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are lengths of axes (major/minor depends on magnitude)
            major = max(axis1, axis2)
            minor = min(axis1, axis2)
            if major > 0:
                eccentricity = np.sqrt(1 - (minor / major) ** 2)
        except:
            eccentricity = 0.0

    # Assign scalars (indices 7-10)
    result[7] = aspect_ratio
    result[8] = solidity
    result[9] = extent
    result[10] = eccentricity

    return result


def _extract_features_for_dataframe(df):
    """
    Iterates over a dataframe, processes images, and appends morphometric features.
    """
    # Pre-allocate numpy array for features
    n_samples = len(df)
    n_features = len(MORPHO_COLS)
    features_matrix = np.zeros((n_samples, n_features), dtype=FLOAT_PRECISION)

    image_paths = df["image_path"].values

    # Process images (could be parallelized, but keeping simple linear for stability inside docker)
    for i, rel_path in enumerate(image_paths):
        features_matrix[i] = _process_single_image(rel_path)

    # Create DataFrame
    features_df = pd.DataFrame(features_matrix, columns=MORPHO_COLS, index=df.index)

    # Concatenate with original dataframe
    # We drop image_path as it is no longer needed for modeling
    df_out = pd.concat([df, features_df], axis=1)

    return df_out


def get_data(load_cached_data=True):
    """
    Main entry point to load data.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed parquet files
                                 from the working directory.

    Returns:
        tuple: (df_train, df_val, df_test)
               DataFrames containing original features + extracted morphometrics.
    """
    set_seed()

    # Define cache paths
    cache_train_path = os.path.join(WORKING_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(WORKING_DIR, "val_processed.parquet")
    cache_test_path = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Check if cache exists and is requested
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):

            print("Loading cached data from:", WORKING_DIR)
            df_train = pd.read_parquet(cache_train_path)
            df_val = pd.read_parquet(cache_val_path)
            df_test = pd.read_parquet(cache_test_path)

            # Ensure proper types
            return df_train, df_val, df_test
        else:
            print("Cache not found or incomplete. Computing features from scratch...")
    else:
        print("Forcing feature re-computation...")

    # Load Metadata
    print("Loading metadata...")
    df_train_meta = pd.read_csv(TRAIN_DATA_PATH)
    df_val_meta = pd.read_csv(VAL_DATA_PATH)
    df_test_meta = pd.read_csv(TEST_DATA_PATH)

    # Extract Features
    print("Extracting morphometrics for Training set...")
    df_train = _extract_features_for_dataframe(df_train_meta)

    print("Extracting morphometrics for Validation set...")
    df_val = _extract_features_for_dataframe(df_val_meta)

    print("Extracting morphometrics for Test set...")
    df_test = _extract_features_for_dataframe(df_test_meta)

    # Ensure output directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Save to Cache
    print("Saving processed data to cache...")
    df_train.to_parquet(cache_train_path, index=False)
    df_val.to_parquet(cache_val_path, index=False)
    df_test.to_parquet(cache_test_path, index=False)

    return df_train, df_val, df_test
