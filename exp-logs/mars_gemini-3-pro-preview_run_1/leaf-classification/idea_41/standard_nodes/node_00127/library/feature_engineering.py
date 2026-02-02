import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    GEOMETRIC_FEATURES,
    PRECISION_TYPE,
    ID_COL,
    FILE_PATH_COL,
)
from library.utils import validate_precision


def extract_geometric_properties(image_path):
    """
    Extracts geometric features from a binary leaf image using OpenCV.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: A dictionary containing the calculated geometric features.
              Returns a dict of zeros if the image cannot be processed.
    """
    # Initialize default features with 0.0
    features = {k: 0.0 for k in GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Read image in grayscale
    # The dataset description says: "binary black leaves against white backgrounds"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Thresholding
    # We want the leaf to be the foreground (white/255) and background black (0).
    # Since inputs are black leaves on white, we use THRESH_BINARY_INV.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # Assume the largest contour corresponds to the leaf
    cnt = max(contours, key=cv2.contourArea)

    # --- 1. Scale Features ---
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    hull = cv2.convexHull(cnt)
    convex_perimeter = float(cv2.arcLength(hull, True))
    convex_area = float(cv2.contourArea(hull))

    # Axis Lengths
    # Try fitEllipse (requires >= 5 points), fallback to minAreaRect
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except Exception:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            major_axis = max(w, h)
            minor_axis = min(w, h)
    else:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        major_axis = max(w, h)
        minor_axis = min(w, h)

    # --- 2. Morphology Features ---
    # Solidity: Area / Convex_Area
    solidity = area / convex_area if convex_area > 0 else 0.0

    # Eccentricity: sqrt(1 - (b/a)^2) for ellipse
    if major_axis > 0:
        ratio_sq = (minor_axis / major_axis) ** 2
        # Clamp value to >= 0 before sqrt to avoid numerical errors
        eccentricity = np.sqrt(max(0.0, 1.0 - ratio_sq))
    else:
        eccentricity = 0.0

    # Min_Area_Aspect_Ratio: Width / Height of rotated bounding box
    # We use min_dim / max_dim to be rotation invariant and bounded [0, 1]
    rect_rot = cv2.minAreaRect(cnt)
    w_rot, h_rot = rect_rot[1]
    if max(w_rot, h_rot) > 0:
        min_area_aspect_ratio = min(w_rot, h_rot) / max(w_rot, h_rot)
    else:
        min_area_aspect_ratio = 0.0

    # Extent: Area / Bounding_Rect_Area
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # --- 3. Topology Features ---
    # Convexity: Convex_Perimeter / Perimeter
    convexity = convex_perimeter / perimeter if perimeter > 0 else 0.0

    # Equivalent Diameter: sqrt(4 * Area / pi)
    # Scales linearly with size, unlike Area (quadratic).
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    # Roundness: (4 * pi * Area) / (Perimeter^2)
    # Measures compactness (1.0 for circle).
    roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

    # Update features dictionary
    features["Area"] = area
    features["Perimeter"] = perimeter
    features["Convex_Perimeter"] = convex_perimeter
    features["Major_Axis_Length"] = major_axis
    features["Minor_Axis_Length"] = minor_axis
    features["Solidity"] = solidity
    features["Eccentricity"] = eccentricity
    features["Min_Area_Aspect_Ratio"] = min_area_aspect_ratio
    features["Extent"] = extent
    features["Convexity"] = convexity
    features["Equivalent_Diameter"] = equiv_diameter
    features["Roundness"] = roundness

    return features


def augment_dataset(df, dataset_name="Dataset"):
    """
    Augments the provided dataframe with geometric features extracted from images.

    Args:
        df (pd.DataFrame): The input dataframe containing 'file_path'.
        dataset_name (str): Name for logging purposes.

    Returns:
        pd.DataFrame: The augmented dataframe.
    """
    print(f"Augmenting {dataset_name} with geometric features...")

    # Pre-allocate lists for new features
    new_data = {k: [] for k in GEOMETRIC_FEATURES}

    # Iterate and extract
    for idx, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(INPUT_DIR, row[FILE_PATH_COL])

        # Extract
        feats = extract_geometric_properties(full_path)

        # Append
        for k, v in feats.items():
            new_data[k].append(v)

    # Add new columns to DataFrame
    for k in GEOMETRIC_FEATURES:
        df[k] = np.array(new_data[k], dtype=PRECISION_TYPE)

    # Validate precision for one of the new columns
    validate_precision(
        df[GEOMETRIC_FEATURES[0]].values, f"{dataset_name} - {GEOMETRIC_FEATURES[0]}"
    )

    print(f"Finished augmentation for {dataset_name}. Shape: {df.shape}")
    return df


def process_data(load_cached_data=True):
    """
    Main function to load, augment, and return the training, validation, and test datasets.
    Handles caching to disk to save time on subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (df_train, df_val, df_test) containing augmented features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_train = os.path.join(CACHE_DIR, "train_augmented.parquet")
    cache_val = os.path.join(CACHE_DIR, "val_augmented.parquet")
    cache_test = os.path.join(CACHE_DIR, "test_augmented.parquet")

    # Check if cache exists and is requested
    if (
        load_cached_data
        and os.path.exists(cache_train)
        and os.path.exists(cache_val)
        and os.path.exists(cache_test)
    ):
        print("Loading augmented datasets from cache...")
        df_train = pd.read_parquet(cache_train)
        df_val = pd.read_parquet(cache_val)
        df_test = pd.read_parquet(cache_test)

        # Validate precision after loading
        # Parquet usually preserves types, but good to check
        if len(GEOMETRIC_FEATURES) > 0:
            validate_precision(
                df_train[GEOMETRIC_FEATURES[0]].values, "Cached Train Geometric"
            )

        return df_train, df_val, df_test

    print("Cache not found or disabled. Processing from scratch...")

    # Load metadata
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Augment datasets
    df_train = augment_dataset(df_train, "Train")
    df_val = augment_dataset(df_val, "Val")
    df_test = augment_dataset(df_test, "Test")

    # Save to cache
    print("Saving augmented datasets to cache...")
    df_train.to_parquet(cache_train, index=False)
    df_val.to_parquet(cache_val, index=False)
    df_test.to_parquet(cache_test, index=False)

    return df_train, df_val, df_test
