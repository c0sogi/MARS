import os
import cv2
import numpy as np
import pandas as pd
from library import config


def extract_morphometrics(image_path):
    """
    Extracts Hu Moments and Geometric Scalars from a binary leaf image.
    Returns a numpy array of shape (11,) containing:
    [hu1, hu2, hu3, hu4, hu5, hu6, hu7, aspect_ratio, solidity, extent, eccentricity]
    """
    # Initialize default feature vector (11 features)
    features = np.zeros(11, dtype=config.FLOAT_PRECISION)

    if not os.path.exists(image_path):
        return features

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Threshold: Leaf is black, background is white.
    # Invert so leaf is white (255) for contour detection.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    features[0:7] = hu_moments

    # 2. Geometric Scalars

    # Aspect Ratio and Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = float(area) / rect_area if rect_area > 0 else 0.0

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    # Needs at least 5 points to fit ellipse
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            # fitEllipse returns ((x,y), (width, height), angle)
            (center, (d1, d2), angle) = cv2.fitEllipse(cnt)
            # Sort axes to identify major and minor
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2) where b is semi-minor, a is semi-major
                # (b/a)^2 is equivalent to (minor_axis/major_axis)^2
                term = 1.0 - (minor_axis / major_axis) ** 2
                if term > 0:
                    eccentricity = np.sqrt(term)
        except Exception:
            eccentricity = 0.0

    features[7] = aspect_ratio
    features[8] = solidity
    features[9] = extent
    features[10] = eccentricity

    return features


def _process_dataframe_images(df, cache_path, load_cached_data):
    """
    Internal helper to process images listed in a dataframe or load from cache.
    """
    # Check cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached macro features from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting macro features for {len(df)} images...")

    # Pre-allocate array
    n_samples = len(df)
    macro_features = np.zeros((n_samples, 11), dtype=config.FLOAT_PRECISION)

    image_paths = df["image_path"].values

    for i, rel_path in enumerate(image_paths):
        full_path = os.path.join(config.INPUT_DIR, rel_path)
        macro_features[i] = extract_morphometrics(full_path)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, macro_features)
    print(f"Saved macro features to {cache_path}")

    return macro_features


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, extracts/loads macro features, and combines them with provided micro features.

    Returns:
        (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids)

        Where X is a DataFrame containing both micro (192) and macro (11) features.
        y is a Series/Array of species labels.
        ids is a Series/Array of image IDs.
    """

    # 1. Load Metadata
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # 2. Extract/Load Macro Features
    macro_train = _process_dataframe_images(
        df_train, config.CACHE_TRAIN_MACRO, load_cached_data
    )
    macro_val = _process_dataframe_images(
        df_val, config.CACHE_VAL_MACRO, load_cached_data
    )
    macro_test = _process_dataframe_images(
        df_test, config.CACHE_TEST_MACRO, load_cached_data
    )

    # Define Macro Column Names
    macro_cols = [
        "macro_hu1",
        "macro_hu2",
        "macro_hu3",
        "macro_hu4",
        "macro_hu5",
        "macro_hu6",
        "macro_hu7",
        "macro_aspect_ratio",
        "macro_solidity",
        "macro_extent",
        "macro_eccentricity",
    ]

    # 3. Identify Micro Features
    # Identify columns that start with margin, shape, or texture
    feature_cols = [
        c
        for c in df_train.columns
        if any(c.startswith(p) for p in config.MICRO_FEATURE_PREFIXES)
    ]

    # 4. Combine Features
    def combine(df, macro_array):
        # Micro features
        X_micro = df[feature_cols].values.astype(config.FLOAT_PRECISION)

        # Concatenate [Micro, Macro]
        X_combined = np.hstack([X_micro, macro_array])

        all_cols = feature_cols + macro_cols
        X_df = pd.DataFrame(X_combined, columns=all_cols, index=df.index)
        return X_df

    X_train = combine(df_train, macro_train)
    X_val = combine(df_val, macro_val)
    X_test = combine(df_test, macro_test)

    # 5. Extract Targets and IDs
    y_train = df_train["species"].values
    y_val = df_val["species"].values

    train_ids = df_train["id"].values
    val_ids = df_val["id"].values
    test_ids = df_test["id"].values

    return (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids)
