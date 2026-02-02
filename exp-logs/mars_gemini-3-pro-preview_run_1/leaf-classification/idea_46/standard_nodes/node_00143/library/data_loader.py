import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    TABULAR_FEATURES,
    GEOMETRIC_PRIMITIVES,
    RATIO_FEATURES,
    ALL_FEATURES,
    FLOAT_PRECISION,
    ID_COL,
    TARGET_COL,
    FILE_PATH_COL,
)

# Epsilon for numerical stability in ratio calculations
EPSILON = 1e-9


def extract_geometric_features(image_path):
    """
    Extracts geometric primitives from a binary leaf image.
    """
    # Initialize default values
    defaults = {k: 0.0 for k in GEOMETRIC_PRIMITIVES}
    defaults["Bounding_Width"] = 0.0
    defaults["Bounding_Height"] = 0.0

    if not os.path.exists(image_path):
        return defaults

    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return defaults

    # Invert image: Dataset is black leaf on white background.
    # We want white leaf (255) on black background (0) for contour detection.
    img = cv2.bitwise_not(img)

    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return defaults

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Basic Measures
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # 2. Convex Hull
    hull = cv2.convexHull(cnt)
    convex_area = cv2.contourArea(hull)
    convex_perimeter = cv2.arcLength(hull, True)

    # 3. Bounding Rect (for Extent and Aspect Ratio)
    x, y, w, h = cv2.boundingRect(cnt)

    # 4. Ellipse Fit (for Major/Minor Axis)
    # Requires at least 5 points
    if len(cnt) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except:
            major_axis = max(w, h)
            minor_axis = min(w, h)
    else:
        major_axis = max(w, h)
        minor_axis = min(w, h)

    # 5. Equivalent Diameter
    equiv_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    return {
        "Area": float(area),
        "Perimeter": float(perimeter),
        "Convex_Area": float(convex_area),
        "Convex_Perimeter": float(convex_perimeter),
        "Major_Axis_Length": float(major_axis),
        "Minor_Axis_Length": float(minor_axis),
        "Equivalent_Diameter": float(equiv_diameter),
        "Bounding_Width": float(w),
        "Bounding_Height": float(h),
    }


def compute_ratio_projections(df):
    """
    Computes dimensionless shape descriptors from geometric primitives.
    """
    # Solidity: Area / Convex_Area (Cite solution_lesson_node_00120)
    df["Solidity"] = df["Area"] / (df["Convex_Area"] + EPSILON)

    # Extent: Area / (Bounding_Width * Bounding_Height) (Cite solution_lesson_node_00120)
    df["Extent"] = df["Area"] / (
        (df["Bounding_Width"] * df["Bounding_Height"]) + EPSILON
    )

    # Aspect_Ratio: Bounding_Width / Bounding_Height (Cite solution_lesson_node_00120)
    df["Aspect_Ratio"] = df["Bounding_Width"] / (df["Bounding_Height"] + EPSILON)

    # Roundness (formerly Form_Factor): 4 * pi * Area / Perimeter^2
    # Prioritize scale-invariant ratios (Cite solution_lesson_node_00119)
    df["Roundness"] = (4 * np.pi * df["Area"]) / (df["Perimeter"] ** 2 + EPSILON)

    # Eccentricity: sqrt(1 - (Minor / Major)^2)
    # Prefer bounded descriptors [0, 1] over unbounded Elongation (Cite solution_lesson_node_00142)
    axis_ratio = df["Minor_Axis_Length"] / (df["Major_Axis_Length"] + EPSILON)
    # Clip inside sqrt to avoid numerical noise producing negative values
    df["Eccentricity"] = np.sqrt(np.clip(1.0 - axis_ratio**2, 0.0, 1.0))

    return df


def process_subset(metadata_path, subset_name):
    """
    Loads metadata, extracts features, computes ratios, and merges with tabular data.
    """
    print(f"Processing {subset_name} data from {metadata_path}...")

    # Load metadata
    df_meta = pd.read_csv(metadata_path)

    # Extract Geometric Features
    geo_features_list = []
    for idx, row in df_meta.iterrows():
        # Construct full path: INPUT_DIR + relative path from metadata
        full_path = os.path.join(INPUT_DIR, row[FILE_PATH_COL])
        feats = extract_geometric_features(full_path)
        geo_features_list.append(feats)

    df_geo = pd.DataFrame(geo_features_list)

    # Compute Ratios
    df_geo = compute_ratio_projections(df_geo)

    # Merge with original tabular features
    # We concatenate horizontally. Ensure indices align (they should as we iterated)
    df_combined = pd.concat([df_meta, df_geo], axis=1)

    # Select and Sort Columns according to configuration
    X = df_combined[ALL_FEATURES].copy()

    # Enforce Precision
    X = X.astype(FLOAT_PRECISION)

    # Extract IDs
    ids = df_combined[ID_COL].values

    # Extract Targets if available
    y = None
    if TARGET_COL in df_combined.columns:
        y = df_combined[TARGET_COL].values

    return X, y, ids


def load_and_process_data(load_cached_data=True):
    """
    Main data loading function with caching mechanism.
    Returns: (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids)
    """
    # Cache file paths
    cache_files = {
        "X_train": os.path.join(WORKING_DIR, "X_train.parquet"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "train_ids": os.path.join(WORKING_DIR, "train_ids.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.parquet"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "val_ids": os.path.join(WORKING_DIR, "val_ids.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.parquet"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        X_train = pd.read_parquet(cache_files["X_train"])
        y_train = np.load(cache_files["y_train"], allow_pickle=True)
        train_ids = np.load(cache_files["train_ids"], allow_pickle=True)

        X_val = pd.read_parquet(cache_files["X_val"])
        y_val = np.load(cache_files["y_val"], allow_pickle=True)
        val_ids = np.load(cache_files["val_ids"], allow_pickle=True)

        X_test = pd.read_parquet(cache_files["X_test"])
        test_ids = np.load(cache_files["test_ids"], allow_pickle=True)

        return (
            (X_train, y_train, train_ids),
            (X_val, y_val, val_ids),
            (X_test, test_ids),
        )

    print("Cache missing or reload requested. Processing data from scratch...")

    # Define metadata paths
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    # Process Datasets
    X_train, y_train, train_ids = process_subset(train_meta_path, "Train")
    X_val, y_val, val_ids = process_subset(val_meta_path, "Validation")
    X_test, _, test_ids = process_subset(test_meta_path, "Test")

    # Save to Cache
    print("Saving processed data to cache...")
    X_train.to_parquet(cache_files["X_train"])
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["train_ids"], train_ids)

    X_val.to_parquet(cache_files["X_val"])
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["val_ids"], val_ids)

    X_test.to_parquet(cache_files["X_test"])
    np.save(cache_files["test_ids"], test_ids)

    return (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids)
