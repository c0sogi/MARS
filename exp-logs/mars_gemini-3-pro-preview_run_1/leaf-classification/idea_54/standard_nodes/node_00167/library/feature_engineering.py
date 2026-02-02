import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    IMAGES_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    BINARY_THRESHOLD_TYPE,
    BINARY_THRESHOLD_VALUE,
    CONTOUR_APPROX_METHOD,
    GEOMETRIC_FEATURES,
    FLOAT_PRECISION,
    TABULAR_PREFIXES,
    NUM_TABULAR_FEATURES_PER_SET,
)


def extract_geometric_features_single(image_rel_path):
    """
    Extracts the Parsimonious Geometric Basis (6 features) from a single image.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Initialize default features
    features = {k: 0.0 for k in GEOMETRIC_FEATURES}

    if not os.path.exists(full_path):
        return features

    # Read image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Polarity Correction: Invert so leaf is white (foreground)
    _, thresh = cv2.threshold(img, BINARY_THRESHOLD_VALUE, 255, BINARY_THRESHOLD_TYPE)

    # Find Contours (Lossless)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, CONTOUR_APPROX_METHOD)

    if not contours:
        return features

    # Select largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # 1. Equivalent Diameter (Absolute Scale, Linear)
    # D = sqrt(4 * Area / pi)
    # Cite solution_lesson_node_00118: Absolute geometry outperforms invariant.
    features["equivalent_diameter"] = np.sqrt(4 * area / np.pi)

    # 2. Eccentricity (Elongation via Ellipse Fit)
    # Requires at least 5 points to fit an ellipse
    # Cite solution_lesson_node_00142: Prefer bounded descriptors.
    if len(cnt) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            # MA and ma are axis lengths (width, height of enclosing box of ellipse)
            # Sort to get major (a) and minor (b) axes
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                # e = sqrt(1 - (b/a)^2)
                features["eccentricity"] = np.sqrt(1 - (b / a) ** 2)
            else:
                features["eccentricity"] = 0.0
        except:
            features["eccentricity"] = 0.0
    else:
        features["eccentricity"] = 0.0

    # 3. Solidity (Area / ConvexArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["solidity"] = area / hull_area
    else:
        features["solidity"] = 0.0

    # 4. Extent (Area / BoundingArea)
    # Cite solution_lesson_node_00120: Macro-Geometric Feature Completeness.
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["extent"] = area / rect_area
    else:
        features["extent"] = 0.0

    # 5. Aspect Ratio (BoundingWidth / BoundingHeight)
    if h > 0:
        features["aspect_ratio"] = float(w) / h
    else:
        features["aspect_ratio"] = 0.0

    # 6. Roundness (4 * pi * Area / Perimeter^2)
    # Cite solution_lesson_node_00151: Do not prune theoretically redundant non-linear ratios.
    # Cite solution_lesson_node_00162: Avoid Epsilon-Patching (Zero-Imputation).
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        features["roundness"] = (4 * np.pi * area) / (perimeter**2)
    else:
        features["roundness"] = 0.0

    # Sanitize features: Replace inf/nan with 0.0
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features


def process_dataset(metadata_path, is_test=False):
    """
    Loads metadata, extracts geometric features, combines with tabular features,
    and returns processed X and y (or ids).
    """
    print(f"Processing dataset: {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Identify tabular columns
    tabular_cols = []
    for prefix in TABULAR_PREFIXES:
        for i in range(1, NUM_TABULAR_FEATURES_PER_SET + 1):
            col_name = f"{prefix}_{i}"
            if col_name in df.columns:
                tabular_cols.append(col_name)

    # Extract geometric features for each image
    geo_features_list = []
    for _, row in df.iterrows():
        geo_feats = extract_geometric_features_single(row["file_path"])
        geo_features_list.append(geo_feats)

    df_geo = pd.DataFrame(geo_features_list)

    # Combine tabular and geometric features
    # Select only the specific tabular columns + new geometric columns
    X_df = pd.concat([df[tabular_cols], df_geo], axis=1)

    # Enforce Alphanumeric Column Ordering for determinism
    X_df = X_df.reindex(sorted(X_df.columns), axis=1)

    # Convert to high-precision float64
    X = X_df.values.astype(FLOAT_PRECISION)

    if is_test:
        ids = df["id"].values
        return X, ids
    else:
        y = df["species"].values
        return X, y


def load_and_process_data(load_cached_data=True):
    """
    Main function to load train, val, and test data.
    Uses caching to avoid re-processing images if possible.
    """
    # Define cache file paths
    cache_X_train = os.path.join(CACHE_DIR, "X_train.parquet")
    cache_y_train = os.path.join(CACHE_DIR, "y_train.npy")
    cache_X_val = os.path.join(CACHE_DIR, "X_val.parquet")
    cache_y_val = os.path.join(CACHE_DIR, "y_val.npy")
    cache_X_test = os.path.join(CACHE_DIR, "X_test.parquet")
    cache_test_ids = os.path.join(CACHE_DIR, "test_ids.npy")

    files_exist = all(
        os.path.exists(f)
        for f in [
            cache_X_train,
            cache_y_train,
            cache_X_val,
            cache_y_val,
            cache_X_test,
            cache_test_ids,
        ]
    )

    if load_cached_data and files_exist:
        print("Loading data from cache...")
        X_train_df = pd.read_parquet(cache_X_train)
        X_train = X_train_df.values.astype(FLOAT_PRECISION)
        y_train = np.load(cache_y_train, allow_pickle=True)

        X_val_df = pd.read_parquet(cache_X_val)
        X_val = X_val_df.values.astype(FLOAT_PRECISION)
        y_val = np.load(cache_y_val, allow_pickle=True)

        X_test_df = pd.read_parquet(cache_X_test)
        X_test = X_test_df.values.astype(FLOAT_PRECISION)
        test_ids = np.load(cache_test_ids, allow_pickle=True)

        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Cache missing or reload requested. Processing data from scratch...")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Process Train
    X_train, y_train = process_dataset(TRAIN_METADATA_PATH, is_test=False)
    # Process Val
    X_val, y_val = process_dataset(VAL_METADATA_PATH, is_test=False)
    # Process Test
    X_test, test_ids = process_dataset(TEST_METADATA_PATH, is_test=True)

    # Save to cache
    # Convert numpy arrays back to DataFrame for Parquet to preserve column names if needed,
    # but here we just need storage. Using simple numeric columns or just saving values.
    # To be safe and consistent with "read_parquet", we save as DataFrame.
    # We need to reconstruct the DataFrame with sorted columns for saving.
    # However, process_dataset returns numpy array.
    # Let's just save the numpy array as a DataFrame with generic or reconstructed headers
    # to satisfy the requirement "Use parquet (via pandas)".
    # Re-generating column names for saving:
    # Note: process_dataset sorts columns. We need to know the names to save properly?
    # Actually, for the model, we just need the matrix.
    # But to be strictly correct with parquet, we need column names.
    # Let's reconstruct column names logic briefly to save with names.

    # Re-determine column names to save meaningful parquet files
    df_sample = pd.read_csv(TRAIN_METADATA_PATH, nrows=1)
    tabular_cols = []
    for prefix in TABULAR_PREFIXES:
        for i in range(1, NUM_TABULAR_FEATURES_PER_SET + 1):
            col_name = f"{prefix}_{i}"
            if col_name in df_sample.columns:
                tabular_cols.append(col_name)
    all_cols = sorted(tabular_cols + GEOMETRIC_FEATURES)

    pd.DataFrame(X_train, columns=all_cols).to_parquet(cache_X_train)
    np.save(cache_y_train, y_train)

    pd.DataFrame(X_val, columns=all_cols).to_parquet(cache_X_val)
    np.save(cache_y_val, y_val)

    pd.DataFrame(X_test, columns=all_cols).to_parquet(cache_X_test)
    np.save(cache_test_ids, test_ids)

    print("Data processing complete and cached.")

    return X_train, y_train, X_val, y_val, X_test, test_ids
