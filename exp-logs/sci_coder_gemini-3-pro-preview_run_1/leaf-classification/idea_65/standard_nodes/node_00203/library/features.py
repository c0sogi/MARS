import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from library import config, utils


def extract_geometric_features(image_path: str) -> dict:
    """
    Extracts the 'Golden 5' robust geometric descriptors from a leaf image.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        dict: Dictionary containing the 5 geometric features.
              Returns zeros if image load fails or no contour is found.
    """
    # Initialize with zeros in case of failure
    features = {k: 0.0 for k in config.GEOMETRIC_FEATURES}

    if not os.path.exists(image_path):
        return features

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # Apply Thresholding
    # Explicitly invert to ensure leaf is foreground (white)
    _, thresh = cv2.threshold(
        img, config.BINARY_THRESHOLD_VALUE, 255, config.BINARY_THRESHOLD_TYPE
    )

    # Find Contours
    # Use CHAIN_APPROX_NONE for lossless boundary
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, config.CONTOUR_MODE)

    if not contours:
        return features

    # Implicit Denoising: Select largest contour by area
    cnt = max(contours, key=cv2.contourArea)

    # 1. Area (Absolute Scale)
    area = cv2.contourArea(cnt)
    features["Area"] = float(area)

    if area == 0:
        return features

    # Compute Bounding Rect for Extent and Aspect Ratio
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h

    # 2. Aspect Ratio (Orientation)
    # Avoid division by zero
    if h > 0:
        features["Aspect_Ratio"] = float(w) / h

    # 3. Extent (Rectangularity)
    if rect_area > 0:
        features["Extent"] = area / rect_area

    # 4. Solidity (Roughness)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["Solidity"] = area / hull_area

    # 5. Eccentricity (Elongation)
    # Requires at least 5 points to fit ellipse
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            major_axis = max(axis1, axis2)
            minor_axis = min(axis1, axis2)
            if major_axis > 0:
                # e = sqrt(1 - (b/a)^2)
                features["Eccentricity"] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            # Fallback if fitEllipse fails
            pass

    return features


def _process_subset(df: pd.DataFrame, is_test: bool = False):
    """
    Internal helper to process a dataframe subset:
    1. Extract tabular features.
    2. Extract geometric features.
    3. Concatenate.
    4. Extract targets (if not test).
    """
    # 1. Identify Tabular Features
    # Filter columns that start with margin, shape, or texture
    tabular_cols = [
        c for c in df.columns if c.startswith(("margin", "shape", "texture"))
    ]
    # Sort to ensure deterministic order
    tabular_cols.sort()

    # Extract Tabular Data (ensure float64)
    X_tabular = df[tabular_cols].values.astype(config.FLOAT_PRECISION)

    # 2. Extract Geometric Features
    geo_features_list = []
    ids = []

    # Iterate and extract
    for idx, row in df.iterrows():
        # Construct full image path
        # Metadata contains relative path 'images/123.jpg'
        full_path = os.path.join(config.INPUT_DIR, row["file_path"])

        feats = extract_geometric_features(full_path)

        # Map to list in specific order defined in config
        feat_vector = [feats[key] for key in config.GEOMETRIC_FEATURES]
        geo_features_list.append(feat_vector)
        ids.append(row["id"])

    X_geo = np.array(geo_features_list, dtype=config.FLOAT_PRECISION)

    # 3. Concatenate (Tabular + Geometric)
    X_combined = np.hstack([X_tabular, X_geo])

    # 4. Handle Targets
    y = None
    if not is_test:
        y = df["species"].values

    return X_combined, y, np.array(ids)


def get_data(load_cached_data: bool = True):
    """
    Main data loading function.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids, classes, scaler_params)
    """
    # Generate Cache Paths
    config_hash = utils.get_config_hash()
    cache_dir = config.WORKING_DIR

    paths = {
        "X_train": os.path.join(cache_dir, f"X_train_{config_hash}.npy"),
        "y_train": os.path.join(cache_dir, f"y_train_{config_hash}.npy"),
        "X_val": os.path.join(cache_dir, f"X_val_{config_hash}.npy"),
        "y_val": os.path.join(cache_dir, f"y_val_{config_hash}.npy"),
        "X_test": os.path.join(cache_dir, f"X_test_{config_hash}.npy"),
        "test_ids": os.path.join(cache_dir, f"test_ids_{config_hash}.npy"),
        "classes": os.path.join(cache_dir, f"classes_{config_hash}.npy"),
    }

    # Check Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            print(f"Loading cached data from {cache_dir} (Hash: {config_hash})...")
            return (
                np.load(paths["X_train"]),
                np.load(paths["y_train"]),
                np.load(paths["X_val"]),
                np.load(paths["y_val"]),
                np.load(paths["X_test"]),
                np.load(paths["test_ids"]),
                np.load(paths["classes"], allow_pickle=True),
            )
        else:
            print("Cache miss or partial cache. Recomputing features...")

    # Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    if config.DEBUG:
        print(f"DEBUG MODE: Subsampling {config.DEBUG_SAMPLE_SIZE} rows.")
        # Filter by class first to ensure train/val consistency (Cite debug_lesson_1)
        # Select first 5 classes found in training data to ensure overlap
        debug_classes = df_train["species"].unique()[:5]

        df_train = df_train[df_train["species"].isin(debug_classes)]
        df_val = df_val[df_val["species"].isin(debug_classes)]

        df_train = df_train.iloc[: config.DEBUG_SAMPLE_SIZE]
        df_val = df_val.iloc[: config.DEBUG_SAMPLE_SIZE]
        df_test = df_test.iloc[: config.DEBUG_SAMPLE_SIZE]

    # Process Raw Data
    print("Extracting features for Training set...")
    X_train_raw, y_train_raw, _ = _process_subset(df_train, is_test=False)

    print("Extracting features for Validation set...")
    X_val_raw, y_val_raw, _ = _process_subset(df_val, is_test=False)

    print("Extracting features for Test set...")
    X_test_raw, _, test_ids = _process_subset(df_test, is_test=True)

    # Encode Targets
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    # Pipeline Sanitization & Transformation
    # 1. Variance Threshold (Sanitization)
    print("Applying Variance Sanitization...")
    vt = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    X_train_vt = vt.fit_transform(X_train_raw)
    X_val_vt = vt.transform(X_val_raw)
    X_test_vt = vt.transform(X_test_raw)

    print(
        f"Features remaining after sanitization: {X_train_vt.shape[1]}/{X_train_raw.shape[1]}"
    )

    # 2. Yeo-Johnson Transformation (Stabilization)
    # standardize=False because we apply StandardScaler explicitly next
    print("Applying Yeo-Johnson Transformation...")
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    X_train_pt = pt.fit_transform(X_train_vt)
    X_val_pt = pt.transform(X_val_vt)
    X_test_pt = pt.transform(X_test_vt)

    # 3. Standard Scaling
    print("Applying Standard Scaling...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_pt)
    X_val = scaler.transform(X_val_pt)
    X_test = scaler.transform(X_test_pt)

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["test_ids"], test_ids)
    np.save(paths["classes"], classes)

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes
