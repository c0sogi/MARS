import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define constants
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/optimized_features"


def extract_geometric_features(image_paths):
    """
    Extracts geometric features (Aspect Ratio, Extent, Solidity, Eccentricity)
    from binary leaf images.

    Args:
        image_paths (array-like): Relative paths to images.

    Returns:
        np.ndarray: Matrix of shape (n_samples, 4) containing extracted features.
    """
    features = []
    for rel_path in image_paths:
        full_path = os.path.join("./input", rel_path)

        # Default feature vector if image load fails
        default_feat = [0.0, 0.0, 0.0, 0.0]

        if not os.path.exists(full_path):
            features.append(default_feat)
            continue

        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            features.append(default_feat)
            continue

        # Threshold: Leaf is black on white, invert to get white leaf on black
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            features.append(default_feat)
            continue

        # Assume largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        # 1. Aspect Ratio
        aspect_ratio = float(w) / h if h > 0 else 0.0

        # 2. Extent
        extent = float(area) / rect_area if rect_area > 0 else 0.0

        # 3. Solidity
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        # 4. Eccentricity
        eccentricity = 0.0
        if len(cnt) >= 5:
            try:
                (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
                major = max(d1, d2)
                minor = min(d1, d2)
                if major > 0:
                    eccentricity = np.sqrt(1 - (minor / major) ** 2)
            except:
                pass

        features.append([aspect_ratio, extent, solidity, eccentricity])

    return np.array(features, dtype=np.float32)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, concatenating train and validation sets for maximum sample size.
    Extracts numerical features (margin, shape, texture) AND geometric features, then encodes targets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        tuple: (X_train, y_train, X_test, test_ids, label_encoder)
    """
    # Define cache file paths
    cache_files = {
        "X_train": os.path.join(CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(CACHE_DIR, "X_test.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # Check if we should and can load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(path) for path in cache_files.values())
        if all_exist:
            print(f"Loading cached data from {CACHE_DIR}...")
            X_train = np.load(cache_files["X_train"])
            y_train = np.load(cache_files["y_train"])
            X_test = np.load(cache_files["X_test"])
            test_ids = np.load(cache_files["test_ids"])
            classes = np.load(cache_files["classes"], allow_pickle=True)

            # Reconstruct LabelEncoder
            le = LabelEncoder()
            le.classes_ = classes

            return X_train, y_train, X_test, test_ids, le
        else:
            print("Cache miss or partial cache found. Re-processing data...")

    # Load metadata
    print("Loading metadata CSVs...")
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    df_train_part = pd.read_csv(train_path)
    df_val_part = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Concatenate train and val sets as per strategy (maximize sample size)
    df_train = pd.concat([df_train_part, df_val_part], axis=0, ignore_index=True)

    # Identify feature columns (margin, shape, texture)
    # We exclude id, species, image_path
    feature_cols = [
        c for c in df_train.columns if c not in ["id", "species", "image_path"]
    ]

    # Sort columns to ensure consistency
    feature_cols.sort()

    print(f"Extracting {len(feature_cols)} tabular features...")

    # Extract tabular features
    X_train_tab = df_train[feature_cols].values.astype(np.float32)
    X_test_tab = df_test[feature_cols].values.astype(np.float32)

    # Extract geometric features
    print("Extracting geometric features from images...")
    X_train_geo = extract_geometric_features(df_train["image_path"].values)
    X_test_geo = extract_geometric_features(df_test["image_path"].values)

    # Concatenate features
    X_train = np.hstack([X_train_tab, X_train_geo])
    X_test = np.hstack([X_test_tab, X_test_geo])

    # Extract IDs
    test_ids = df_test["id"].values

    # Encode Targets
    print("Encoding targets...")
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["species"])

    # Save to cache
    print(f"Saving processed data to {CACHE_DIR}...")
    os.makedirs(CACHE_DIR, exist_ok=True)

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"], X_test)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], le.classes_)

    return X_train, y_train, X_test, test_ids, le
