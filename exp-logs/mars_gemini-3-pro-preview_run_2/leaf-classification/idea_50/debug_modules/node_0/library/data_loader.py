import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_50"


def extract_morphometrics(image_rel_path):
    """
    Extracts polarity-corrected morphometric features from a leaf image.
    Features: 7 Hu Moments + 4 Geometric Scalars (Aspect Ratio, Solidity, Extent, Eccentricity).

    Args:
        image_rel_path (str): Relative path to the image (e.g., 'images/1.jpg').

    Returns:
        np.ndarray: A 1D array of shape (11,) containing float64 features.
    """
    full_path = os.path.join(INPUT_DIR, image_rel_path)

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fallback for missing or corrupt images: return zeros
        return np.zeros(11, dtype=np.float64)

    # Polarity Correction
    # Check corners to determine background color
    h, w = img.shape
    corners = [
        img[0:10, 0:10],
        img[0:10, w - 10 : w],
        img[h - 10 : h, 0:10],
        img[h - 10 : h, w - 10 : w],
    ]
    # Calculate mean of corners. If > 127, background is white.
    corner_mean = np.mean([np.mean(c) for c in corners])

    if corner_mean > 127:
        # Invert image: Background becomes Black (0), Leaf becomes White (255)
        img = cv2.bitwise_not(img)

    # Threshold to binary to ensure clean shapes
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(11, dtype=np.float64)

    # Assume largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Hu Moments (7 features)
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 2. Geometric Scalars (4 features)
    # Aspect Ratio
    x, y, rect_w, rect_h = cv2.boundingRect(cnt)
    aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0.0

    # Area calculations
    area = moments["m00"]  # Same as cv2.contourArea(cnt)

    # Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Extent
    rect_area = rect_w * rect_h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Eccentricity
    # Needs at least 5 points to fit ellipse
    if len(cnt) >= 5:
        try:
            (center, (MA, ma), angle) = cv2.fitEllipse(cnt)
            # ma is major axis, MA is minor axis (opencv convention can vary, usually (width, height) order)
            # fitEllipse returns (center, (width, height), angle)
            # let's sort axes
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                eccentricity = np.sqrt(1 - (b**2 / a**2))
            else:
                eccentricity = 0.0
        except:
            eccentricity = 0.0
    else:
        eccentricity = 0.0

    features = np.concatenate(
        [hu_moments, [aspect_ratio, solidity, extent, eccentricity]]
    )
    return features.astype(np.float64)


def load_dataset(load_cached_data=True):
    """
    Loads the dataset, performing feature extraction and caching.

    Args:
        load_cached_data (bool): If True, attempts to load from ./working/idea_50/.

    Returns:
        tuple: (
            (X_train_global, X_train_morph, y_train),
            (X_val_global, X_val_morph, y_val),
            (X_test_global, X_test_morph, test_ids, classes)
        )
    """
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # File paths for cache
    cache_files = {
        "X_train_global": os.path.join(CACHE_DIR, "X_train_global.npy"),
        "X_train_morph": os.path.join(CACHE_DIR, "X_train_morph.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val_global": os.path.join(CACHE_DIR, "X_val_global.npy"),
        "X_val_morph": os.path.join(CACHE_DIR, "X_val_morph.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test_global": os.path.join(CACHE_DIR, "X_test_global.npy"),
        "X_test_morph": os.path.join(CACHE_DIR, "X_test_morph.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        X_train_global = np.load(cache_files["X_train_global"])
        X_train_morph = np.load(cache_files["X_train_morph"])
        y_train = np.load(cache_files["y_train"])

        X_val_global = np.load(cache_files["X_val_global"])
        X_val_morph = np.load(cache_files["X_val_morph"])
        y_val = np.load(cache_files["y_val"])

        X_test_global = np.load(cache_files["X_test_global"])
        X_test_morph = np.load(cache_files["X_test_morph"])
        test_ids = np.load(cache_files["test_ids"])
        classes = np.load(cache_files["classes"], allow_pickle=True)

        return (
            (X_train_global, X_train_morph, y_train),
            (X_val_global, X_val_morph, y_val),
            (X_test_global, X_test_morph, test_ids, classes),
        )

    print("Processing data from scratch...")

    # Load metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Identify Global Feature Columns (Margin, Shape, Texture)
    feature_cols = [
        c for c in df_train.columns if c not in ["id", "species", "image_path"]
    ]
    # Ensure consistent order
    feature_cols.sort()

    # --- Process Training Data ---
    print("Processing Training Set...")
    X_train_global = df_train[feature_cols].values.astype(np.float64)
    X_train_morph = np.array(
        [extract_morphometrics(p) for p in df_train["image_path"]], dtype=np.float64
    )

    # --- Process Validation Data ---
    print("Processing Validation Set...")
    X_val_global = df_val[feature_cols].values.astype(np.float64)
    X_val_morph = np.array(
        [extract_morphometrics(p) for p in df_val["image_path"]], dtype=np.float64
    )

    # --- Process Test Data ---
    print("Processing Test Set...")
    X_test_global = df_test[feature_cols].values.astype(np.float64)
    X_test_morph = np.array(
        [extract_morphometrics(p) for p in df_test["image_path"]], dtype=np.float64
    )
    test_ids = df_test["id"].values

    # --- Encode Labels ---
    le = LabelEncoder()
    # Fit on combined train and val species to ensure all classes are covered
    all_species = pd.concat([df_train["species"], df_val["species"]]).unique()
    le.fit(all_species)

    y_train = le.transform(df_train["species"])
    y_val = le.transform(df_val["species"])
    classes = le.classes_

    # --- Save to Cache ---
    print("Saving to cache...")
    np.save(cache_files["X_train_global"], X_train_global)
    np.save(cache_files["X_train_morph"], X_train_morph)
    np.save(cache_files["y_train"], y_train)

    np.save(cache_files["X_val_global"], X_val_global)
    np.save(cache_files["X_val_morph"], X_val_morph)
    np.save(cache_files["y_val"], y_val)

    np.save(cache_files["X_test_global"], X_test_global)
    np.save(cache_files["X_test_morph"], X_test_morph)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["classes"], classes)

    return (
        (X_train_global, X_train_morph, y_train),
        (X_val_global, X_val_morph, y_val),
        (X_test_global, X_test_morph, test_ids, classes),
    )
