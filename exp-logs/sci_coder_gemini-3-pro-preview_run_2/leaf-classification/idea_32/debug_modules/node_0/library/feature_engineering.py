import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_32"


def extract_macro_features(df, input_dir):
    """
    Extracts Hu Moments and geometric scalars from binary leaf images.
    Returns a float64 numpy array of shape (N, 12).
    """
    features = []

    for _, row in df.iterrows():
        # Construct full path
        img_path = os.path.join(input_dir, row["image_path"])

        # Read image as grayscale (binary images are effectively grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Safety check
        if img is None:
            features.append(np.zeros(12))
            continue

        # Threshold to ensure strict binary (0 or 255)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Calculate Moments
        moments = cv2.moments(thresh)

        # Hu Moments (7 invariant features)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log scale transform to handle wide ranges, preserving sign
        # Use eps to avoid log(0)
        eps = 1e-7
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + eps)

        # Contour for geometric features
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Assume largest contour is the leaf
            cnt = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h

            # Aspect Ratio
            aspect_ratio = float(w) / h if h > 0 else 0

            # Extent (Ratio of object area to bounding rectangle area)
            extent = float(area) / rect_area if rect_area > 0 else 0

            # Solidity (Ratio of object area to convex hull area)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Eccentricity (Ratio of focal distance to major axis length)
            if len(cnt) >= 5:
                # fitEllipse requires at least 5 points
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                a = ma / 2
                b = MA / 2
                if a > 0:
                    # eccentricity = sqrt(1 - (b/a)^2) where a is semi-major axis
                    eccentricity = np.sqrt(1 - (min(a, b) / max(a, b)) ** 2)
                else:
                    eccentricity = 0
            else:
                eccentricity = 0

            # Compactness (Isoperimetric quotient proxy)
            compactness = (perimeter**2) / area if area > 0 else 0

        else:
            aspect_ratio, extent, solidity, eccentricity, compactness = 0, 0, 0, 0, 0

        # Combine features: 7 Hu moments + 5 geometric scalars
        row_features = np.concatenate(
            [hu_moments, [aspect_ratio, extent, solidity, eccentricity, compactness]]
        )
        features.append(row_features)

    return np.array(features, dtype=np.float64)


def get_genus_labels(species_series):
    """
    Extracts genus labels from species strings (e.g., 'Acer_Capillipes' -> 'Acer').
    """
    return species_series.apply(lambda x: x.split("_")[0])


def load_and_process_data(load_cached_data=True):
    """
    Loads data, extracts features, preprocesses, and caches results.

    Returns:
        dict: A dictionary containing processed feature arrays and encoders.
    """
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "X_train_global": os.path.join(CACHE_DIR, "X_train_global.npy"),
        "X_val_global": os.path.join(CACHE_DIR, "X_val_global.npy"),
        "X_test_global": os.path.join(CACHE_DIR, "X_test_global.npy"),
        "X_train_macro": os.path.join(CACHE_DIR, "X_train_macro.npy"),
        "X_val_macro": os.path.join(CACHE_DIR, "X_val_macro.npy"),
        "X_test_macro": os.path.join(CACHE_DIR, "X_test_macro.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "y_train_genus": os.path.join(CACHE_DIR, "y_train_genus.npy"),
        "y_val_genus": os.path.join(CACHE_DIR, "y_val_genus.npy"),
        "classes": os.path.join(CACHE_DIR, "classes.npy"),
        "genus_classes": os.path.join(CACHE_DIR, "genus_classes.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}

        # Reconstruct Encoders
        species_le = LabelEncoder()
        species_le.classes_ = data["classes"]
        genus_le = LabelEncoder()
        genus_le.classes_ = data["genus_classes"]

        data["species_encoder"] = species_le
        data["genus_encoder"] = genus_le
        return data

    print("Processing data from scratch...")

    # Load Metadata
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 1. Global Features (Provided)
    # Identify feature columns (exclude metadata)
    feature_cols = [
        c for c in df_train.columns if c not in ["id", "species", "image_path"]
    ]

    X_train_global = df_train[feature_cols].values.astype(np.float64)
    X_val_global = df_val[feature_cols].values.astype(np.float64)
    X_test_global = df_test[feature_cols].values.astype(np.float64)

    # 2. Macro Features (Extracted from Images)
    print("Extracting Macro features...")
    X_train_macro = extract_macro_features(df_train, INPUT_DIR)
    X_val_macro = extract_macro_features(df_val, INPUT_DIR)
    X_test_macro = extract_macro_features(df_test, INPUT_DIR)

    # 3. Targets
    # Species Encoding
    species_le = LabelEncoder()
    y_train = species_le.fit_transform(df_train["species"])
    y_val = species_le.transform(df_val["species"])

    # Genus Encoding
    train_genus = get_genus_labels(df_train["species"])
    val_genus = get_genus_labels(df_val["species"])

    genus_le = LabelEncoder()
    y_train_genus = genus_le.fit_transform(train_genus)
    y_val_genus = genus_le.transform(val_genus)

    test_ids = df_test["id"].values

    # 4. Preprocessing (PowerTransformer)
    # Gaussianization is critical for LDA/QDA
    print("Applying PowerTransformer (Yeo-Johnson)...")

    # Global View
    pt_global = PowerTransformer(method="yeo-johnson")
    X_train_global = pt_global.fit_transform(X_train_global)
    X_val_global = pt_global.transform(X_val_global)
    X_test_global = pt_global.transform(X_test_global)

    # Macro View
    pt_macro = PowerTransformer(method="yeo-johnson")
    X_train_macro = pt_macro.fit_transform(X_train_macro)
    X_val_macro = pt_macro.transform(X_val_macro)
    X_test_macro = pt_macro.transform(X_test_macro)

    # 5. Save to Cache
    np.save(cache_files["X_train_global"], X_train_global)
    np.save(cache_files["X_val_global"], X_val_global)
    np.save(cache_files["X_test_global"], X_test_global)
    np.save(cache_files["X_train_macro"], X_train_macro)
    np.save(cache_files["X_val_macro"], X_val_macro)
    np.save(cache_files["X_test_macro"], X_test_macro)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["y_train_genus"], y_train_genus)
    np.save(cache_files["y_val_genus"], y_val_genus)
    np.save(cache_files["classes"], species_le.classes_)
    np.save(cache_files["genus_classes"], genus_le.classes_)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "X_train_global": X_train_global,
        "X_val_global": X_val_global,
        "X_test_global": X_test_global,
        "X_train_macro": X_train_macro,
        "X_val_macro": X_val_macro,
        "X_test_macro": X_test_macro,
        "y_train": y_train,
        "y_val": y_val,
        "y_train_genus": y_train_genus,
        "y_val_genus": y_val_genus,
        "species_encoder": species_le,
        "genus_encoder": genus_le,
        "test_ids": test_ids,
        "classes": species_le.classes_,
        "genus_classes": genus_le.classes_,
    }
