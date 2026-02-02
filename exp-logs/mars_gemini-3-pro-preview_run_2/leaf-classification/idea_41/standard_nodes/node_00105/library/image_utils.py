import os
import cv2
import numpy as np
import pandas as pd

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_41"
RANDOM_SEED = 42


def check_polarity(img):
    """
    Checks if the background is white (high intensity). If so, inverts the image
    so the object is white (foreground) and background is black.
    Assumes img is a grayscale numpy array.
    """
    h, w = img.shape
    # Sample 4 corners
    corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
    # If average corner intensity is high (> 127), assume white background
    if np.mean(corners) > 127:
        img = cv2.bitwise_not(img)
    return img


def extract_single_image_features(image_path):
    """
    Extracts morphometric features from a single image.
    Returns a dictionary of features.
    """
    full_path = os.path.join(INPUT_DIR, image_path)

    # Initialize default values (zeros)
    features = {f"hu_{i}": 0.0 for i in range(7)}
    features.update(
        {"aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0, "eccentricity": 0.0}
    )

    if not os.path.exists(full_path):
        return features

    # Read image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return features

    # 1. Polarity Check
    img = check_polarity(img)

    # 2. Threshold to ensure binary (0 or 255)
    # Using Otsu's binarization or simple threshold since data is binary
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return features

    # 4. Get largest contour (assume it is the leaf)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area == 0:
        return features

    # 5. Hu Moments
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        features[f"hu_{i}"] = hu[i]

    # 6. Geometric Scalars

    # Bounding Rect -> Aspect Ratio, Extent
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    if rect_area > 0:
        features["extent"] = area / rect_area
    if h > 0:
        features["aspect_ratio"] = float(w) / h

    # Convex Hull -> Solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        features["solidity"] = area / hull_area

    # Fit Ellipse -> Eccentricity
    # Fit ellipse requires at least 5 points
    if len(cnt) >= 5:
        try:
            # fitEllipse returns (center, (MA, ma), angle)
            # where (MA, ma) are the lengths of the major and minor axes
            (x_e, y_e), (d1, d2), angle = cv2.fitEllipse(cnt)

            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)

            if major_axis > 0:
                # Eccentricity e = sqrt(1 - (b/a)^2) where a is semi-major, b is semi-minor
                # Ratio b/a is same as minor_axis/major_axis
                features["eccentricity"] = np.sqrt(1 - (minor_axis / major_axis) ** 2)
        except Exception:
            features["eccentricity"] = 0.0

    return features


def process_images(df, load_cached_data=True, cache_name="data"):
    """
    Process a dataframe of images to extract morphometric features.
    Handles caching to ./working/idea_41/
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}_morphometrics.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Extracting morphometrics for {cache_name}...")

    # List to store feature dicts
    data_list = []

    # Iterate over dataframe
    # Assuming df has 'image_path' and 'id'
    image_paths = df["image_path"].values
    ids = df["id"].values

    for i, path in enumerate(image_paths):
        feats = extract_single_image_features(path)
        feats["id"] = ids[i]  # Keep ID for merging
        data_list.append(feats)

    # Create DataFrame
    result_df = pd.DataFrame(data_list)

    # Ensure ID is integer
    result_df["id"] = result_df["id"].astype(int)

    # Save to cache
    print(f"Saving morphometrics to {cache_path}")
    result_df.to_parquet(cache_path, index=False)

    return result_df


def get_morphometric_features(metadata_dir="./metadata", load_cached_data=True):
    """
    Main entry point to get features for train, val, and test.
    Returns tuple of DataFrames: (train_feats, val_feats, test_feats)
    """
    # Load metadata
    train_meta = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_meta = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_meta = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Process
    train_feats = process_images(
        train_meta, load_cached_data=load_cached_data, cache_name="train"
    )
    val_feats = process_images(
        val_meta, load_cached_data=load_cached_data, cache_name="val"
    )
    test_feats = process_images(
        test_meta, load_cached_data=load_cached_data, cache_name="test"
    )

    return train_feats, val_feats, test_feats
