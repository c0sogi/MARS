import os
import cv2
import numpy as np
import pandas as pd
from library.config import Config


def extract_hu_moments(contour):
    """
    Extracts the 7 invariant Hu Moments from a contour.

    Args:
        contour: The contour array.

    Returns:
        np.array: A 7-element array of Hu moments (float64).
    """
    try:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # We return raw Hu moments; scaling/transforming is handled by the pipeline's PowerTransformer
        return hu_moments.astype(Config.FLOAT_TYPE)
    except Exception:
        return np.zeros(7, dtype=Config.FLOAT_TYPE)


def extract_geometric_properties(contour):
    """
    Extracts geometric scalars: Aspect Ratio, Solidity, Extent, Eccentricity.

    Args:
        contour: The contour array.

    Returns:
        dict: A dictionary containing the 4 scalar features.
    """
    props = {"aspect_ratio": 0.0, "solidity": 0.0, "extent": 0.0, "eccentricity": 0.0}

    if contour is None or len(contour) == 0:
        return props

    try:
        # Area and Bounding Rect
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Aspect Ratio
        if h > 0:
            props["aspect_ratio"] = float(w) / h

        # Extent
        rect_area = w * h
        if rect_area > 0:
            props["extent"] = area / rect_area

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            props["solidity"] = area / hull_area

        # Eccentricity
        # Requires at least 5 points to fit an ellipse
        if len(contour) >= 5:
            (center, (ma, MA), angle) = cv2.fitEllipse(contour)
            # ma = minor axis, MA = major axis (opencv returns them this way sometimes, or swapped)
            # We sort them to be sure
            a = MA / 2.0
            b = ma / 2.0
            if a < b:
                a, b = b, a

            if a > 0:
                # e = sqrt(1 - (b^2/a^2))
                props["eccentricity"] = np.sqrt(1 - (b**2 / a**2))

    except Exception:
        # Fallback to zeros on failure
        pass

    return props


def process_single_image(image_path):
    """
    Loads an image and extracts all macro features.

    Args:
        image_path: Full path to the image file.

    Returns:
        np.array: Concatenated feature vector (Hu moments + Geometric scalars).
    """
    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        # Return zero vector of length 7 (Hu) + 4 (Geo) = 11
        return np.zeros(11, dtype=Config.FLOAT_TYPE)

    # Invert image: Dataset is black leaf (0) on white background (255).
    # Contours are found on white objects against black background.
    # So we invert: Leaf becomes 255, Background becomes 0.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(11, dtype=Config.FLOAT_TYPE)

    # Assume the largest contour is the leaf
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract features
    hu = extract_hu_moments(largest_contour)
    geo = extract_geometric_properties(largest_contour)

    geo_vec = np.array(
        [geo["aspect_ratio"], geo["solidity"], geo["extent"], geo["eccentricity"]],
        dtype=Config.FLOAT_TYPE,
    )

    return np.concatenate([hu, geo_vec])


def generate_macro_features(metadata_df, cache_filename, load_cached_data=True):
    """
    Generates or loads Macro-Resolution features for a given metadata DataFrame.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id' and 'image_path'.
        cache_filename (str): Name of the parquet file to cache results.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and extracted features.
    """
    cache_path = Config.get_cache_path(cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached macro features from {cache_path}...")
        try:
            df_features = pd.read_parquet(cache_path)
            # Verify length matches
            if len(df_features) == len(metadata_df):
                return df_features
            else:
                print("Cached file length mismatch. Recomputing...")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing macro features for {len(metadata_df)} images...")

    feature_list = []
    ids = metadata_df["id"].values
    image_paths = metadata_df["image_path"].values

    # Define column names
    hu_cols = [f"hu_{i+1}" for i in range(7)]
    geo_cols = ["aspect_ratio", "solidity", "extent", "eccentricity"]
    all_cols = ["id"] + hu_cols + geo_cols

    for i, rel_path in enumerate(image_paths):
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        if os.path.exists(full_path):
            feats = process_single_image(full_path)
        else:
            # Should not happen given metadata verification, but handle safely
            feats = np.zeros(11, dtype=Config.FLOAT_TYPE)

        # Prepend ID
        row = np.concatenate([[ids[i]], feats])
        feature_list.append(row)

    # Create DataFrame
    df_features = pd.DataFrame(feature_list, columns=all_cols)

    # Ensure ID is int (it might have been cast to float during concat)
    df_features["id"] = df_features["id"].astype(int)

    # 3. Save to cache
    print(f"Saving macro features to {cache_path}...")
    # Ensure directory exists (Config.setup() usually handles this, but being safe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path, index=False)

    return df_features
