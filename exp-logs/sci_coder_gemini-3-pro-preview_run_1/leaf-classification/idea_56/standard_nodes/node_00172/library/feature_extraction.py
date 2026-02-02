import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    FLOAT_PRECISION,
    SEED,
)
from library.utils import set_seed

# Set global seed for reproducibility
set_seed(SEED)


class GeometricFeatureExtractor:
    """
    Implements the Hybrid Geometric Fusion strategy.
    Extracts both boundary-fitted (shape) and integral (structure) features
    from binary leaf images.
    """

    def __init__(self):
        pass

    def process_image(self, image_path):
        """
        Loads and preprocesses the image.
        Applies cv2.THRESH_BINARY_INV to ensure the leaf is foreground (white)
        against a background (black), suitable for contour finding and distance transforms.
        """
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        # Dataset description: "binary black leaves against white backgrounds"
        # We want leaf=255 (white), background=0 (black)
        # THRESH_BINARY_INV: src > thresh ? 0 : maxval
        # Pixel 0 (leaf) -> 0 > 127 False -> 255
        # Pixel 255 (bg) -> 255 > 127 True -> 0
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        return img, binary

    def extract_all(self, image_path):
        """
        Pipeline to extract parsimonious geometric features for a single image.
        Implements the 'Parsimonious Geometric Fusion' strategy.
        Cite solution_lesson_node_00140 (Parsimony), solution_lesson_node_00150 (Redundancy).
        """
        img, binary = self.process_image(image_path)

        # Default zero features
        default_feats = {
            "geo_solidity": 0.0,
            "geo_extent": 0.0,
            "geo_aspect_ratio": 0.0,
            "geo_eccentricity": 0.0,
            "geo_roundness": 0.0,
            "geo_equivalent_diameter": 0.0,
        }

        if binary is None:
            return default_feats

        # Find contours
        # RETR_EXTERNAL: Only outer contours
        # CHAIN_APPROX_NONE: Store all contour points (lossless)
        # Cite solution_lesson_node_00149
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            # Assume the largest contour is the leaf
            c = max(contours, key=cv2.contourArea)
        else:
            return default_feats

        # --- Raw Measurements (Internal Use Only) ---
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        # Convex Hull
        hull = cv2.convexHull(c)
        convex_area = cv2.contourArea(hull)

        # Bounding Rect
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h

        # Ellipse Fitting
        major_axis = 0.0
        minor_axis = 0.0
        if len(c) >= 5:
            try:
                (cx, cy), (d1, d2), angle = cv2.fitEllipse(c)
                axes = sorted([d1, d2])
                minor_axis = axes[0]
                major_axis = axes[1]
            except Exception:
                pass

        # --- Feature Computation with Zero-Imputation ---
        # Cite solution_lesson_node_00162

        # 1. Solidity (Area / ConvexArea)
        solidity = 0.0
        if convex_area > 1e-9:
            solidity = area / convex_area

        # 2. Extent (Area / BoundingRectArea)
        extent = 0.0
        if rect_area > 1e-9:
            extent = area / rect_area

        # 3. Aspect Ratio (Width / Height)
        aspect_ratio = 0.0
        if h > 1e-9:
            aspect_ratio = float(w) / float(h)

        # 4. Eccentricity (sqrt(1 - (b/a)^2))
        eccentricity = 0.0
        if major_axis > 1e-9:
            ratio_sq = (minor_axis / major_axis) ** 2
            if ratio_sq <= 1.0:
                eccentricity = np.sqrt(1.0 - ratio_sq)

        # 5. Roundness (4 * pi * Area / Perimeter^2)
        roundness = 0.0
        if perimeter > 1e-9:
            roundness = (4 * np.pi * area) / (perimeter**2)

        # 6. Equivalent Diameter (sqrt(4 * Area / pi))
        # Cite solution_lesson_node_00118 (Importance), solution_lesson_node_00150 (Avoid Area redundancy)
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        return {
            "geo_solidity": float(solidity),
            "geo_extent": float(extent),
            "geo_aspect_ratio": float(aspect_ratio),
            "geo_eccentricity": float(eccentricity),
            "geo_roundness": float(roundness),
            "geo_equivalent_diameter": float(equivalent_diameter),
        }


def load_data_with_features(dataset_type="train", load_cached_data=True, limit=None):
    """
    Loads data, extracts geometric features, and merges them with tabular features.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.
        limit (int, optional): If set, only process/load the first N rows (for debugging).

    Returns:
        X (pd.DataFrame): Feature matrix (tabular + geometric).
        y (np.array): Target labels (if available).
        ids (np.array): Image IDs.
    """
    # 1. Determine paths
    if dataset_type == "train":
        meta_path = TRAIN_METADATA_PATH
        cache_filename = "train_features_geo.parquet"
    elif dataset_type == "val":
        meta_path = VAL_METADATA_PATH
        cache_filename = "val_features_geo.parquet"
    elif dataset_type == "test":
        meta_path = TEST_METADATA_PATH
        cache_filename = "test_features_geo.parquet"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Handle limit in cache filename to avoid collisions
    if limit is not None:
        cache_filename = cache_filename.replace(".parquet", f"_limit_{limit}.parquet")

    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # 2. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"[{dataset_type}] Loading cached features from {cache_path}...")
        df = pd.read_parquet(cache_path)

        # Extract components
        ids = df["id"].values

        if "species" in df.columns:
            y = df["species"].values
            X = df.drop(columns=["id", "species"])
        else:
            y = None
            X = df.drop(columns=["id"])

        return X, y, ids

    # 3. Process from Scratch
    print(f"[{dataset_type}] Generating features (Hybrid Geometric Fusion)...")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)

    if limit is not None:
        print(f"[{dataset_type}] Limiting to first {limit} samples.")
        df_meta = df_meta.head(limit)

    extractor = GeometricFeatureExtractor()
    geo_features_list = []

    # Iterate through images
    for idx, row in df_meta.iterrows():
        # Metadata contains relative path (e.g., 'images/123.jpg')
        # We need full path: ./input/images/123.jpg
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Extract features
        feats = extractor.extract_all(full_path)
        geo_features_list.append(feats)

    # Convert new features to DataFrame
    df_geo = pd.DataFrame(geo_features_list)

    # 4. Merge and Format
    # Concatenate original metadata (which has tabular features) with new geo features
    # df_meta has: id, species (optional), margin_*, shape_*, texture_*, file_path
    df_combined = pd.concat(
        [df_meta.reset_index(drop=True), df_geo.reset_index(drop=True)], axis=1
    )

    # Drop file_path as it is not a feature
    if "file_path" in df_combined.columns:
        df_combined = df_combined.drop(columns=["file_path"])

    # Enforce Alphanumeric Column Ordering
    # Identify feature columns (exclude id, species)
    non_feature_cols = ["id", "species"]
    feature_cols = [c for c in df_combined.columns if c not in non_feature_cols]
    feature_cols.sort()  # Deterministic sort

    # Reorder DataFrame
    final_cols = [
        c for c in non_feature_cols if c in df_combined.columns
    ] + feature_cols
    df_final = df_combined[final_cols]

    # Enforce float64 precision for features
    for col in feature_cols:
        df_final[col] = df_final[col].astype(FLOAT_PRECISION)

    # 5. Save to Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_final.to_parquet(cache_path, index=False)
    print(f"[{dataset_type}] Features saved to {cache_path}")

    # 6. Return
    ids = df_final["id"].values
    if "species" in df_final.columns:
        y = df_final["species"].values
        X = df_final.drop(columns=["id", "species"])
    else:
        y = None
        X = df_final.drop(columns=["id"])

    return X, y, ids
