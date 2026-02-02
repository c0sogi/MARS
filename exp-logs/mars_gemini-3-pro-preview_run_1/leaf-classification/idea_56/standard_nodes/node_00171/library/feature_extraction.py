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

    def extract_boundary_features(self, contour):
        """
        Extracts features based on the external boundary of the leaf.
        """
        if contour is None or len(contour) == 0:
            return {
                "geo_area": 0.0,
                "geo_perimeter": 0.0,
                "geo_convex_perimeter": 0.0,
                "geo_major_axis": 0.0,
                "geo_minor_axis": 0.0,
            }

        # Basic moments
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Convex Hull
        hull = cv2.convexHull(contour)
        convex_perimeter = cv2.arcLength(hull, True)

        # Ellipse Fitting (Requires at least 5 points)
        major_axis = 0.0
        minor_axis = 0.0
        if len(contour) >= 5:
            try:
                # fitEllipse returns (center, (MA, ma), angle)
                # MA, ma are lengths of the axes (diameters)
                (x, y), (d1, d2), angle = cv2.fitEllipse(contour)
                axes = sorted([d1, d2])
                minor_axis = axes[0]
                major_axis = axes[1]
            except Exception:
                # Fallback if fit fails (e.g., collinear points)
                pass

        return {
            "geo_area": float(area),
            "geo_perimeter": float(perimeter),
            "geo_convex_perimeter": float(convex_perimeter),
            "geo_major_axis": float(major_axis),
            "geo_minor_axis": float(minor_axis),
        }

    def extract_integral_features(self, binary_mask):
        """
        Extracts features based on the internal structure (fleshiness) of the leaf
        using Euclidean Distance Transform.
        """
        if binary_mask is None:
            return {"geo_mean_thickness": 0.0}

        # Distance Transform: Calculates distance to nearest zero pixel (background)
        # binary_mask should be leaf=255, bg=0
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # Extract distances only for the leaf pixels
        leaf_pixels = dist_transform[binary_mask > 0]

        if len(leaf_pixels) > 0:
            mean_thickness = np.mean(leaf_pixels)
        else:
            mean_thickness = 0.0

        return {"geo_mean_thickness": float(mean_thickness)}

    def compute_ratios(self, boundary_feats, contour):
        """
        Computes invariant geometric ratios.
        Includes zero-imputation for degenerate cases.
        """
        # Initialize with 0.0
        solidity = 0.0
        extent = 0.0
        aspect_ratio = 0.0
        eccentricity = 0.0
        roundness = 0.0

        area = boundary_feats["geo_area"]
        perimeter = boundary_feats["geo_perimeter"]
        major = boundary_feats["geo_major_axis"]
        minor = boundary_feats["geo_minor_axis"]

        if contour is not None and len(contour) > 0:
            # Solidity: Area / ConvexArea
            hull = cv2.convexHull(contour)
            convex_area = cv2.contourArea(hull)
            if convex_area > 1e-9:
                solidity = area / convex_area

            # Extent: Area / BoundingRectArea
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area > 1e-9:
                extent = area / rect_area

            # Aspect Ratio: Width / Height of bounding rect
            if h > 1e-9:
                aspect_ratio = float(w) / float(h)

        # Eccentricity: sqrt(1 - (minor/major)^2)
        if major > 1e-9:
            ratio_sq = (minor / major) ** 2
            if ratio_sq <= 1.0:
                eccentricity = np.sqrt(1.0 - ratio_sq)

        # Roundness: 4 * pi * Area / Perimeter^2
        if perimeter > 1e-9:
            roundness = (4 * np.pi * area) / (perimeter**2)

        return {
            "geo_solidity": float(solidity),
            "geo_extent": float(extent),
            "geo_aspect_ratio": float(aspect_ratio),
            "geo_eccentricity": float(eccentricity),
            "geo_roundness": float(roundness),
        }

    def extract_all(self, image_path):
        """
        Pipeline to extract all geometric features for a single image.
        """
        img, binary = self.process_image(image_path)

        # Default zero features if image processing fails
        if binary is None:
            keys = [
                "geo_area",
                "geo_perimeter",
                "geo_convex_perimeter",
                "geo_major_axis",
                "geo_minor_axis",
                "geo_mean_thickness",
                "geo_solidity",
                "geo_extent",
                "geo_aspect_ratio",
                "geo_eccentricity",
                "geo_roundness",
            ]
            return {k: 0.0 for k in keys}

        # Find contours
        # RETR_EXTERNAL: Only outer contours
        # CHAIN_APPROX_NONE: Store all contour points (lossless)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            # Assume the largest contour is the leaf
            c = max(contours, key=cv2.contourArea)
        else:
            c = None

        b_feats = self.extract_boundary_features(c)
        i_feats = self.extract_integral_features(binary)
        r_feats = self.compute_ratios(b_feats, c)

        # Merge dictionaries
        features = {**b_feats, **i_feats, **r_feats}
        return features


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
