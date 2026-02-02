import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHE_DIR, GEOMETRIC_FEATURES, FLOAT_PRECISION
from library.utils import set_seed


class GeometricFeatureExtractor:
    """
    Extracts the 'Golden Geometric Scalars' from raw binary images.
    Implements polarity correction and implicit filtering.
    """

    def __init__(self, metadata_df):
        """
        Initialize the extractor with metadata.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
        """
        self.metadata_df = metadata_df.copy()

    def _process_single_image(self, rel_path):
        """
        Extracts geometric features from a single image using OpenCV.

        Args:
            rel_path (str): Relative path to the image file.

        Returns:
            dict: Dictionary containing the extracted features.
        """
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Initialize features with 0.0
        features = {k: 0.0 for k in GEOMETRIC_FEATURES}

        if not os.path.exists(full_path):
            return features

        # Load image in grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return features

        # Polarity Correction:
        # The dataset consists of binary black leaves (0) against white backgrounds (255).
        # OpenCV's findContours assumes white objects on black background.
        # Therefore, we invert the image.
        img_inverted = cv2.bitwise_not(img)

        # Find contours
        # Cite solution_lesson_node_00149: Use CHAIN_APPROX_NONE to avoid contour compression
        # and maximize precision of geometric feature extraction.
        contours, _ = cv2.findContours(
            img_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return features

        # Implicit Filtering:
        # Select the largest contour by area to filter out noise/artifacts.
        cnt = max(contours, key=cv2.contourArea)

        # 1. Area
        area = cv2.contourArea(cnt)
        features["Area"] = float(area)

        if area == 0:
            return features

        # 2. Solidity (Area / Convex Hull Area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            features["Solidity"] = float(area / hull_area)
        else:
            features["Solidity"] = 0.0

        # 3. Extent (Area / Bounding Rect Area)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area > 0:
            features["Extent"] = float(area / rect_area)
        else:
            features["Extent"] = 0.0

        # 4. Aspect Ratio (Bounding Rect Width / Height)
        # Using Axis Aligned Bounding Box (AABB)
        if h > 0:
            features["AspectRatio"] = float(w / h)
        else:
            features["AspectRatio"] = 0.0

        # 5. Eccentricity (Fit Ellipse)
        # Requires at least 5 points to fit an ellipse
        if len(cnt) >= 5:
            try:
                # fitEllipse returns ((x,y), (MA, ma), angle)
                # MA and ma are the lengths of the major and minor axes (not semi-axes)
                (x_e, y_e), (d1, d2), angle = cv2.fitEllipse(cnt)

                if d1 > 0 and d2 > 0:
                    major_axis = max(d1, d2)
                    minor_axis = min(d1, d2)
                    # Eccentricity e = sqrt(1 - (b/a)^2)
                    # where a is semi-major, b is semi-minor
                    # (b/a)^2 is equivalent to (minor_axis/major_axis)^2
                    features["Eccentricity"] = np.sqrt(
                        1 - (minor_axis / major_axis) ** 2
                    )
                else:
                    features["Eccentricity"] = 0.0
            except Exception:
                features["Eccentricity"] = 0.0
        else:
            features["Eccentricity"] = 0.0

        return features

    def extract_features(
        self,
        load_cached_data=True,
        cache_name="geometric_features.parquet",
        debug_limit=None,
    ):
        """
        Extracts features for all images in the metadata, with caching mechanism.

        Args:
            load_cached_data (bool): If True, attempts to load from cache first.
            cache_name (str): Filename for the cache file.
            debug_limit (int, optional): If set, limits the number of images processed (for debugging).

        Returns:
            pd.DataFrame: DataFrame containing 'id' and the extracted geometric features.
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, cache_name)

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path) and debug_limit is None:
            print(f"Loading cached geometric features from {cache_path}")
            try:
                df_features = pd.read_parquet(cache_path)
                # Verify schema
                expected_cols = ["id"] + GEOMETRIC_FEATURES
                if all(col in df_features.columns for col in expected_cols):
                    # Enforce precision
                    for col in GEOMETRIC_FEATURES:
                        df_features[col] = df_features[col].astype(FLOAT_PRECISION)
                    return df_features
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Extracting geometric features for {len(self.metadata_df)} images...")

        ids = self.metadata_df["id"].values
        paths = self.metadata_df["file_path"].values

        if debug_limit is not None:
            ids = ids[:debug_limit]
            paths = paths[:debug_limit]
            print(f"Debug mode: Limiting to {debug_limit} images.")

        results = []
        for img_id, rel_path in zip(ids, paths):
            feats = self._process_single_image(rel_path)
            feats["id"] = img_id
            results.append(feats)

        df_features = pd.DataFrame(results)

        # Organize columns
        cols = ["id"] + GEOMETRIC_FEATURES
        df_features = df_features[cols]

        # Enforce float64 precision
        for col in GEOMETRIC_FEATURES:
            df_features[col] = df_features[col].astype(FLOAT_PRECISION)

        # 3. Save to cache (only if not debugging)
        if debug_limit is None:
            try:
                df_features.to_parquet(cache_path, index=False)
                print(f"Saved geometric features to {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to save cache to {cache_path}: {e}")

        return df_features
