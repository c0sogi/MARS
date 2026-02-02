import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, MORPHOLOGICAL_COLS, ID_COL, IMAGE_PATH_COL


class MorphologyExtractor:
    """
    Extracts explicit geometric/morphological descriptors from binary leaf images.
    Features include Hu Moments, Aspect Ratio, Solidity, Extent, and Eccentricity.
    """

    def __init__(self):
        pass

    def extract_single_image(self, rel_image_path):
        """
        Extracts morphological features for a single image.

        Args:
            rel_image_path (str): Relative path to the image (e.g., 'images/1.jpg').

        Returns:
            dict: Dictionary containing extracted features matching MORPHOLOGICAL_COLS.
        """
        full_path = os.path.join(INPUT_DIR, rel_image_path)

        # Initialize default features (zeros) in case of failure
        features = {col: 0.0 for col in MORPHOLOGICAL_COLS}

        if not os.path.exists(full_path):
            return features

        # Load image as grayscale (binary)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return features

        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return features

        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area == 0:
            return features

        # 1. Bounding Rect Features (Aspect Ratio, Extent)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        if h > 0:
            features["aspect_ratio"] = float(w) / h

        if rect_area > 0:
            features["extent"] = float(area) / rect_area

        # 2. Convex Hull Features (Solidity)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            features["solidity"] = float(area) / hull_area

        # 3. Ellipse Fit Features (Eccentricity)
        # fitEllipse requires at least 5 points
        if len(cnt) >= 5:
            try:
                (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
                major_axis = max(axis1, axis2)
                minor_axis = min(axis1, axis2)

                if major_axis > 0:
                    # Eccentricity = sqrt(1 - (b/a)^2)
                    # ratio of squares of axes lengths is equivalent to ratio of squares of semi-axes
                    features["eccentricity"] = np.sqrt(
                        1 - (minor_axis / major_axis) ** 2
                    )
            except Exception:
                # Fallback if ellipse fit fails numerically
                features["eccentricity"] = 0.0

        # 4. Hu Moments
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Assign Hu Moments
        for i in range(7):
            features[f"hu_moment_{i}"] = hu_moments[i]

        return features

    def process_dataset(self, metadata_df, cache_path, load_cached_data=True):
        """
        Process a dataset (dataframe with image paths) to extract features.
        Handles caching to avoid re-computation.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id' and 'image_path'.
            cache_path (str): Path to save/load the parquet cache.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame with 'id' and extracted morphological features.
        """
        # Ensure cache directory exists
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading morphological features from cache: {cache_path}")
            try:
                cached_df = pd.read_parquet(cache_path)
                # Verify it has the correct columns and IDs
                if (
                    set(MORPHOLOGICAL_COLS).issubset(cached_df.columns)
                    and ID_COL in cached_df.columns
                ):
                    # Check if IDs match (optional but good for safety, here we assume cache is valid if exists)
                    return cached_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Extracting morphological features for {len(metadata_df)} images...")

        results = []

        # Iterate over the dataframe
        # Note: Using a simple loop as dataset is small (<2000 images).
        # For larger datasets, multiprocessing would be preferred.
        for _, row in metadata_df.iterrows():
            img_id = row[ID_COL]
            rel_path = row[IMAGE_PATH_COL]

            # Extract features
            feats = self.extract_single_image(rel_path)
            feats[ID_COL] = img_id
            results.append(feats)

        # Create DataFrame
        feature_df = pd.DataFrame(results)

        # Ensure column order (ID first, then features)
        cols = [ID_COL] + MORPHOLOGICAL_COLS
        feature_df = feature_df[cols]

        # 3. Save to cache
        print(f"Saving morphological features to cache: {cache_path}")
        feature_df.to_parquet(cache_path, index=False)

        return feature_df
