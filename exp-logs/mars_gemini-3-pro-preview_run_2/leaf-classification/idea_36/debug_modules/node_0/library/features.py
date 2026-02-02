import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHE_DIR, FLOAT_PRECISION, VIEW_CONFIGS


class MacroFeatureExtractor:
    """
    Extracts deterministic morphological descriptors (Macro View) from binary leaf images.
    Implements caching to avoid redundant computation.
    """

    def __init__(self):
        self.descriptors = VIEW_CONFIGS["macro"]["descriptors"]

    def _extract_single_image(self, rel_image_path):
        """
        Computes morphological features for a single image.
        """
        full_path = os.path.join(INPUT_DIR, rel_image_path)

        # Initialize default vector (in case of failure)
        # 7 Hu moments + 6 scalars = 13 features
        default_features = np.zeros(13, dtype=FLOAT_PRECISION)

        if not os.path.exists(full_path):
            return default_features

        # Read image in grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return default_features

        # Invert image: The dataset has black leaves on white background.
        # Contours are best found on white objects against black background.
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return default_features

        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)

        # 1. Basic Moments
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return default_features

        # 2. Hu Moments (7 invariants)
        hu = cv2.HuMoments(M).flatten()

        # 3. Geometric Scalars
        area = M["m00"]
        perimeter = cv2.arcLength(cnt, True)

        # Bounding Rect for Aspect Ratio and Extent
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        # Convex Hull for Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Ellipse fit for Eccentricity
        eccentricity = 0
        if len(cnt) >= 5:
            try:
                (x_e, y_e), (MA, ma), angle = cv2.fitEllipse(cnt)
                # MA, ma are axis lengths. Sort to find major (a) and minor (b)
                a = max(MA, ma) / 2
                b = min(MA, ma) / 2
                if a > 0:
                    eccentricity = np.sqrt(1 - (b**2) / (a**2))
            except:
                eccentricity = 0

        # Combine all features
        # Order: Hu[0]..Hu[6], Area, Perimeter, AR, Solidity, Extent, Eccentricity
        features = np.concatenate(
            [
                hu,
                np.array(
                    [area, perimeter, aspect_ratio, solidity, extent, eccentricity]
                ),
            ]
        )

        return features.astype(FLOAT_PRECISION)

    def process(self, metadata_df, split_name, load_cached_data=True):
        """
        Extracts features for the entire dataframe, using caching.

        Args:
            metadata_df (pd.DataFrame): Dataframe containing 'image_path'.
            split_name (str): Name of the split (train/val/test) for cache naming.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            pd.DataFrame: DataFrame of macro features.
        """
        cache_file = os.path.join(CACHE_DIR, f"macro_features_{split_name}.parquet")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_file):
            # print(f"Loading cached macro features for {split_name} from {cache_file}")
            return pd.read_parquet(cache_file)

        # 2. Compute Features
        # print(f"Computing macro features for {split_name}...")
        feature_list = []

        # Define column names
        hu_cols = [f"hu_moment_{i}" for i in range(7)]
        scalar_cols = [
            "area",
            "perimeter",
            "aspect_ratio",
            "solidity",
            "extent",
            "eccentricity",
        ]
        cols = hu_cols + scalar_cols

        for _, row in metadata_df.iterrows():
            feats = self._extract_single_image(row["image_path"])
            feature_list.append(feats)

        df_macro = pd.DataFrame(feature_list, columns=cols, dtype=FLOAT_PRECISION)

        # 3. Save Cache
        df_macro.to_parquet(cache_file)

        return df_macro


class DataLoader:
    """
    Handles loading of metadata, provided global features, and extracted macro features.
    Ensures strict float64 precision and separates data into Views.
    """

    def __init__(self):
        self.macro_extractor = MacroFeatureExtractor()

    def load_split(self, split_name, metadata_path, load_cached_data=True):
        """
        Loads the dataset for a specific split.

        Args:
            split_name (str): 'train', 'val', or 'test'.
            metadata_path (str): Path to the metadata CSV.
            load_cached_data (bool): Whether to use cached macro features.

        Returns:
            ids (np.array): Array of image IDs.
            y (np.array or None): Array of target labels (None for test).
            views (dict): Dictionary of feature views {'global': DataFrame, 'macro': DataFrame}.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # 1. Extract IDs
        ids = df["id"].values

        # 2. Extract Target (if exists)
        y = None
        if "species" in df.columns:
            y = df["species"].values

        # 3. Prepare Global View
        # Identify columns for Margin, Shape, Texture
        feature_cols = [
            c for c in df.columns if c.startswith(("margin", "shape", "texture"))
        ]
        X_global = df[feature_cols].astype(FLOAT_PRECISION)

        # 4. Prepare Macro View
        X_macro = self.macro_extractor.process(
            df, split_name=split_name, load_cached_data=load_cached_data
        )

        # Ensure row counts match
        assert (
            len(X_global) == len(X_macro) == len(df)
        ), "Mismatch in feature row counts."

        views = {"global": X_global, "macro": X_macro}

        return ids, y, views
