import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from library.config import (
    WORKING_DIR,
    FLOAT_PRECISION,
    POLY_PARAMS,
    get_cache_path,
    set_seed,
)
from library.data_loader import load_image_paths


class MorphometricExtractor:
    """
    Extracts physical morphometric features from binary leaf images.

    Features extracted:
    1. Hu Moments (7 invariants): Scale, rotation, and translation invariant shape descriptors.
    2. Geometric Scalars: Aspect Ratio, Solidity, Extent, Eccentricity.

    Includes automatic polarity correction to ensure the leaf is always the foreground object.
    """

    def __init__(self):
        # Define feature names
        self.hu_names = [f"hu_{i}" for i in range(7)]
        self.scalar_names = ["aspect_ratio", "solidity", "extent", "eccentricity"]
        self.feature_names = self.hu_names + self.scalar_names

    def process_dataset(self, df, dataset_name, load_cached_data=True):
        """
        Extracts features for all images in the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing 'id' and 'image_path'.
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: Dataframe with 'id' and extracted morphometric features.
        """
        set_seed()

        cache_filename = f"morph_features_{dataset_name}.parquet"
        cache_path = get_cache_path(cache_filename)

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached morphometric features from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute Features
        print(f"Extracting morphometric features for {dataset_name}...")

        # Resolve absolute paths
        image_paths = load_image_paths(df)
        ids = df["id"].values

        features_list = []
        for path in image_paths:
            features_list.append(self._process_single_image(path))

        # Create DataFrame
        feat_df = pd.DataFrame(features_list, columns=self.feature_names)

        # Enforce precision
        feat_df = feat_df.astype(FLOAT_PRECISION)

        # Add ID column
        feat_df["id"] = ids

        # 3. Save Cache
        os.makedirs(WORKING_DIR, exist_ok=True)
        feat_df.to_parquet(cache_path, index=False)

        return feat_df

    def _process_single_image(self, path):
        """
        Process a single image file to extract features.
        """
        # Default zero vector in case of failure
        default_feats = np.zeros(len(self.feature_names), dtype=float)

        if not os.path.exists(path):
            return default_feats

        # Load Image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return default_feats

        # Polarity Correction
        # Check 4 corners (5x5 patches) to determine background color
        h, w = img.shape
        corners = [
            img[0:5, 0:5],
            img[0:5, w - 5 : w],
            img[h - 5 : h, 0:5],
            img[h - 5 : h, w - 5 : w],
        ]
        # Calculate mean intensity of corners
        corner_mean = np.mean([np.mean(c) for c in corners])

        # If background is white (high intensity), invert so leaf becomes foreground (white)
        # Threshold 127 is safe for binary/grayscale images
        if corner_mean > 127:
            img = cv2.bitwise_not(img)

        # Find Contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return default_feats

        # Select largest contour (the leaf)
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Filter noise
        if area < 1e-5:
            return default_feats

        # 1. Hu Moments
        moments = cv2.moments(cnt)
        # HuMoments returns 7x1 array, flatten to 1D
        hu = cv2.HuMoments(moments).flatten()

        # 2. Geometric Scalars
        # Bounding Rect
        x, y, rw, rh = cv2.boundingRect(cnt)
        rect_area = rw * rh
        aspect_ratio = float(rw) / rh if rh > 0 else 0.0
        extent = area / rect_area if rect_area > 0 else 0.0

        # Convex Hull -> Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Ellipse Fit -> Eccentricity
        eccentricity = 0.0
        if len(cnt) >= 5:
            try:
                # fitEllipse returns (center, (width, height), angle)
                (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(cnt)
                # ax1, ax2 are full axis lengths
                a = max(ax1, ax2) / 2.0
                b = min(ax1, ax2) / 2.0
                if a > 0:
                    eccentricity = np.sqrt(1 - (b**2 / a**2))
            except Exception:
                # Fallback if fitEllipse fails numerically
                pass

        return np.concatenate([hu, [aspect_ratio, solidity, extent, eccentricity]])


class PolynomialExpander:
    """
    Applies Polynomial Expansion to the morphometric features.
    Wraps sklearn PolynomialFeatures to handle DataFrames and ID columns.
    """

    def __init__(self):
        self.poly = PolynomialFeatures(**POLY_PARAMS)
        self.input_features = None
        self.output_features = None

    def fit(self, df):
        """
        Fits the PolynomialFeatures transformer.
        Ignores 'id' column.
        """
        X = df.drop(columns=["id"], errors="ignore")
        self.poly.fit(X)
        self.input_features = X.columns.tolist()
        self.output_features = self.poly.get_feature_names_out(self.input_features)
        return self

    def transform(self, df):
        """
        Transforms the dataframe.
        Preserves 'id' column.
        """
        if "id" in df.columns:
            ids = df["id"]
            X = df.drop(columns=["id"])
        else:
            ids = None
            X = df

        X_poly = self.poly.transform(X)

        # Create DataFrame with proper names
        df_poly = pd.DataFrame(X_poly, columns=self.output_features)

        # Enforce precision
        df_poly = df_poly.astype(FLOAT_PRECISION)

        # Re-attach ID
        if ids is not None:
            df_poly["id"] = ids.values

        return df_poly

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def get_morph_poly_features(df, dataset_name, load_cached_data=True):
    """
    Orchestrator function to generate the complete Polynomial-Morphometric view.

    1. Extracts Morphometric features (Cached to disk as it is computationally expensive).
    2. Expands with PolynomialFeatures (Computed on-the-fly as it is fast and deterministic).

    Args:
        df (pd.DataFrame): Input dataframe with image paths.
        dataset_name (str): Name of split (train/val/test).
        load_cached_data (bool): Whether to use caching.

    Returns:
        pd.DataFrame: Dataframe containing polynomial expanded morphometric features and 'id'.
    """
    # 1. Extract Physical Features
    extractor = MorphometricExtractor()
    df_morph = extractor.process_dataset(
        df, dataset_name, load_cached_data=load_cached_data
    )

    # 2. Polynomial Expansion
    # Note: PolynomialFeatures(degree=2) is a deterministic expansion (x -> 1, x, x^2).
    # Fitting on each dataset independently yields identical feature spaces provided
    # the input columns (morphometric features) are identical.
    expander = PolynomialExpander()
    df_poly = expander.fit_transform(df_morph)

    return df_poly
