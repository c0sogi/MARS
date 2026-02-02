import os
import cv2
import numpy as np
import pandas as pd
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_70"
METADATA_DIR = "./metadata"


class ImageProcessor:
    def __init__(self):
        pass

    def _correct_polarity(self, img):
        """
        Invert image if the background is white.
        Checks 10x10 patches at the four corners.
        Assumes img is grayscale uint8.
        """
        h, w = img.shape
        # Sample corners
        corners = [
            img[0:10, 0:10],
            img[0:10, w - 10 : w],
            img[h - 10 : h, 0:10],
            img[h - 10 : h, w - 10 : w],
        ]
        # Calculate mean intensity of corners
        corner_mean = np.mean([np.mean(c) for c in corners])

        # If mean > 127 (approx 0.5 for normalized), background is likely white
        if corner_mean > 127:
            return cv2.bitwise_not(img)
        return img

    def _extract_features(self, img):
        """
        Extract Hu Moments (7) and Geometric Scalars (4).
        Total 11 features.
        """
        # Ensure binary thresholding
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Handle case with no contours
        if not contours:
            return np.zeros(11, dtype=np.float64)

        # Take the largest contour by area
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area == 0:
            return np.zeros(11, dtype=np.float64)

        # 1. Hu Moments (7 invariants)
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()

        # 2. Geometric Scalars
        # Bounding Rect
        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect Ratio
        aspect_ratio = float(w) / h if h > 0 else 0.0

        # Extent (Object Area / Bounding Rect Area)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0.0

        # Solidity (Object Area / Convex Hull Area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Eccentricity
        # Requires at least 5 points to fit ellipse
        eccentricity = 0.0
        if len(cnt) >= 5:
            try:
                (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
                # Sort axes to ensure minor <= major
                axes = sorted([MA, ma])
                minor_axis, major_axis = axes[0], axes[1]
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            except Exception:
                eccentricity = 0.0

        geo_features = np.array(
            [aspect_ratio, solidity, extent, eccentricity], dtype=np.float64
        )

        return np.concatenate([hu_moments, geo_features])

    def process_images(self, image_paths):
        """
        Process a list of image paths and return a DataFrame of features.
        """
        features_list = []

        for rel_path in image_paths:
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Default zero vector if file missing or unreadable
            feat_vector = np.zeros(11, dtype=np.float64)

            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_corrected = self._correct_polarity(img)
                    feat_vector = self._extract_features(img_corrected)

            features_list.append(feat_vector)

        feature_names = [f"hu_{i}" for i in range(7)] + [
            "aspect_ratio",
            "solidity",
            "extent",
            "eccentricity",
        ]
        return pd.DataFrame(features_list, columns=feature_names)


class DatasetLoader:
    def __init__(self):
        self.img_processor = ImageProcessor()

    def load_data(self, load_cached_data=True):
        """
        Load train, val, and test data.
        Merges provided tabular features with extracted image features.
        Returns: X_train, y_train, X_val, y_val, X_test, test_ids
        """
        set_seed(42)

        # Define cache file paths
        cache_train = os.path.join(CACHE_DIR, "train_merged.parquet")
        cache_val = os.path.join(CACHE_DIR, "val_merged.parquet")
        cache_test = os.path.join(CACHE_DIR, "test_merged.parquet")

        # Check if cache exists and is requested
        cache_exists = (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        )

        if load_cached_data and cache_exists:
            print("Loading data from cache...")
            df_train = pd.read_parquet(cache_train)
            df_val = pd.read_parquet(cache_val)
            df_test = pd.read_parquet(cache_test)
        else:
            print("Processing data from scratch...")
            # Load metadata
            df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
            df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
            df_test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

            # Extract Image Features
            print("Extracting features for Training set...")
            train_img_feats = self.img_processor.process_images(
                df_train_meta["image_path"]
            )

            print("Extracting features for Validation set...")
            val_img_feats = self.img_processor.process_images(df_val_meta["image_path"])

            print("Extracting features for Test set...")
            test_img_feats = self.img_processor.process_images(
                df_test_meta["image_path"]
            )

            # Merge features
            # We drop 'image_path' from meta as it's no longer needed in the feature matrix
            df_train = pd.concat(
                [df_train_meta.drop(columns=["image_path"]), train_img_feats], axis=1
            )
            df_val = pd.concat(
                [df_val_meta.drop(columns=["image_path"]), val_img_feats], axis=1
            )
            df_test = pd.concat(
                [df_test_meta.drop(columns=["image_path"]), test_img_feats], axis=1
            )

            # Ensure strict float64 precision for all feature columns
            # Exclude non-feature columns
            exclude_cols = ["id", "species"]
            for col in df_train.columns:
                if col not in exclude_cols:
                    df_train[col] = df_train[col].astype(np.float64)
                    df_val[col] = df_val[col].astype(np.float64)
                    df_test[col] = df_test[col].astype(np.float64)

            # Save to cache
            os.makedirs(CACHE_DIR, exist_ok=True)
            df_train.to_parquet(cache_train)
            df_val.to_parquet(cache_val)
            df_test.to_parquet(cache_test)
            print(f"Data saved to {CACHE_DIR}")

        # Prepare return values
        # X: Feature DataFrames (preserving column names)
        # y: Target Arrays
        # ids: ID Arrays

        y_train = df_train["species"].values
        X_train = df_train.drop(columns=["id", "species"])

        y_val = df_val["species"].values
        X_val = df_val.drop(columns=["id", "species"])

        test_ids = df_test["id"].values
        # Test set does not have 'species' column in the provided metadata/test.csv
        X_test = df_test.drop(columns=["id"])

        return X_train, y_train, X_val, y_val, X_test, test_ids
