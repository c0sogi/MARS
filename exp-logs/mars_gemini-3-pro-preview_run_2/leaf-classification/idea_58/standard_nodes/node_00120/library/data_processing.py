import os
import cv2
import numpy as np
import pandas as pd
from library import config
from library import utils


class ImageFeatureExtractor:
    """
    Extracts morphological features from binary leaf images.
    Features: Hu Moments (7) + Geometric Scalars (Aspect Ratio, Solidity, Extent, Eccentricity).
    """

    def __init__(self, images_dir=config.IMAGES_DIR):
        self.images_dir = images_dir

    def process_image(self, rel_path):
        """
        Reads an image and extracts features.
        Args:
            rel_path (str): Relative path from metadata (e.g., 'images/10.jpg').
        Returns:
            np.array: 11-dimensional feature vector (float64).
        """
        # Construct full path. Metadata path is 'images/id.jpg'.
        # self.images_dir is './input/images'.
        # We use basename to ensure we construct ./input/images/id.jpg correctly.
        filename = os.path.basename(rel_path)
        full_path = os.path.join(self.images_dir, filename)

        # Read image in grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing images (though EDA showed none)
        if img is None:
            return np.zeros(11, dtype=np.float64)

        # Normalize to [0, 1] float
        img = img.astype(np.float64) / 255.0

        # Polarity Correction
        # Check corners to see if background is white (high value)
        h, w = img.shape
        corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
        avg_corner = np.mean(corners)

        # If background is white, invert so leaf is white (1.0) and background is black (0.0)
        if avg_corner > config.INVERT_THRESHOLD:
            img = 1.0 - img

        # Binarize for contour detection
        # Leaf is now foreground (~1.0)
        _, bin_img = cv2.threshold(
            (img * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY
        )

        # 1. Hu Moments (7 features)
        moments = cv2.moments(bin_img)
        hu_moments = cv2.HuMoments(moments).flatten()

        # 2. Geometric Scalars (4 features)
        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        aspect_ratio = 0.0
        solidity = 0.0
        extent = 0.0
        eccentricity = 0.0

        if contours:
            # Assume largest contour is the leaf
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)

            if area > 0:
                # Bounding Rect -> Aspect Ratio, Extent
                x, y, w_rect, h_rect = cv2.boundingRect(cnt)
                rect_area = w_rect * h_rect

                if h_rect > 0:
                    aspect_ratio = float(w_rect) / float(h_rect)
                if rect_area > 0:
                    extent = area / rect_area

                # Convex Hull -> Solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area

                # Ellipse Fit -> Eccentricity
                # Requires at least 5 points
                if len(cnt) >= 5:
                    try:
                        (center, (MA, ma), angle) = cv2.fitEllipse(cnt)
                        # ma is major axis, MA is minor axis (opencv notation can be tricky, usually (MA, ma) = (width, height))
                        # Let's sort axes
                        axes = sorted([MA, ma])
                        minor, major = axes[0], axes[1]
                        if major > 0:
                            eccentricity = np.sqrt(1 - (minor / major) ** 2)
                    except:
                        pass

        geom_scalars = np.array(
            [aspect_ratio, solidity, extent, eccentricity], dtype=np.float64
        )

        # Combine
        return np.concatenate([hu_moments, geom_scalars])

    def extract_features(self, df):
        """
        Extracts features for all rows in the dataframe.
        """
        features = []
        for idx, row in df.iterrows():
            feat_vec = self.process_image(row["image_path"])
            features.append(feat_vec)

        cols = [f"phys_hu_{i}" for i in range(7)] + [
            "phys_aspect_ratio",
            "phys_solidity",
            "phys_extent",
            "phys_eccentricity",
        ]

        return pd.DataFrame(features, columns=cols, index=df.index)


class DatasetManager:
    """
    Manages data loading, feature merging, and scope slicing.
    """

    def __init__(self):
        self.extractor = ImageFeatureExtractor()

    def _get_paths(self, split):
        if split == "train":
            return (
                os.path.join(config.METADATA_DIR, "train.csv"),
                config.CACHE_PHYSICAL_TRAIN,
            )
        elif split == "val":
            return (
                os.path.join(config.METADATA_DIR, "val.csv"),
                config.CACHE_PHYSICAL_VAL,
            )
        elif split == "test":
            return (
                os.path.join(config.METADATA_DIR, "test.csv"),
                config.CACHE_PHYSICAL_TEST,
            )
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_data(self, split, load_cached_data=True):
        """
        Loads the dataset for a given split.

        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load physical features from parquet cache.

        Returns:
            pd.DataFrame: Merged dataframe containing metadata features and physical features.
        """
        meta_path, cache_path = self._get_paths(split)

        # 1. Load Metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        df_meta = pd.read_csv(meta_path)

        # 2. Load or Compute Physical Features
        df_phys = None
        if load_cached_data and os.path.exists(cache_path):
            try:
                df_phys = pd.read_parquet(cache_path)
                # Verify alignment
                if len(df_phys) != len(df_meta):
                    print(f"Warning: Cache length mismatch for {split}. Recomputing.")
                    df_phys = None
            except Exception as e:
                print(f"Warning: Failed to load cache for {split} ({e}). Recomputing.")
                df_phys = None

        if df_phys is None:
            print(f"Computing physical features for {split}...")
            df_phys = self.extractor.extract_features(df_meta)
            # Save to cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df_phys.to_parquet(cache_path, index=False)

        # 3. Merge
        # Reset index to ensure safe concatenation by row position
        df_meta_reset = df_meta.reset_index(drop=True)
        df_phys_reset = df_phys.reset_index(drop=True)
        df_full = pd.concat([df_meta_reset, df_phys_reset], axis=1)

        # 4. Cast Features to float64
        exclude_cols = ["id", "species", "image_path"]
        feature_cols = [c for c in df_full.columns if c not in exclude_cols]
        df_full[feature_cols] = df_full[feature_cols].astype(config.FLOAT_PRECISION)

        return df_full

    def get_scope_slice(self, df, scope):
        """
        Returns the feature matrix (numpy array) for the requested scope.
        """
        all_cols = df.columns
        exclude_cols = ["id", "species", "image_path"]

        if scope == config.SCOPE_GLOBAL:
            # All features
            cols = [c for c in all_cols if c not in exclude_cols]
        elif scope == config.SCOPE_PHYSICAL:
            cols = [c for c in all_cols if c.startswith("phys_")]
        elif scope == config.SCOPE_MARGIN:
            cols = [c for c in all_cols if c.startswith(config.PREFIX_MARGIN)]
        elif scope == config.SCOPE_SHAPE:
            cols = [c for c in all_cols if c.startswith(config.PREFIX_SHAPE)]
        elif scope == config.SCOPE_TEXTURE:
            cols = [c for c in all_cols if c.startswith(config.PREFIX_TEXTURE)]
        else:
            raise ValueError(f"Unknown scope: {scope}")

        return df[cols].values

    def get_targets(self, df):
        """
        Returns target labels if available, else None.
        """
        if "species" in df.columns:
            return df["species"].values
        return None

    def get_ids(self, df):
        """
        Returns image IDs.
        """
        return df["id"].values
