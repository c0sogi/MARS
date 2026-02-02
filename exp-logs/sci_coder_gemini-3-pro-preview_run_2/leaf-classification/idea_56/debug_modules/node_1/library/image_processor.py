import os
import cv2
import numpy as np
import pandas as pd

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_56/"


class ImageProcessor:
    def __init__(self, input_dir=INPUT_DIR, cache_dir=CACHE_DIR):
        """
        Initialize the ImageProcessor.

        Args:
            input_dir (str): Directory containing the input images.
            cache_dir (str): Directory to store cached features.
        """
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _preprocess_image(self, image_path):
        """
        Loads a binary image, corrects polarity so the leaf is foreground (white),
        and returns the binary image.

        Args:
            image_path (str): Relative path to the image (e.g., 'images/1.jpg').

        Returns:
            np.ndarray: Binary image with leaf as foreground (255) and background (0),
                        or None if loading fails.
        """
        full_path = os.path.join(self.input_dir, image_path)
        if not os.path.exists(full_path):
            return None

        # Load as grayscale
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Polarity Check
        # Check 4 corners to determine background color
        h, w = img.shape
        # Define corner size (min 5 pixels or image dims)
        c_sz = min(5, h, w)
        corners = [
            img[0:c_sz, 0:c_sz],
            img[0:c_sz, w - c_sz : w],
            img[h - c_sz : h, 0:c_sz],
            img[h - c_sz : h, w - c_sz : w],
        ]

        # Calculate mean intensity of corners
        corner_mean = np.mean([np.mean(c) for c in corners])

        # If background is white (high intensity), invert so leaf becomes white (foreground)
        # Threshold 127 is middle of 0-255
        if corner_mean > 127:
            img = cv2.bitwise_not(img)

        # Ensure binary (0 or 255)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        return img_bin

    def _calculate_features(self, img_bin):
        """
        Extracts Hu Moments and Geometric Scalars from a binary image.

        Args:
            img_bin (np.ndarray): Binary image (leaf=255, bg=0).

        Returns:
            np.ndarray: 1D array of 11 features (7 Hu Moments + 4 Geometric Scalars).
        """
        # Find contours
        contours, _ = cv2.findContours(
            img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros(11, dtype=np.float64)

        # Assume largest contour is the leaf
        c = max(contours, key=cv2.contourArea)

        # 1. Hu Moments
        moments = cv2.moments(c)
        hu_moments = cv2.HuMoments(moments).flatten()

        # 2. Geometric Scalars
        area = moments["m00"]
        if area == 0:
            return np.zeros(11, dtype=np.float64)

        # Aspect Ratio & Extent
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0.0

        # Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        # Eccentricity
        eccentricity = 0.0
        if len(c) >= 5:
            try:
                # fitEllipse returns ((center_x, center_y), (width, height), angle)
                (center), (axis1, axis2), angle = cv2.fitEllipse(c)
                major_axis = max(axis1, axis2)
                minor_axis = min(axis1, axis2)

                if major_axis > 0:
                    # e = sqrt(1 - (b/a)^2)
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
            except Exception:
                eccentricity = 0.0

        # Concatenate all features
        features = np.concatenate(
            [hu_moments, np.array([aspect_ratio, solidity, extent, eccentricity])]
        )

        return features.astype(np.float64)

    def extract_morphometrics(self, metadata_df, dataset_name, load_cached_data=True):
        """
        Extracts morphometric features for the given dataframe.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'image_path'.
            dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            np.ndarray: Feature matrix of shape (N, 11).
        """
        cache_file = os.path.join(self.cache_dir, f"morphometrics_{dataset_name}.npy")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached morphometrics for {dataset_name} from {cache_file}")
            return np.load(cache_file)

        print(f"Extracting morphometrics for {dataset_name}...")
        features_list = []

        for idx, row in metadata_df.iterrows():
            img_path = row["image_path"]
            img_bin = self._preprocess_image(img_path)

            if img_bin is not None:
                feats = self._calculate_features(img_bin)
            else:
                # Fallback for missing/bad images
                feats = np.zeros(11, dtype=np.float64)

            features_list.append(feats)

        features_array = np.array(features_list, dtype=np.float64)

        # Save to cache
        np.save(cache_file, features_array)
        print(f"Saved morphometrics for {dataset_name} to {cache_file}")

        return features_array
