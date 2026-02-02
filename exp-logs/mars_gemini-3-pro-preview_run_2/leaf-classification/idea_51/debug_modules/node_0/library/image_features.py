import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    POLARITY_THRESHOLD,
    DTYPE,
    RANDOM_SEED,
)
from library.utils import set_seed


class ImageFeatureExtractor:
    def __init__(self):
        self.cache_dir = WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        set_seed(RANDOM_SEED)

    def _check_and_correct_polarity(self, image):
        """
        Checks the background color based on corner pixels.
        If the background is white (mean > threshold), inverts the image
        so the leaf becomes the foreground (white) and background becomes black.

        Args:
            image (np.ndarray): Grayscale image (0-255).

        Returns:
            np.ndarray: Polarity-corrected image.
        """
        h, w = image.shape
        # Define corner regions (5x5 pixels)
        corners = [
            image[0:5, 0:5],
            image[0:5, w - 5 : w],
            image[h - 5 : h, 0:5],
            image[h - 5 : h, w - 5 : w],
        ]

        # Calculate mean intensity of corners
        corner_mean = np.mean([np.mean(c) for c in corners])

        # Normalize to 0-1 for comparison with POLARITY_THRESHOLD
        normalized_mean = corner_mean / 255.0

        if normalized_mean > POLARITY_THRESHOLD:
            # Background is bright/white, invert to make it black
            return cv2.bitwise_not(image)
        else:
            # Background is already dark/black
            return image

    def _extract_single_image_features(self, image_path):
        """
        Extracts Hu Moments and Geometric Scalars from a single image.

        Args:
            image_path (str): Full path to the image file.

        Returns:
            np.ndarray: 1D array of extracted features (float64).
        """
        # Initialize feature vector (11 features: 7 Hu + 4 Geometric)
        # Default to 0.0 if extraction fails
        features = np.zeros(11, dtype=DTYPE)

        if not os.path.exists(image_path):
            return features

        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return features

        # Correct polarity (ensure object is white, background is black)
        img = self._check_and_correct_polarity(img)

        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return features

        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)

        # 1. Moments and Hu Moments
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log transform is common for Hu moments, but we stick to raw values
        # as PowerTransformer is used in the pipeline later.
        features[0:7] = hu_moments

        # 2. Geometric Scalars
        area = moments["m00"]
        if area == 0:
            return features

        # Aspect Ratio and Extent
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        aspect_ratio = float(w) / h if h > 0 else 0.0
        extent = area / rect_area if rect_area > 0 else 0.0

        features[7] = aspect_ratio
        features[8] = extent

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        features[9] = solidity

        # Eccentricity
        # Requires at least 5 points to fit an ellipse
        if len(cnt) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                # ma = major axis, MA = minor axis (OpenCV notation can be tricky, usually (MA, ma) is (width, height) of rect)
                # Standard formula: a = major/2, b = minor/2.
                # In fitEllipse return: axes lengths are (minor_axis_length, major_axis_length)
                a = ma / 2
                b = MA / 2
                if a > 0:
                    # Eccentricity = sqrt(1 - (b/a)^2)
                    # Ensure term inside sqrt is non-negative
                    term = 1 - (min(a, b) / max(a, b)) ** 2
                    eccentricity = np.sqrt(max(0, term))
                    features[10] = eccentricity
            except Exception:
                features[10] = 0.0
        else:
            features[10] = 0.0

        return features

    def process_dataset(self, metadata_path, dataset_name, load_cached_data=True):
        """
        Processes a dataset defined by a metadata CSV file.
        Extracts morphometric features for all images listed.

        Args:
            metadata_path (str): Path to the metadata CSV (e.g., './metadata/train.csv').
            dataset_name (str): Name tag for the dataset (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: Matrix of shape (n_samples, 11) containing extracted features.
            np.ndarray: Array of IDs corresponding to the rows.
        """
        cache_file_features = os.path.join(
            self.cache_dir, f"morph_features_{dataset_name}.npy"
        )
        cache_file_ids = os.path.join(self.cache_dir, f"morph_ids_{dataset_name}.npy")

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(cache_file_features) and os.path.exists(cache_file_ids):
                print(f"Loading cached morphometric features for {dataset_name}...")
                X = np.load(cache_file_features)
                ids = np.load(cache_file_ids)
                return X, ids
            else:
                print(f"Cache not found for {dataset_name}. Processing from scratch...")
        else:
            print(f"Force processing morphometric features for {dataset_name}...")

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # 3. Process Images
        n_samples = len(df)
        n_features = 11
        X = np.zeros((n_samples, n_features), dtype=DTYPE)
        ids = df["id"].values

        print(f"Extracting features for {n_samples} images in {dataset_name} set...")

        for i, row in df.iterrows():
            # Construct full image path. Metadata contains relative path 'images/x.jpg'
            # INPUT_DIR is './input'.
            # So full path is './input/images/x.jpg'
            # Note: The metadata generation script in the prompt created paths like 'images/1.jpg'
            # inside the 'image_path' column.
            rel_path = row["image_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            features = self._extract_single_image_features(full_path)
            X[i, :] = features

            # Optional: Simple progress indicator for large datasets
            if (i + 1) % 100 == 0:
                pass  # Silent execution as requested

        # 4. Save to Cache
        np.save(cache_file_features, X)
        np.save(cache_file_ids, ids)
        print(f"Saved morphometric features to {cache_file_features}")

        return X, ids


def get_morphometric_features(load_cached_data=True):
    """
    Main entry point to get morphometric features for all splits (train, val, test).

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' feature matrices and IDs.
              Format: {
                  'train': (X_train, ids_train),
                  'val': (X_val, ids_val),
                  'test': (X_test, ids_test)
              }
    """
    extractor = ImageFeatureExtractor()

    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        metadata_path = os.path.join("./metadata", f"{split}.csv")
        X, ids = extractor.process_dataset(
            metadata_path, split, load_cached_data=load_cached_data
        )
        results[split] = (X, ids)

    return results
