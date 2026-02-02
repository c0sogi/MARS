import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    FLOAT_PRECISION,
    GEOMETRIC_FEATURES,
    ID_COL,
    FILE_PATH_COL,
)
from library.utils import set_seed


class GeometryExtractor:
    """
    Implements the Orthogonal-Geometric Basis Projection logic.
    Extracts 7 specific scalar descriptors from binary leaf images.
    """

    def __init__(self):
        # Ensure deterministic behavior where possible
        set_seed(42)

    def _extract_single_image(self, image_path):
        """
        Process a single image to extract geometric features.

        Args:
            image_path (str): Full path to the image file.

        Returns:
            np.ndarray: A 1D array of float64 features.
        """
        # Initialize default values
        feature_dict = {name: 0.0 for name in GEOMETRIC_FEATURES}

        if not os.path.exists(image_path):
            return np.array(
                [feature_dict[name] for name in GEOMETRIC_FEATURES],
                dtype=FLOAT_PRECISION,
            )

        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.array(
                [feature_dict[name] for name in GEOMETRIC_FEATURES],
                dtype=FLOAT_PRECISION,
            )

        # Polarity Correction: Leaves are binary black on white.
        # Invert so leaf is white (foreground) and background is black.
        # Cite solution_lesson_node_00145: Check your polarity.
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find Contours
        # CHAIN_APPROX_NONE to keep all boundary points for high fidelity
        # Cite solution_lesson_node_00149: Avoid Contour Compression.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return np.array(
                [feature_dict[name] for name in GEOMETRIC_FEATURES],
                dtype=FLOAT_PRECISION,
            )

        # Assume the largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)

        # 1. Absolute Scale: Area
        # Cite solution_lesson_node_00118: Absolute properties are defining characteristics.
        area = cv2.contourArea(cnt)

        # 2. Elongation: Eccentricity
        # Needs at least 5 points to fit an ellipse
        eccentricity = 0.0
        if len(cnt) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                # MA and ma are lengths of axes (Major and minor, but order not guaranteed)
                a = max(MA, ma) / 2.0
                b = min(MA, ma) / 2.0
                if a > 0:
                    eccentricity = np.sqrt(1 - (b**2 / a**2))
            except:
                eccentricity = 0.0

        # Derived Geometric Properties
        perimeter = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        convex_area = cv2.contourArea(hull)

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h

        # 3. Roughness: Solidity
        solidity = area / convex_area if convex_area > 0 else 0.0

        # 4. Rectangularity: Extent
        extent = area / rect_area if rect_area > 0 else 0.0

        # 5. Orientation: Aspect Ratio
        aspect_ratio = w / float(h) if h > 0 else 0.0

        # 6. Compactness: Roundness
        # 4 * pi * Area / Perimeter^2
        # Cite solution_lesson_node_00162: Use Zero-Imputation for degenerate cases.
        roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Populate Dictionary
        feature_dict["Area"] = area
        feature_dict["Eccentricity"] = eccentricity
        feature_dict["Solidity"] = solidity
        feature_dict["Extent"] = extent
        feature_dict["Aspect_Ratio"] = aspect_ratio
        feature_dict["Roundness"] = roundness

        # Assemble vector dynamically based on config order
        # Cite solution_lesson_node_00186: Decouple feature calculation from vector assembly.
        features = np.array(
            [feature_dict[name] for name in GEOMETRIC_FEATURES], dtype=FLOAT_PRECISION
        )

        return features

    def process_dataset(self, metadata_df, dataset_name, load_cached_data=True):
        """
        Extracts geometric features for an entire dataset (train/val/test).
        Handles caching to parquet files.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id' and 'file_path'.
            dataset_name (str): Name tag for the dataset (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: DataFrame with 'id' and geometric features.
        """
        cache_file = os.path.join(WORKING_DIR, f"{dataset_name}_geometry.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_file):
            print(
                f"Loading cached geometric features for {dataset_name} from {cache_file}..."
            )
            try:
                df_cached = pd.read_parquet(cache_file)
                # Verify columns match
                expected_cols = [ID_COL] + GEOMETRIC_FEATURES
                if all(col in df_cached.columns for col in expected_cols):
                    return df_cached
                else:
                    print("Cache schema mismatch. Recomputing...")
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Extracting geometric features for {dataset_name}...")

        ids = metadata_df[ID_COL].values
        file_paths = metadata_df[FILE_PATH_COL].values

        feature_matrix = []

        # Iterate with index to keep alignment safe
        for rel_path in tqdm(file_paths, desc=f"Processing {dataset_name}"):
            full_path = os.path.join(INPUT_DIR, rel_path)
            feats = self._extract_single_image(full_path)
            feature_matrix.append(feats)

        feature_matrix = np.array(feature_matrix, dtype=FLOAT_PRECISION)

        # Create DataFrame
        df_geo = pd.DataFrame(feature_matrix, columns=GEOMETRIC_FEATURES)
        df_geo.insert(0, ID_COL, ids)

        # 3. Save to cache
        try:
            df_geo.to_parquet(cache_file, index=False)
            print(f"Saved geometric features to {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache to {cache_file}: {e}")

        return df_geo


def get_geometric_features(metadata_path, dataset_name, load_cached_data=True):
    """
    Wrapper function to load metadata and process geometric features.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        dataset_name (str): Identifier for the dataset (e.g., 'train').
        load_cached_data (bool): Whether to use cached data.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and geometric features.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    extractor = GeometryExtractor()
    df_features = extractor.process_dataset(
        df_meta, dataset_name, load_cached_data=load_cached_data
    )

    return df_features
