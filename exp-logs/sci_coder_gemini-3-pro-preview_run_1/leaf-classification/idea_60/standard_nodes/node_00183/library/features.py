import os
import cv2
import numpy as np
import pandas as pd
from library import config, utils

# Set seed for reproducibility
utils.set_seed(config.RANDOM_SEED)


class ImageFeatureExtractor:
    """
    Extracts geometric features from binary leaf images.
    """

    def __init__(self):
        pass

    def process_image(self, image_path):
        """
        Loads image, preprocesses, and extracts geometric features.
        Returns a dictionary of features.
        """
        # Default zero features in case of failure
        defaults = {k: 0.0 for k in config.GEOMETRIC_FEATURES}

        if not os.path.exists(image_path):
            return defaults

        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return defaults

        # Polarity Correction:
        # Dataset description: "binary black leaves against white backgrounds".
        # We need Leaf=Foreground(255), Background=Black(0).
        # Use THRESH_BINARY_INV.
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Find Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return defaults

        # Assume largest contour is the leaf
        cnt = max(contours, key=cv2.contourArea)

        # --- Basic Geometric Features ---
        area = cv2.contourArea(cnt)
        if area == 0:
            return defaults

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0.0
        aspect_ratio_rect = float(w) / h if h > 0 else 0.0

        # Ellipse fit for Eccentricity
        if len(cnt) >= 5:
            try:
                # fitEllipse returns ((x,y), (MA, ma), angle)
                (_, (d1, d2), _) = cv2.fitEllipse(cnt)
                a = max(d1, d2)  # Major axis
                b = min(d1, d2)  # Minor axis
                eccentricity = np.sqrt(1 - (b / a) ** 2) if a > 0 else 0.0
            except:
                eccentricity = 0.0
        else:
            eccentricity = 0.0

        # Roundness: 4 * pi * Area / Perimeter^2
        # Cite Lesson 151: Explicitly providing non-linear ratios is crucial for linear models.
        # Cite Lesson 162: Use Zero-Imputation for degenerate cases, not epsilon.
        perimeter = cv2.arcLength(cnt, True)
        roundness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        return {
            "area": float(area),
            "eccentricity": float(eccentricity),
            "solidity": float(solidity),
            "extent": float(extent),
            "aspect_ratio": float(aspect_ratio_rect),
            "roundness": float(roundness),
        }


def process_dataset(split, load_cached_data=True, debug_limit=None):
    """
    Main function to process a dataset split (train/val/test).
    Loads metadata, extracts features (or loads cache), and returns DataFrame.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.
        debug_limit (int, optional): Limit number of rows for debugging.

    Returns:
        pd.DataFrame: DataFrame containing ID, Target (if avail), Tabular Features, and Geometric Features.
    """

    # Determine cache path based on split
    if split == "train":
        cache_path = config.CACHE_TRAIN_FEATURES
    elif split == "val":
        cache_path = config.CACHE_VAL_FEATURES
    elif split == "test":
        cache_path = config.CACHE_TEST_FEATURES
    else:
        raise ValueError(f"Invalid split: {split}")

    # 1. Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        utils.Logger.info(f"Loading cached features for {split} from {cache_path}")
        df = utils.load_cache_parquet(cache_path)
        if debug_limit:
            df = df.head(debug_limit)
        return df

    # 2. Compute from scratch
    utils.Logger.info(f"Generating features for {split} from scratch...")

    # Load metadata
    meta_df = utils.load_metadata(split)

    if debug_limit:
        meta_df = meta_df.head(debug_limit)
        utils.Logger.info(f"Debug mode: processing first {debug_limit} rows.")

    extractor = ImageFeatureExtractor()
    new_features_list = []

    total = len(meta_df)
    log_interval = max(1, total // 10)

    with utils.Timer(f"Feature Extraction for {split}"):
        for idx, row in meta_df.iterrows():
            # Construct full image path
            # Metadata 'file_path' is relative (e.g., 'images/1.jpg')
            full_path = os.path.join(config.INPUT_DIR, row[config.FILE_PATH_COL])

            # Extract features
            feats = extractor.process_image(full_path)
            new_features_list.append(feats)

            if (idx + 1) % log_interval == 0:
                utils.Logger.info(f"Processed {idx + 1}/{total} images...")

    # Create DataFrame from new features
    new_feats_df = pd.DataFrame(new_features_list)

    # Concatenate with original metadata
    # Reset indices to ensure correct alignment
    meta_df = meta_df.reset_index(drop=True)
    new_feats_df = new_feats_df.reset_index(drop=True)

    final_df = pd.concat([meta_df, new_feats_df], axis=1)

    # 3. Save to cache (only if not debugging to avoid overwriting full cache with partial)
    if not debug_limit:
        utils.save_cache_parquet(cache_path, final_df)

    return final_df
