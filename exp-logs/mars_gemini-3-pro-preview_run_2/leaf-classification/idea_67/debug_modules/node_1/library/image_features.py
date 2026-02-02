import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    CACHE_DIR,
    INPUT_DIR,
    INVERT_THRESHOLD,
    ID_COL,
    IMAGE_PATH_COL,
)
from library.utils import set_seed


def extract_morphometrics(
    metadata_df: pd.DataFrame, dataset_name: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Extracts morphometric features (Hu Moments + Geometric Scalars) from images.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing image paths and IDs.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and extracted feature columns.
    """
    set_seed()

    # Define cache path
    cache_path = os.path.join(CACHE_DIR, f"morphometrics_{dataset_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached morphometrics for {dataset_name} from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Extracting morphometrics for {dataset_name}...")

    features_list = []

    # 2. Process each image
    # Ensure we iterate in the order of the dataframe
    for _, row in metadata_df.iterrows():
        image_id = row[ID_COL]
        rel_path = row[IMAGE_PATH_COL]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Default features in case of failure
        feats = {
            ID_COL: image_id,
            "hu_1": 0.0,
            "hu_2": 0.0,
            "hu_3": 0.0,
            "hu_4": 0.0,
            "hu_5": 0.0,
            "hu_6": 0.0,
            "hu_7": 0.0,
            "aspect_ratio": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "eccentricity": 0.0,
        }

        if os.path.exists(full_path):
            # Read image in grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                h, w = img.shape

                # --- Polarity Correction ---
                # Check corners to determine if background is white or black
                # Sample 5x5 regions at 4 corners
                margin = 5
                if h > margin and w > margin:
                    corners = [
                        img[0:margin, 0:margin],
                        img[0:margin, w - margin : w],
                        img[h - margin : h, 0:margin],
                        img[h - margin : h, w - margin : w],
                    ]
                    # Calculate mean intensity of corners (0-255)
                    corner_mean = np.mean([np.mean(c) for c in corners])
                else:
                    corner_mean = np.mean(img)  # Fallback for tiny images

                # Normalize to 0-1 for threshold comparison
                if (corner_mean / 255.0) > INVERT_THRESHOLD:
                    # Background is white (likely), Leaf is black.
                    # Invert so Leaf is White (255) and Background is Black (0) for contour finding.
                    img = 255 - img

                # --- Contour Extraction ---
                # Threshold to ensure strict binary (though dataset is already binary)
                _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Assume largest contour is the leaf
                    cnt = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(cnt)

                    if area > 0:
                        # --- Hu Moments ---
                        moments = cv2.moments(cnt)
                        hu = cv2.HuMoments(moments).flatten()
                        # We use raw Hu moments as requested (or log transformed is common, but prompt implies raw/poly input)
                        # Given the "Physical Polynomial Experts" description, raw values are often safer inputs for subsequent PowerTransformer.
                        for i in range(7):
                            feats[f"hu_{i+1}"] = hu[i]

                        # --- Geometric Scalars ---

                        # 1. Aspect Ratio & Extent (Bounding Rect)
                        x, y, rw, rh = cv2.boundingRect(cnt)
                        rect_area = rw * rh
                        if rh > 0:
                            feats["aspect_ratio"] = float(rw) / rh
                        if rect_area > 0:
                            feats["extent"] = area / rect_area

                        # 2. Solidity (Convex Hull)
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            feats["solidity"] = area / hull_area

                        # 3. Eccentricity (Fit Ellipse)
                        if len(cnt) >= 5:
                            try:
                                (center, (MA, ma), angle) = cv2.fitEllipse(cnt)
                                # MA, ma are axis lengths (diameters).
                                major_axis = max(MA, ma)
                                minor_axis = min(MA, ma)
                                if major_axis > 0:
                                    # e = sqrt(1 - (b/a)^2)
                                    feats["eccentricity"] = np.sqrt(
                                        1 - (minor_axis / major_axis) ** 2
                                    )
                            except Exception:
                                pass  # Keep default 0.0

        features_list.append(feats)

    # 3. Create DataFrame
    df_features = pd.DataFrame(features_list)

    # 4. Save to Cache
    # Ensure cache directory exists (handled by config, but good to be safe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_features.to_parquet(cache_path, index=False)

    return df_features
