import os
import cv2
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, WORKING_DIR


def extract_geometric_features(
    metadata_df, load_cached_data=True, cache_name="geometric_features"
):
    """
    Extracts geometric features (Hu Moments and shape scalars) from images listed in the metadata DataFrame.

    This function computes 'Macro' resolution features:
    1. Hu Moments (7 invariants)
    2. Geometric Scalars: Aspect Ratio, Solidity, Extent, Eccentricity

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path' column.
        load_cached_data (bool): If True, attempts to load from cache.
        cache_name (str): Filename for the cache (without extension).

    Returns:
        pd.DataFrame: DataFrame containing the extracted features, aligned with the input DataFrame index.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached geometric features from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Extracting geometric features from scratch...")

    features_list = []

    # 2. Iterate through images
    # We use iterrows to preserve order matching the input dataframe
    for idx, row in metadata_df.iterrows():
        # Construct full path. metadata 'image_path' is relative to input dir.
        rel_path = row["image_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Initialize default features (NaN)
        feats = {
            "hu_1": np.nan,
            "hu_2": np.nan,
            "hu_3": np.nan,
            "hu_4": np.nan,
            "hu_5": np.nan,
            "hu_6": np.nan,
            "hu_7": np.nan,
            "aspect_ratio": np.nan,
            "solidity": np.nan,
            "extent": np.nan,
            "eccentricity": np.nan,
        }

        if os.path.exists(full_path):
            try:
                # Read image in grayscale
                # Dataset description: "binary black leaves against white backgrounds"
                # Pixel values: White=255, Black=0.
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    # We need the leaf to be the foreground (white/255) for analysis.
                    # Invert: Black(0) -> 255 (Leaf), White(255) -> 0 (Background)
                    img_inv = cv2.bitwise_not(img)

                    # Ensure strict binary (0 or 255)
                    _, bin_img = cv2.threshold(img_inv, 127, 255, cv2.THRESH_BINARY)

                    # --- Hu Moments ---
                    # cv2.moments calculates spatial moments up to 3rd order
                    moments = cv2.moments(bin_img)
                    # cv2.HuMoments calculates 7 invariant moments
                    hu_moments = cv2.HuMoments(moments).flatten()

                    for i, hu in enumerate(hu_moments):
                        feats[f"hu_{i+1}"] = hu

                    # --- Geometric Scalars via OpenCV Contours ---
                    # Replace skimage.measure.regionprops with cv2.findContours
                    contours, _ = cv2.findContours(
                        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Take the largest contour by area (assuming it's the leaf)
                        cnt = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(cnt)

                        # 1. Solidity: Area / Convex Hull Area
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            feats["solidity"] = area / hull_area
                        else:
                            feats["solidity"] = 0.0

                        # 2. Extent: Area / Bounding Rect Area
                        x, y, w, h = cv2.boundingRect(cnt)
                        rect_area = w * h
                        if rect_area > 0:
                            feats["extent"] = area / rect_area
                        else:
                            feats["extent"] = 0.0

                        # 3. Aspect Ratio: Width / Height of bounding box
                        if h > 0:
                            feats["aspect_ratio"] = float(w) / h
                        else:
                            feats["aspect_ratio"] = 0.0

                        # 4. Eccentricity: sqrt(1 - (minor_axis / major_axis)^2)
                        # Requires at least 5 points to fit an ellipse
                        if len(cnt) >= 5:
                            (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
                            # fitEllipse returns (center, size, angle) where size is (width, height)
                            # These are the diameters (lengths) of the axes.
                            major_axis = max(MA, ma)
                            minor_axis = min(MA, ma)

                            if major_axis > 0:
                                feats["eccentricity"] = np.sqrt(
                                    1 - (minor_axis / major_axis) ** 2
                                )
                            else:
                                feats["eccentricity"] = 0.0
                        else:
                            feats["eccentricity"] = 0.0

                    else:
                        # Image was completely empty (after inversion) or no contours found
                        pass

            except Exception as e:
                print(f"Error processing image {rel_path}: {e}")

        features_list.append(feats)

    # 3. Create DataFrame
    # Align index with input metadata_df to ensure safe concatenation later
    features_df = pd.DataFrame(features_list, index=metadata_df.index)

    # Fill NaNs with 0 (assuming missing object implies 0 geometric properties)
    features_df = features_df.fillna(0.0)

    # 4. Save to cache
    features_df.to_parquet(cache_path)
    print(f"Saved geometric features to {cache_path}")

    return features_df
