import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from library.config import Config


def extract_morphological_features(df, dataset_name, load_cached_data=True):
    """
    Extracts morphological features (Hu Moments and Geometric Scalars) from images
    referenced in the provided DataFrame. Implements caching to Parquet.

    Args:
        df (pd.DataFrame): Metadata DataFrame containing 'id' and 'image_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the extracted morphological features.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Construct cache file path
    cache_filename = f"morph_features_{dataset_name}.parquet"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_path):
            print(
                f"Loading cached morphological features for '{dataset_name}' from {cache_path}"
            )
            try:
                cached_df = pd.read_parquet(cache_path)
                # Verify length matches
                if len(cached_df) == len(df):
                    return cached_df
                else:
                    print(
                        f"Cache length mismatch ({len(cached_df)} vs {len(df)}). Recomputing..."
                    )
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")
        else:
            print(f"No cache found for '{dataset_name}' at {cache_path}. Computing...")
    else:
        print(f"Ignoring cache for '{dataset_name}'. Computing from scratch...")

    # 2. Compute Features
    print(f"Starting morphological feature extraction for {len(df)} images...")

    features_list = []

    # Define column names for clarity
    hu_cols = [f"hu_moment_{i}" for i in range(7)]
    scalar_cols = ["aspect_ratio", "solidity", "extent", "eccentricity"]

    for idx, row in df.iterrows():
        image_id = row["id"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Initialize feature dict with ID
        record = {"id": image_id}

        # Load Image
        if not os.path.exists(full_path):
            # Handle missing file gracefully (though metadata check passed)
            print(f"Warning: Image not found at {full_path}. Filling with zeros.")
            for col in hu_cols + scalar_cols:
                record[col] = 0.0
            features_list.append(record)
            continue

        # Read as grayscale (binary images are provided)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Failed to read image {full_path}. Filling with zeros.")
            for col in hu_cols + scalar_cols:
                record[col] = 0.0
            features_list.append(record)
            continue

        # Binarize just in case (though dataset says binary)
        _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # --- A. Hu Moments ---
        # Calculate moments
        moments = cv2.moments(bin_img)
        # Calculate Hu Moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log transform Hu moments to handle scale (sign * log10(abs(x)))
        # This is standard practice as raw Hu moments have vast dynamic ranges
        for i, val in enumerate(hu_moments):
            # Avoid log(0)
            if val == 0:
                record[hu_cols[i]] = 0.0
            else:
                # We store the raw-ish log transformed value usually: -1 * copysign(1.0, val) * log10(abs(val))
                # However, for simplicity and raw feature preservation as requested by prompt "Explicit Morphological Features",
                # we will store the raw values. The downstream PowerTransformer (Yeo-Johnson) handles the distribution better than manual log.
                record[hu_cols[i]] = float(val)

        # --- B. Geometric Scalars ---
        # Use skimage regionprops
        # Label the image to find connected components
        lbl_img = label(bin_img > 127)  # label expects boolean or integer
        regions = regionprops(lbl_img)

        if not regions:
            # Empty image
            for col in scalar_cols:
                record[col] = 0.0
        else:
            # If multiple regions, assume the leaf is the largest one by area
            leaf_props = max(regions, key=lambda r: r.area)

            # 1. Aspect Ratio
            # bounding_box: (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = leaf_props.bbox
            height = maxr - minr
            width = maxc - minc

            if height > 0:
                aspect_ratio = float(width) / float(height)
            else:
                aspect_ratio = 0.0

            record["aspect_ratio"] = aspect_ratio

            # 2. Solidity (Area / Convex Area)
            record["solidity"] = float(leaf_props.solidity)

            # 3. Extent (Area / Bounding Box Area)
            record["extent"] = float(leaf_props.extent)

            # 4. Eccentricity
            record["eccentricity"] = float(leaf_props.eccentricity)

        features_list.append(record)

    # Create DataFrame
    features_df = pd.DataFrame(features_list)

    # Ensure ID is integer (it might be converted to float during dict creation if not careful)
    features_df["id"] = features_df["id"].astype(int)

    # 3. Save to Cache
    print(f"Saving extracted features to {cache_path}...")
    features_df.to_parquet(cache_path, index=False)

    return features_df
