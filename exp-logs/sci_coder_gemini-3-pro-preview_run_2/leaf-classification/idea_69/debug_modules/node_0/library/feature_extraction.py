import os
import cv2
import numpy as np
import pandas as pd
from library.data_loader import get_data_split

# Directory for caching processed features
CACHE_DIR = "./working/idea_69"


def _extract_morphometrics_single(img: np.ndarray) -> np.ndarray:
    """
    Extracts Polarity-Corrected Morphometrics from a single binary leaf image.

    Features (11 dims):
        - Log-transformed Hu Moments (7)
        - Geometric Scalars: Aspect Ratio, Solidity, Extent, Eccentricity (4)
    """
    # Ensure image is single channel (grayscale)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Work on a copy to avoid modifying the original
    img_proc = img.copy()

    # Polarity Correction
    # Assumption: Leaf should be foreground (white/255) on background (black/0).
    # Check corners to detect background color.
    h, w = img_proc.shape
    corners = [
        img_proc[0, 0],
        img_proc[0, w - 1],
        img_proc[h - 1, 0],
        img_proc[h - 1, w - 1],
    ]

    # If corners are bright (> 127), assume white background and invert
    if np.mean(corners) > 127:
        img_proc = cv2.bitwise_not(img_proc)

    # Find Contours
    # RETR_EXTERNAL: retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(img_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Handle empty image or no contours found
    if not contours:
        return np.zeros(11, dtype=np.float64)

    # Assume the largest contour by area is the leaf
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # Filter noise
    if area == 0:
        return np.zeros(11, dtype=np.float64)

    # --- 1. Hu Moments (7 invariants) ---
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    # Log transform to handle scale differences and improve linear separability
    # Formula: -1 * sign(h) * log10(abs(h))
    hu_log = []
    for h_val in hu:
        if h_val == 0:
            hu_log.append(0.0)
        else:
            # Using absolute value inside log, preserving sign outside
            hu_log.append(-1 * np.sign(h_val) * np.log10(np.abs(h_val)))
    hu_features = np.array(hu_log, dtype=np.float64)

    # --- 2. Geometric Scalars (4) ---

    # Bounding Rectangle
    x, y, rect_w, rect_h = cv2.boundingRect(cnt)

    # Aspect Ratio
    aspect_ratio = float(rect_w) / rect_h if rect_h > 0 else 0.0

    # Extent (Object Area / Bounding Box Area)
    rect_area = rect_w * rect_h
    extent = float(area) / rect_area if rect_area > 0 else 0.0

    # Solidity (Object Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    # Requires fitting an ellipse (needs at least 5 points)
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (center, (axis1, axis2), angle) = cv2.fitEllipse(cnt)
            # axis1 and axis2 are the lengths of the axes (major/minor not guaranteed order)
            ma = max(axis1, axis2)
            ma_minor = min(axis1, axis2)

            if ma > 0:
                # e = sqrt(1 - (b/a)^2) where a is semi-major, b is semi-minor
                # (ma/2) / (MA/2) ratio is same as ma/MA
                eccentricity = np.sqrt(1 - (ma_minor / ma) ** 2)
        except Exception:
            # Fallback if ellipse fit fails
            eccentricity = 0.0

    geo_features = np.array(
        [aspect_ratio, solidity, extent, eccentricity], dtype=np.float64
    )

    # Concatenate
    return np.concatenate([hu_features, geo_features])


def get_feature_views(split: str, load_cached_data: bool = True):
    """
    Generates or loads the feature views for the Expert Library.

    Views generated:
    1. Global (192 features): Original Margin + Shape + Texture
    2. Margin (64 features)
    3. Shape (64 features)
    4. Texture (64 features)
    5. Morphometrics (11 features): Extracted from raw images

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to use cached .npz files.

    Returns:
        dict: {
            "ids": np.ndarray,
            "y": np.ndarray (or None for test),
            "views": {
                "Global": np.ndarray,
                "Margin": np.ndarray,
                "Shape": np.ndarray,
                "Texture": np.ndarray,
                "Morphometrics": np.ndarray
            }
        }
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"features_{split}.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=True is needed to load the string array 'y' (species names)
            with np.load(cache_path, allow_pickle=True) as data:
                result = {
                    "ids": data["ids"],
                    "views": {
                        "Global": data["view_Global"],
                        "Margin": data["view_Margin"],
                        "Shape": data["view_Shape"],
                        "Texture": data["view_Texture"],
                        "Morphometrics": data["view_Morphometrics"],
                    },
                }
                # Handle 'y' which might be None (stored as None object or missing)
                if "y" in data:
                    # Check if it was saved as a valid array
                    if data["y"].shape == ():  # 0-d array (None)
                        result["y"] = None
                    else:
                        result["y"] = data["y"]
                else:
                    result["y"] = None
                return result
        except Exception as e:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from Scratch

    # Load metadata and raw images
    # Note: get_data_split handles the caching of raw images internally
    df, images = get_data_split(split, load_cached_data=load_cached_data)

    # A. Tabular Views Extraction
    # Define column groups
    margin_cols = [f"margin_{i}" for i in range(1, 65)]
    shape_cols = [f"shape_{i}" for i in range(1, 65)]
    texture_cols = [f"texture_{i}" for i in range(1, 65)]

    # Extract and cast to float64
    X_margin = df[margin_cols].values.astype(np.float64)
    X_shape = df[shape_cols].values.astype(np.float64)
    X_texture = df[texture_cols].values.astype(np.float64)

    # Global view is the concatenation of the three pre-extracted sets
    X_global = np.hstack([X_margin, X_shape, X_texture])

    # B. Morphometric View Extraction
    # Process images to extract physical features
    morph_features_list = []
    for img in images:
        feat_vec = _extract_morphometrics_single(img)
        morph_features_list.append(feat_vec)

    X_morph = np.array(morph_features_list, dtype=np.float64)

    # C. Metadata
    ids = df["id"].values
    y = df["species"].values if "species" in df.columns else None

    # 3. Save to Cache
    save_dict = {
        "ids": ids,
        "view_Global": X_global,
        "view_Margin": X_margin,
        "view_Shape": X_shape,
        "view_Texture": X_texture,
        "view_Morphometrics": X_morph,
    }

    if y is not None:
        save_dict["y"] = y
    else:
        # Save a placeholder for consistency if needed, or handle existence check on load
        pass

    np.savez(cache_path, **save_dict)

    # 4. Return Result
    return {
        "ids": ids,
        "y": y,
        "views": {
            "Global": X_global,
            "Margin": X_margin,
            "Shape": X_shape,
            "Texture": X_texture,
            "Morphometrics": X_morph,
        },
    }
