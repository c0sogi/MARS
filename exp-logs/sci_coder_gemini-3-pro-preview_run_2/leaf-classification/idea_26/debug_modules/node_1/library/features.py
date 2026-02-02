import os
import cv2
import numpy as np
import pandas as pd
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    RANDOM_SEED,
    N_AUGMENTATIONS,
    AUG_ROTATION_RANGE,
    AUG_SCALE_RANGE,
    AUG_SHEAR_RANGE,
    DTYPE,
)
from library.utils import set_seed


def augment_image(image, rng):
    """
    Applies random affine transformations (rotation, scale, shear) to a binary image.

    Args:
        image (np.ndarray): Binary input image.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        np.ndarray: Augmented binary image.
    """
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)

    # 1. Rotation and Scale
    angle = rng.uniform(-AUG_ROTATION_RANGE, AUG_ROTATION_RANGE)
    scale = rng.uniform(1.0 - AUG_SCALE_RANGE, 1.0 + AUG_SCALE_RANGE)

    # Get rotation matrix (2x3)
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)

    # 2. Shear
    # Shear angles in degrees converted to radians
    shear_deg_x = rng.uniform(-AUG_SHEAR_RANGE, AUG_SHEAR_RANGE)
    shear_deg_y = rng.uniform(-AUG_SHEAR_RANGE, AUG_SHEAR_RANGE)

    shear_x = np.tan(np.deg2rad(shear_deg_x))
    shear_y = np.tan(np.deg2rad(shear_deg_y))

    # Shear matrix (3x3 for composition)
    M_shear = np.float32([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]])

    # Convert Rotation matrix to 3x3 for composition
    M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])

    # Combined Affine Matrix
    M_combined = np.matmul(M_shear, M_rot_3x3)

    # Extract 2x3 for warpAffine
    M_final = M_combined[:2, :]

    # Apply transformation
    # Border value is 255 (white background) based on dataset description
    augmented = cv2.warpAffine(
        image, M_final, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )

    # Ensure binary integrity after interpolation (thresholding)
    _, augmented_bin = cv2.threshold(augmented, 127, 255, cv2.THRESH_BINARY)

    return augmented_bin


def extract_descriptors(image):
    """
    Extracts geometric descriptors from a binary leaf image.

    Features (11 total):
    - Hu Moments (7)
    - Aspect Ratio
    - Solidity
    - Extent
    - Eccentricity

    Args:
        image (np.ndarray): Binary image (leaf is black, background is white).

    Returns:
        np.ndarray: 1D array of shape (11,) containing descriptors.
    """
    # Invert image so leaf is white (255) and background is black (0) for contour finding
    # The dataset description says "binary black leaves against white backgrounds".
    # OpenCV finds contours of white objects on black background.
    img_inv = cv2.bitwise_not(image)

    contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback if no contour found (should not happen with valid data)
        return np.zeros(11, dtype=DTYPE)

    # Assume the largest contour is the leaf
    cnt = max(contours, key=cv2.contourArea)

    # 1. Hu Moments (7 features)
    moments = cv2.moments(cnt)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Log transform to handle scale differences in Hu Moments,
    # but we will rely on PowerTransformer downstream, so raw values are fine.
    # However, sign restoration is common: -1 * copysign(1.0, hu) * log10(abs(hu))
    # We will return raw values as PowerTransformer handles distribution shaping better.

    # 2. Geometric Scalars
    area = moments["m00"]
    if area == 0:
        return np.zeros(11, dtype=DTYPE)

    # Aspect Ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0.0

    # Extent (Object Area / Bounding Rect Area)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Solidity (Object Area / Convex Hull Area)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Eccentricity
    # Using moments of inertia
    # (mu20 - mu02)^2 + 4*mu11^2
    # This relates to the ratio of major/minor axes
    if moments["m00"] != 0:
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]

        delta = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
        lambda1 = (mu20 + mu02 + delta) / 2
        lambda2 = (mu20 + mu02 - delta) / 2

        if lambda1 == 0:
            eccentricity = 0.0
        else:
            eccentricity = np.sqrt(1 - lambda2 / lambda1)
    else:
        eccentricity = 0.0

    descriptors = np.concatenate(
        [hu_moments, [aspect_ratio, solidity, extent, eccentricity]]
    )

    return descriptors.astype(DTYPE)


def process_single_image(image_path, rng):
    """
    Loads an image, generates augmentations, and computes probabilistic features.

    Args:
        image_path (str): Relative path to the image.
        rng (np.random.Generator): Random generator.

    Returns:
        np.ndarray: 22-dimensional feature vector (Mean + Std).
    """
    full_path = os.path.join(INPUT_DIR, image_path)

    # Load image in grayscale
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Return zeros if image load fails
        return np.zeros(22, dtype=DTYPE)

    # Ensure binary
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Storage for descriptors of all augmentations
    # We include the original image as one "augmentation" (identity transform) implicitly?
    # The prompt says "Generate K augmented versions". Let's do K augmentations.

    descriptors_list = []

    # 1. Extract from original (optional, but good for baseline mean)
    # We will stick to strictly N_AUGMENTATIONS perturbed versions to capture stability around the mode.
    # Or we can include original. Let's include original + K-1 augmentations or just K augmentations.
    # To strictly follow "Monte-Carlo Augmentation... generate K=10 affine perturbations",
    # we will generate 10 perturbations.

    for _ in range(N_AUGMENTATIONS):
        aug_img = augment_image(img_bin, rng)
        desc = extract_descriptors(aug_img)
        descriptors_list.append(desc)

    descriptors_array = np.array(descriptors_list, dtype=DTYPE)

    # Compute Mean and Std
    mean_vec = np.mean(descriptors_array, axis=0)
    std_vec = np.std(descriptors_array, axis=0)

    return np.concatenate([mean_vec, std_vec])


def get_probabilistic_features(metadata_df, dataset_name, load_cached_data=True):
    """
    Main function to generate or load probabilistic morphometric features.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'image_path'.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for caching.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Feature matrix of shape (N_samples, 22).
    """
    cache_path = os.path.join(WORKING_DIR, f"{dataset_name}_probabilistic_features.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(
            f"Loading cached probabilistic features for {dataset_name} from {cache_path}..."
        )
        try:
            features = np.load(cache_path)
            if features.shape[0] == len(metadata_df):
                return features
            else:
                print(
                    f"Cache shape mismatch ({features.shape[0]} vs {len(metadata_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Compute Features
    print(f"Generating probabilistic features for {dataset_name}...")

    # Initialize random generator
    # We use a specific seed for feature generation to ensure deterministic augmentations across runs
    rng = np.random.default_rng(RANDOM_SEED)

    feature_list = []
    image_paths = metadata_df["image_path"].values

    for idx, img_path in enumerate(image_paths):
        # Optional: Print progress every 100 images
        # if idx % 100 == 0:
        #     print(f"Processing {idx}/{len(image_paths)}")

        feat_vec = process_single_image(img_path, rng)
        feature_list.append(feat_vec)

    features = np.array(feature_list, dtype=DTYPE)

    # 3. Save Cache
    try:
        np.save(cache_path, features)
        print(f"Saved probabilistic features to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return features
