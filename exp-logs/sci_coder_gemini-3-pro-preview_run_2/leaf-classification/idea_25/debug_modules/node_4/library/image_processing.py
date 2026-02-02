import os
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from library.config import (
    IMAGES_DIR,
    AUGMENTATION_PARAMS,
    N_AUGMENTATIONS,
    RANDOM_STATE,
    N_JOBS,
    USE_FLOAT64,
    DEBUG_MODE,
    DEBUG_SAMPLE_SIZE,
)


class ImageProcessor:
    """
    Handles loading, augmenting, and extracting robust morphometric features
    from binary leaf images using Monte-Carlo simulations.
    """

    def __init__(self):
        self.rng = np.random.RandomState(RANDOM_STATE)

    def _load_and_preprocess(self, image_rel_path):
        """
        Loads image, inverts it (black leaf -> white foreground), and binarizes.
        """
        full_path = os.path.join(IMAGES_DIR, os.path.basename(image_rel_path))
        if not os.path.exists(full_path):
            # Fallback for full paths provided in metadata vs relative
            # The metadata generator puts 'images/id.jpg', but IMAGES_DIR is './input/images'
            # So joining IMAGES_DIR + basename works if metadata has 'images/x.jpg'
            # or if metadata has just 'x.jpg'.
            pass

        # Try loading
        # Note: The metadata usually contains 'images/123.jpg'.
        # The provided config IMAGES_DIR is './input/images'.
        # If we join './input/images' with 'images/123.jpg', we get double 'images'.
        # We should rely on the structure provided in the task description.
        # Metadata 'image_path' is relative to input dir.
        # Let's try constructing the path relative to input root if possible,
        # but here we assume the passed path needs to be resolved.

        # Strategy: Try resolving against input root (../input)
        # But config IMAGES_DIR points to ./input/images.
        # Let's assume the caller passes the path as it appears in the metadata csv.
        # If metadata says "images/1.jpg", and we are in root, and input is "./input",
        # then file is "./input/images/1.jpg".

        path_to_try = os.path.join("./input", image_rel_path)

        img = cv2.imread(path_to_try, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Return None to handle gracefully
            return None

        # Invert: Input is black leaf (0) on white (255). We want leaf (255) on black (0).
        img = cv2.bitwise_not(img)

        # Binarize to ensure clean mask
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return thresh

    def _get_descriptors(self, mask):
        """
        Computes shape descriptors for a single binary mask.
        Returns a dictionary of features.
        """
        # Initialize defaults
        feats = {
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

        if mask is None or np.sum(mask) == 0:
            return feats

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return feats

        # Assume largest contour is the leaf
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area == 0:
            return feats

        # 1. Hu Moments
        moments = cv2.moments(c)
        hu = cv2.HuMoments(moments).flatten()
        # Log-modulus transform to handle scale: sign(x) * log(|x|)
        # Avoid log(0)
        for i in range(7):
            val = hu[i]
            if val != 0:
                feats[f"hu_{i+1}"] = np.sign(val) * np.log(np.abs(val))
            else:
                feats[f"hu_{i+1}"] = 0.0

        # 2. Geometric Scalars
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        # Aspect Ratio
        if h > 0:
            feats["aspect_ratio"] = float(w) / h

        # Extent
        if rect_area > 0:
            feats["extent"] = float(area) / rect_area

        # Solidity
        if hull_area > 0:
            feats["solidity"] = float(area) / hull_area

        # Eccentricity
        if len(c) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                if ma > 0:
                    a = ma / 2
                    b = MA / 2
                    # eccentricity = sqrt(1 - (b/a)^2)
                    # fitEllipse returns diameter, so divide by 2 for axis
                    # Ensure argument to sqrt is positive
                    ratio_sq = (b / a) ** 2
                    if ratio_sq <= 1:
                        feats["eccentricity"] = np.sqrt(1 - ratio_sq)
            except:
                pass  # Fallback to 0

        return feats

    def _augment_image(self, img, seed_offset):
        """
        Applies random affine transformation to the image.
        """
        if img is None:
            return None

        rows, cols = img.shape
        local_rng = np.random.RandomState(RANDOM_STATE + seed_offset)

        # Parameters
        rot_range = AUGMENTATION_PARAMS["rotation_range"]
        scale_range = AUGMENTATION_PARAMS["scale_range"]
        shear_range = AUGMENTATION_PARAMS["shear_range"]

        # Random values
        angle = local_rng.uniform(rot_range[0], rot_range[1])
        scale = local_rng.uniform(scale_range[0], scale_range[1])
        shear_deg = local_rng.uniform(shear_range[0], shear_range[1])

        # 1. Rotation and Scale
        M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)

        # 2. Shear
        # Shear matrix in x: x' = x + sh*y
        shear_rad = np.deg2rad(shear_deg)
        M_shear = np.float32([[1, np.tan(shear_rad), 0], [0, 1, 0]])

        # Apply Rotation/Scale first
        # Border constant 0 (black) because we inverted image (leaf is white)
        img_rot = cv2.warpAffine(
            img,
            M_rot,
            (cols, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Apply Shear
        # Adjust center for shear to avoid clipping?
        # For simplicity, we just warp. The leaf is usually centered.
        img_aug = cv2.warpAffine(
            img_rot,
            M_shear,
            (cols, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return img_aug

    def _process_single_row(self, row):
        """
        Worker function to process a single image row from metadata.
        """
        img_id = row["id"]
        img_path = row["image_path"]

        # Load base image
        base_img = self._load_and_preprocess(img_path)

        if base_img is None:
            # Return zeros if image load fails
            dummy_keys = [
                "hu_1",
                "hu_2",
                "hu_3",
                "hu_4",
                "hu_5",
                "hu_6",
                "hu_7",
                "aspect_ratio",
                "solidity",
                "extent",
                "eccentricity",
            ]
            res = {"id": img_id}
            for k in dummy_keys:
                res[f"{k}_mean"] = 0.0
                res[f"{k}_std"] = 0.0
            return res

        # Monte-Carlo Simulation
        results = []

        # Include the original image as one "view"?
        # The prompt says "generate M=10 affine perturbations".
        # We will generate N augmentations.

        for i in range(N_AUGMENTATIONS):
            aug_img = self._augment_image(base_img, seed_offset=int(img_id) * 100 + i)
            desc = self._get_descriptors(aug_img)
            results.append(desc)

        # Aggregation
        df_res = pd.DataFrame(results)
        agg_stats = {}
        agg_stats["id"] = img_id

        for col in df_res.columns:
            # Calculate Mean and Std
            vals = df_res[col].values
            agg_stats[f"{col}_mean"] = np.mean(vals)
            agg_stats[f"{col}_std"] = np.std(vals)

        return agg_stats

    def process_dataset(self, metadata_df, cache_path, load_cached_data=True):
        """
        Main driver to process a dataset (train/val/test).
        Handles caching and parallel execution.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            return pd.read_parquet(cache_path)

        print(
            f"Processing {len(metadata_df)} images with Monte-Carlo Augmentation (N={N_AUGMENTATIONS})..."
        )

        # Debug Mode
        if DEBUG_MODE:
            print(f"DEBUG_MODE: Processing subset of {DEBUG_SAMPLE_SIZE} images.")
            metadata_df = metadata_df.head(DEBUG_SAMPLE_SIZE)

        # 2. Parallel Processing
        # Convert dataframe to list of dicts for iteration
        rows = metadata_df.to_dict("records")

        results = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(self._process_single_row)(row) for row in rows
        )

        # 3. Create DataFrame
        features_df = pd.DataFrame(results)

        # Ensure ID is correct type
        features_df["id"] = features_df["id"].astype(int)

        # Cast to float64 if requested
        if USE_FLOAT64:
            feat_cols = [c for c in features_df.columns if c != "id"]
            features_df[feat_cols] = features_df[feat_cols].astype(np.float64)

        # 4. Save Cache
        print(f"Saving features to {cache_path}...")
        features_df.to_parquet(cache_path, index=False)

        return features_df


def extract_robust_morphometrics(metadata_df, cache_path, load_cached_data=True):
    """
    Public API function to run the processor.
    """
    processor = ImageProcessor()
    return processor.process_dataset(metadata_df, cache_path, load_cached_data)
