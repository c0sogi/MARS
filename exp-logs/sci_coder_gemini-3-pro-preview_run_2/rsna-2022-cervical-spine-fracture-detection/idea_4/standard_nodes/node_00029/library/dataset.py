import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="Dataset")


def get_natural_key(text):
    """
    Helper to sort strings containing numbers naturally (e.g., '10.dcm' after '2.dcm').
    """
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


import re


def cache_study_paths(metadata_df, cache_file, load_cached_data=True):
    """
    Generates or loads a cache of sorted image paths for each study.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'StudyInstanceUID' and 'image_path'.
        cache_file (str): Path to the parquet file for caching.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping of StudyInstanceUID -> List[full_file_paths]
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            logger.info(f"Loading paths cache from {cache_file}...")
            cache_df = pd.read_parquet(cache_file)
            # Convert back to dict
            paths_dict = cache_df.set_index("StudyInstanceUID")["paths"].to_dict()
            # Ensure paths are lists (parquet might store as array)
            paths_dict = {k: list(v) for k, v in paths_dict.items()}
            return paths_dict
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    logger.info("Computing sorted file paths for all studies...")
    paths_dict = {}

    # Get unique studies and their directories
    # Assuming metadata has one row per study or we just need unique UIDs
    unique_studies = metadata_df[["StudyInstanceUID", "image_path"]].drop_duplicates()

    for _, row in unique_studies.iterrows():
        study_id = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_dir = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_dir):
            paths_dict[study_id] = []
            continue

        # List all files
        try:
            files = [f for f in os.listdir(full_dir) if f.endswith(".dcm")]
            # Sort by integer value of filename (Instance Number)
            # e.g. "10.dcm" -> 10. This is standard for this dataset structure.
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            full_paths = [os.path.join(full_dir, f) for f in files]
            paths_dict[study_id] = full_paths
        except Exception:
            paths_dict[study_id] = []

    # 3. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # Convert to DataFrame for parquet storage
        # Parquet handles lists/arrays in columns well
        cache_df = pd.DataFrame(
            {
                "StudyInstanceUID": list(paths_dict.keys()),
                "paths": list(paths_dict.values()),
            }
        )
        cache_df.to_parquet(cache_file)
        logger.info(f"Saved paths cache to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return paths_dict


def load_dicom_slice(path, image_size):
    """
    Reads a DICOM file, applies bone windowing, and resizes.

    Args:
        path (str): Path to DICOM file.
        image_size (tuple): (height, width).

    Returns:
        np.ndarray: Preprocessed image (H, W) in range [0, 1].
    """
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope/Intercept if present
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Bone Windowing
        # Center (Level) = 500, Width = 2000
        center = 500
        width = 2000

        lower = center - (width / 2)
        upper = center + (width / 2)

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)  # Normalize to [0, 1]

    except Exception:
        # Fallback for corrupt/missing files
        img = np.zeros(image_size, dtype=np.float32)

    # Resize
    if img.shape[:2] != image_size:
        img = cv2.resize(
            img, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR
        )

    return img


class CervicalSpineDataset(Dataset):
    """
    Dataset for Cervical Spine Fracture Detection.
    Features:
    - 2.5D Stacking (z-1, z, z+1)
    - Uniform Z-axis sampling to fixed sequence length
    - Volumetric Augmentation (identical geometric transform across slices)
    """

    def __init__(
        self, metadata_path, phase="train", load_cached_data=True, transform=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            phase (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached file paths.
            transform (A.Compose): Optional albumentations pipeline (mostly for pixel-level).
        """
        self.phase = phase
        self.transform = transform

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        # Debug Mode
        if Config.DEBUG:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)
            logger.info(f"DEBUG MODE: Reduced dataset to {len(self.df)} samples.")

        # Cache Paths
        # We use a unique cache file name per dataset split/idea to avoid conflicts
        cache_name = f"{phase}_paths_cache.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_name)

        self.paths_dict = cache_study_paths(self.df, cache_path, load_cached_data)

        # Filter out studies with no images found
        valid_uids = [
            uid
            for uid in self.df["StudyInstanceUID"]
            if uid in self.paths_dict and len(self.paths_dict[uid]) > 0
        ]
        original_len = len(self.df)
        self.df = self.df[self.df["StudyInstanceUID"].isin(valid_uids)].reset_index(
            drop=True
        )

        if len(self.df) < original_len:
            logger.warning(
                f"Dropped {original_len - len(self.df)} studies due to missing images."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # 1. Get Targets (if not test)
        targets = np.zeros(Config.NUM_CLASSES, dtype=np.float32)
        if self.phase != "test":
            # Columns: C1..C7, patient_overall
            # Config.TARGET_COLS order is important
            for i, col in enumerate(Config.TARGET_COLS):
                targets[i] = row[col]

        # 2. Get Slice Paths
        all_paths = self.paths_dict[study_uid]
        num_slices = len(all_paths)

        # 3. Uniform Sampling
        # We need exactly Config.SEQ_LEN indices
        if num_slices >= Config.SEQ_LEN:
            # Uniformly sample indices
            indices = np.linspace(0, num_slices - 1, Config.SEQ_LEN).astype(int)
        else:
            # Upsample / Interpolate indices
            if num_slices > 0:
                indices = np.linspace(0, num_slices - 1, Config.SEQ_LEN).astype(int)
            else:
                # Fallback for empty (should be filtered out, but safety first)
                indices = np.zeros(Config.SEQ_LEN, dtype=int)

        # 4. Generate Volumetric Augmentation Parameters (only for training)
        # We compute an affine matrix to apply to ALL slices identically
        affine_mat = None
        if self.phase == "train":
            # Random Rotation (-15, 15)
            angle = np.random.uniform(-15, 15)
            # Random Scale (0.8, 1.2)
            scale = np.random.uniform(0.8, 1.2)
            # Random Shift (-10%, 10%)
            h, w = Config.IMAGE_SIZE
            tx = np.random.uniform(-0.1, 0.1) * w
            ty = np.random.uniform(-0.1, 0.1) * h

            # Construct Matrix
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            affine_mat = M

        # 5. Load and Stack Images
        # Output shape: (Seq_Len, 3, H, W)
        images_seq = []

        for i in indices:
            # 2.5D Logic: (z-1, z, z+1)
            # Handle boundary conditions by clamping
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_slices - 1, i + 1)

            # Load the 3 slices
            # Note: If num_slices is 0 (edge case), paths will fail.
            # But we filtered empty studies.

            # Optimization: If idx_prev == idx_curr, we might load twice.
            # But caching individual images in memory is too heavy.
            # OS file cache helps here.

            img_prev = load_dicom_slice(all_paths[idx_prev], Config.IMAGE_SIZE)
            img_curr = load_dicom_slice(all_paths[idx_curr], Config.IMAGE_SIZE)
            img_next = load_dicom_slice(all_paths[idx_next], Config.IMAGE_SIZE)

            # Apply Volumetric Augmentation (Affine)
            if affine_mat is not None:
                h, w = Config.IMAGE_SIZE
                img_prev = cv2.warpAffine(
                    img_prev,
                    affine_mat,
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                img_curr = cv2.warpAffine(
                    img_curr,
                    affine_mat,
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                img_next = cv2.warpAffine(
                    img_next,
                    affine_mat,
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

            # Stack to (H, W, 3)
            # We stack along last dim for now, will transpose later
            slice_25d = np.stack([img_prev, img_curr, img_next], axis=-1)

            # Apply pixel-level transforms (brightness, noise) if any
            # These can be independent per slice as they don't change geometry
            if self.transform:
                augmented = self.transform(image=slice_25d)
                slice_25d = augmented["image"]

            images_seq.append(slice_25d)

        # Stack sequence: (Seq_Len, H, W, 3)
        images_seq = np.array(images_seq, dtype=np.float32)

        # Transpose to (Seq_Len, Channels, H, W) for PyTorch
        # Current: (S, H, W, C) -> (S, C, H, W)
        images_seq = np.transpose(images_seq, (0, 3, 1, 2))

        # Convert to Tensor
        images_tensor = torch.tensor(images_seq)
        targets_tensor = torch.tensor(targets)

        if self.phase == "test":
            return images_tensor, row["StudyInstanceUID"]
        else:
            return images_tensor, targets_tensor
