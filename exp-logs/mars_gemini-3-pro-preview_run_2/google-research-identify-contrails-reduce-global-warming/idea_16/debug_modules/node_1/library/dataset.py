import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# ==========================================
# Constants for Normalization
# ==========================================
# Ash False Color Scheme Bounds (Kelvin)
_T11_BOUNDS = (243, 303)  # Band 14 (Blue)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)  # Band 14 - Band 11 (Green)
_TDIFF_BOUNDS = (-4, 2)  # Band 15 - Band 14 (Red)

# Temporal Difference Bounds (Kelvin)
# Assuming contrail cooling is within +/- 5K for normalization to [0,1]
_TEMPORAL_DIFF_BOUNDS = (-5, 5)


def normalize_range(data, bounds):
    """
    Normalizes data linearly from [min, max] to [0, 1].
    """
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_transforms(stage: str):
    """
    Returns the Albumentations transformation pipeline for the given stage.
    """
    if stage == "train":
        # Affine only: Rotation, Scale, Shift, Flip
        # Elastic/Grid distortions are excluded to preserve linear morphology
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT,
                    # Albumentations scale_limit is relative: (1+min, 1+max)
                    # Config.AUG_SCALE is (0.9, 1.1), so limit is (-0.1, 0.1)
                    scale_limit=(Config.AUG_SCALE[0] - 1.0, Config.AUG_SCALE[1] - 1.0),
                    rotate_limit=Config.AUG_ROTATION,
                    p=0.5,
                    border_mode=0,  # Constant 0 padding
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No augmentation, just tensor conversion
        return A.Compose([ToTensorV2()])


class ContrailDataset(Dataset):
    """
    Dataset class for Contrail Identification.
    Implements Decoupled Spatiotemporal Input (6 channels) and Caching.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        stage: str = "train",
        load_cached_data: bool = True,
    ):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing file paths and record_ids.
            stage (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to use disk caching for processed inputs.
        """
        self.metadata = metadata
        self.stage = stage
        self.load_cached_data = load_cached_data

        # Setup cache directory: ./working/idea_16/cache/{stage}/
        self.cache_dir = os.path.join(Config.CACHE_DIR, stage)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.transform = get_transforms(stage)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Input Data (6-channel tensor)
        # Tries to load from cache first, otherwise computes and saves
        img_np = self._load_input(row, record_id)

        # 2. Load Mask (if available)
        mask_np = None
        if "human_pixel_masks" in row and pd.notna(row["human_pixel_masks"]):
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            if os.path.exists(mask_path):
                # Load mask: H x W x 1
                mask_np = np.load(mask_path).astype(np.float32)
            else:
                # Fallback for missing mask file (should not happen in valid train set)
                mask_np = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32
                )
        else:
            # Test set or missing label
            mask_np = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # 3. Apply Transforms
        if mask_np is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]

            # Ensure mask is (1, H, W)
            # ToTensorV2 usually keeps mask as (H, W) or (H, W, 1) without permuting if it's not 'image'
            # We explicitly handle dimensions here
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.ndim == 3 and mask_tensor.shape[2] == 1:
                mask_tensor = mask_tensor.permute(2, 0, 1)  # HWC -> CHW

            return img_tensor, mask_tensor, record_id
        else:
            # Should not be reached with current logic, but for safety
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
            # Return dummy mask
            dummy_mask = torch.zeros(
                (1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32
            )
            return img_tensor, dummy_mask, record_id

    def _load_input(self, row, record_id):
        """
        Loads the 6-channel input tensor.
        Logic: Check cache -> If miss, load raw bands -> Process -> Save cache -> Return.
        """
        cache_path = os.path.join(self.cache_dir, f"{record_id}.npy")

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                # If load fails (corrupt file), proceed to re-compute
                pass

        # Compute from raw data
        # Load required bands: 11, 14, 15
        try:
            b11 = np.load(os.path.join(Config.INPUT_DIR, row["band_11"]))
            b14 = np.load(os.path.join(Config.INPUT_DIR, row["band_14"]))
            b15 = np.load(os.path.join(Config.INPUT_DIR, row["band_15"]))
        except FileNotFoundError:
            # Fallback for missing files (e.g. during path validation)
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 6), dtype=np.float32)

        # Temporal indices
        # Array shape is H x W x T
        # t=4 is the labeled frame (index 4)
        # t=3 is the previous frame (index 3)
        idx_t4 = Config.N_TIMES_BEFORE
        idx_t3 = Config.N_TIMES_BEFORE - 1

        # Extract frames
        t4_b11 = b11[..., idx_t4]
        t4_b14 = b14[..., idx_t4]
        t4_b15 = b15[..., idx_t4]

        t3_b11 = b11[..., idx_t3]
        t3_b14 = b14[..., idx_t3]
        t3_b15 = b15[..., idx_t3]

        # --- Construct Channels 1-3: Ash False Color (Static t=4) ---
        # Red: T15 - T14
        r = normalize_range(t4_b15 - t4_b14, _TDIFF_BOUNDS)
        # Green: T14 - T11
        g = normalize_range(t4_b14 - t4_b11, _CLOUD_TOP_TDIFF_BOUNDS)
        # Blue: T14
        b = normalize_range(t4_b14, _T11_BOUNDS)

        ash_composite = np.stack([r, g, b], axis=-1)

        # --- Construct Channels 4-6: Temporal Differences (t=4 - t=3) ---
        # We normalize these differences to [0, 1] using a fixed range (e.g., +/- 5K)
        # to make them compatible with the Ash channels while preserving relative dynamics.
        d11 = normalize_range(t4_b11 - t3_b11, _TEMPORAL_DIFF_BOUNDS)
        d14 = normalize_range(t4_b14 - t3_b14, _TEMPORAL_DIFF_BOUNDS)
        d15 = normalize_range(t4_b15 - t3_b15, _TEMPORAL_DIFF_BOUNDS)

        temporal_composite = np.stack([d11, d14, d15], axis=-1)

        # Combine to 6 channels
        final_img = np.concatenate([ash_composite, temporal_composite], axis=-1)

        # Clip to [0, 1] and cast to float32
        final_img = np.clip(final_img, 0, 1).astype(np.float32)

        # Save to cache
        if self.load_cached_data:
            try:
                np.save(cache_path, final_img)
            except Exception:
                pass

        return final_img
