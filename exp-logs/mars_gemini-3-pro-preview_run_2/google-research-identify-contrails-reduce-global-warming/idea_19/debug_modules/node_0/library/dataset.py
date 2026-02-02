import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for the specified split.

    Args:
        split (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,  # Constant padding
                    value=0,
                    p=Config.AUG_PROB,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, we only convert to tensor
        return A.Compose([ToTensorV2()])


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Segmentation.

    Loads satellite imagery bands, constructs an Ash False Color Composite,
    computes temporal differences, and loads binary segmentation masks.
    """

    def __init__(
        self, split="train", transform=None, debug=False, load_cached_data=False
    ):
        """
        Args:
            split (str): One of 'train', 'validation', 'test'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, limits the dataset size for debugging.
            load_cached_data (bool): Placeholder for caching logic (not used for per-image loading).
        """
        self.split = split
        self.transform = transform
        self.debug = debug

        # Determine metadata file path
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "validation":
            meta_path = Config.VALIDATION_METADATA_PATH
        elif split == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load metadata
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.df = pd.read_csv(meta_path)

        # Convert record_id to string to ensure consistent path handling
        self.df["record_id"] = self.df["record_id"].astype(str)

        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        # Ash Composite Normalization Bounds (approximate for GOES-16)
        # Red: T15 - T14
        self.ash_r_min, self.ash_r_max = -4.0, 2.0
        # Green: T14 - T11
        self.ash_g_min, self.ash_g_max = -4.0, 5.0
        # Blue: T14
        self.ash_b_min, self.ash_b_max = 243.0, 303.0

    def __len__(self):
        return len(self.df)

    def normalize_range(self, data, vmin, vmax):
        """
        Normalizes data to [0, 1] based on vmin and vmax.
        """
        return (data - vmin) / (vmax - vmin)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = row["record_id"]

        # ----------------------------------------------------------------
        # 1. Load Bands
        # ----------------------------------------------------------------
        # We need bands 11, 14, 15.
        # The metadata columns are named 'band_11', 'band_14', 'band_15'.

        bands_data = {}
        for band_idx in Config.USED_BANDS:
            col_name = f"band_{band_idx:02d}"
            rel_path = row[col_name]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            try:
                # Shape: H x W x T (T=8)
                # T indices: 0..3 (before), 4 (current), 5..7 (after)
                data = np.load(full_path).astype(np.float32)
                bands_data[band_idx] = data
            except Exception as e:
                # Fallback for missing files (should not happen given metadata validation)
                # Create dummy data of correct shape
                print(f"Error loading {full_path}: {e}")
                bands_data[band_idx] = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8), dtype=np.float32
                )

        # ----------------------------------------------------------------
        # 2. Extract Temporal Frames
        # ----------------------------------------------------------------
        # t_curr = 4 (The labeled frame)
        # t_prev = 3 (The frame 10 mins before)
        t_curr = Config.N_TIMES_BEFORE
        t_prev = t_curr - 1

        # Extract specific bands at specific times
        # Band 11 (8.4um), Band 14 (11.2um), Band 15 (12.3um)

        t11_curr = bands_data[11][:, :, t_curr]
        t14_curr = bands_data[14][:, :, t_curr]
        t15_curr = bands_data[15][:, :, t_curr]

        t11_prev = bands_data[11][:, :, t_prev]
        t14_prev = bands_data[14][:, :, t_prev]
        t15_prev = bands_data[15][:, :, t_prev]

        # ----------------------------------------------------------------
        # 3. Construct Input Tensor (6 Channels)
        # ----------------------------------------------------------------

        # --- Channels 1-3: Ash False Color Composite (at t_curr) ---
        # Red: T15 - T14
        r_ch = self.normalize_range(t15_curr - t14_curr, self.ash_r_min, self.ash_r_max)
        # Green: T14 - T11
        g_ch = self.normalize_range(t14_curr - t11_curr, self.ash_g_min, self.ash_g_max)
        # Blue: T14
        b_ch = self.normalize_range(t14_curr, self.ash_b_min, self.ash_b_max)

        # Clip to [0, 1]
        ash_composite = np.stack([r_ch, g_ch, b_ch], axis=-1)
        ash_composite = np.clip(ash_composite, 0, 1)

        # --- Channels 4-6: Raw Band Differences (t_curr - t_prev) ---
        # We use raw values to preserve dynamic range of cooling
        diff_11 = t11_curr - t11_prev
        diff_14 = t14_curr - t14_prev
        diff_15 = t15_curr - t15_prev

        temporal_diffs = np.stack([diff_11, diff_14, diff_15], axis=-1)

        # Concatenate to form 6-channel input
        # Shape: (H, W, 6)
        image = np.concatenate([ash_composite, temporal_diffs], axis=-1)

        # ----------------------------------------------------------------
        # 4. Load Mask (if available)
        # ----------------------------------------------------------------
        mask = None
        if self.split in ["train", "validation"]:
            mask_rel_path = row["human_pixel_masks"]
            mask_full_path = os.path.join(Config.INPUT_DIR, mask_rel_path)
            try:
                # Shape: H x W x 1
                mask = np.load(mask_full_path).astype(np.float32)
                # Squeeze to H x W for Albumentations
                mask = mask.squeeze(-1)
            except Exception as e:
                print(f"Error loading mask {mask_full_path}: {e}")
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
                )
        else:
            # Dummy mask for test
            mask = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

        # ----------------------------------------------------------------
        # 5. Apply Augmentations
        # ----------------------------------------------------------------
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Albumentations ToTensorV2 converts image to (C, H, W)
            # and mask to Tensor.

            # Ensure mask has channel dimension (1, H, W) for BCE loss
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            # Manual conversion if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {"image": image, "mask": mask, "record_id": record_id}
