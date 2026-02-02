import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(stage="train"):
    """
    Returns the Albumentations transform pipeline for the specified stage.

    Args:
        stage (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if stage == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Strict Affine Transformation: Rotation, Scale, Shift.
                # Elastic/Grid distortions are explicitly excluded.
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=4,  # cv2.BORDER_REFLECT_101
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation and Test: Only convert to Tensor
        return A.Compose([ToTensorV2(transpose_mask=True)])


class ContrailDataset(Dataset):
    """
    Dataset class for Contrail Identification.
    Implements Pure Raw-Physics Input Engineering and Homogeneous Scaling.
    """

    def __init__(self, metadata_path, stage="train", transform=None, cache_dir=None):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            stage (str): 'train', 'validation', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
            cache_dir (str, optional): Directory to save/load processed samples.
        """
        self.stage = stage
        self.transform = transform
        self.cache_dir = cache_dir

        # Load metadata
        self.df = pd.read_csv(metadata_path)

        # Debugging: Sample subset if configured
        if Config.DEBUG_SAMPLE_SIZE is not None:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Ensure cache directory exists if used
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def normalize(self, img, band_idx, is_diff=False):
        """
        Applies Min-Max Normalization to [0, 1] based on Config stats.

        Args:
            img (np.ndarray): Input image array.
            band_idx (int): Band ID (11, 14, 15).
            is_diff (bool): Whether the image is a temporal difference.

        Returns:
            np.ndarray: Normalized image.
        """
        if is_diff:
            stats = Config.DIFF_STATS[band_idx]
        else:
            stats = Config.BAND_STATS[band_idx]

        min_val = stats["min"]
        max_val = stats["max"]

        # Scale to [0, 1]
        img = (img - min_val) / (max_val - min_val)

        # Clip to ensure bounds
        return np.clip(img, 0, 1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # ---------------------------------------------------------------------
        # 1. Caching Logic
        # ---------------------------------------------------------------------
        cached_file = None
        if self.cache_dir:
            cached_file = os.path.join(self.cache_dir, f"{record_id}.npz")
            if os.path.exists(cached_file):
                try:
                    data = np.load(cached_file)
                    img = data["img"]
                    # Load mask if it exists in cache
                    mask = data["mask"] if "mask" in data else None

                    # If we are in train/val but mask is missing in cache (corrupt), fallback
                    if self.stage in ["train", "validation"] and mask is None:
                        raise ValueError("Mask missing in cache")

                    return self._apply_transform(img, mask)
                except Exception:
                    # Fallback to processing from scratch if cache load fails
                    pass

        # ---------------------------------------------------------------------
        # 2. Raw Data Loading & Physics Engineering
        # ---------------------------------------------------------------------
        # We need bands 11, 14, 15
        bands_data = {}
        for b in Config.BANDS:
            # Construct full path: input_dir + relative_path
            path = os.path.join(Config.INPUT_DIR, row[f"band_{b:02d}"])
            try:
                # Load NPY: Shape (H, W, T)
                # T=8 usually (4 before, 1 current, 3 after)
                bands_data[b] = np.load(path)
            except Exception as e:
                # Fallback for missing files (should not happen in valid dataset)
                # Return zeros
                print(f"Warning: Missing file {path}")
                img = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, Config.IN_CHANNELS),
                    dtype=np.float32,
                )
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)
                return self._apply_transform(img, mask)

        # Time indices
        t_curr = Config.N_TIMES_BEFORE  # 4
        t_prev = Config.N_TIMES_BEFORE - 1  # 3

        channels = []

        # A. Static Channels (T=4)
        for b in Config.BANDS:
            raw = bands_data[b][:, :, t_curr]
            norm = self.normalize(raw, b, is_diff=False)
            channels.append(norm)

        # B. Dynamic Channels (T=4 - T=3)
        for b in Config.BANDS:
            diff = bands_data[b][:, :, t_curr] - bands_data[b][:, :, t_prev]
            norm = self.normalize(diff, b, is_diff=True)
            channels.append(norm)

        # Stack to (H, W, 6)
        img = np.dstack(channels).astype(np.float32)

        # ---------------------------------------------------------------------
        # 3. Load Mask (if available)
        # ---------------------------------------------------------------------
        mask = None
        if self.stage in ["train", "validation"]:
            try:
                mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
                mask = np.load(mask_path).astype(np.float32)  # (H, W, 1)
            except Exception:
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # ---------------------------------------------------------------------
        # 4. Save to Cache
        # ---------------------------------------------------------------------
        if self.cache_dir and cached_file:
            save_dict = {"img": img}
            if mask is not None:
                save_dict["mask"] = mask
            np.savez_compressed(cached_file, **save_dict)

        return self._apply_transform(img, mask)

    def _apply_transform(self, img, mask):
        """
        Helper to apply augmentations and format output.
        """
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        else:
            # Manual ToTensor if no transform provided
            img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW
            if mask is not None:
                mask = torch.from_numpy(mask.transpose(2, 0, 1))  # HWC -> CHW

        # Handle Test Stage (No mask)
        if mask is None:
            # Return dummy mask for consistency
            # Shape: (1, H, W)
            mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.float32)

        # Ensure mask is (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[-1] == 1 and mask.shape[0] != 1:
            # Explicitly permute HWC -> CHW if needed
            mask = mask.permute(2, 0, 1)

        return img, mask
