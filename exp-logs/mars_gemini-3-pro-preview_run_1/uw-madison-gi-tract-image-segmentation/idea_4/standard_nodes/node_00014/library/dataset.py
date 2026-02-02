import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(*Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_SIZE[0] // 20,
                    max_width=Config.IMG_SIZE[1] // 20,
                    min_holes=5,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize only
        return A.Compose(
            [A.Resize(*Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR), ToTensorV2()]
        )


class UWDataset(Dataset):
    """
    PyTorch Dataset for Stomach and Intestines MRI Segmentation.
    Handles image loading, robust normalization, 2D->3D channel replication,
    and RLE mask decoding.
    """

    def __init__(self, df, mode="train", transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file paths, RLEs, etc.).
            mode (str): 'train', 'val', or 'test'.
            transforms (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.mode = mode
        self.transforms = transforms
        self.input_dir = Config.INPUT_DIR
        self.classes = Config.CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # Images are 16-bit PNGs, load as-is
        img_path = os.path.join(self.input_dir, row["file_path"])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback for missing files (should not happen based on metadata check)
            # Create a dummy image of correct size
            img = np.zeros((row["height"], row["width"]), dtype=np.uint16)

        # Ensure image is float32 for processing
        img = img.astype(np.float32)

        # 2. Robust Percentile Normalization
        # Clip pixel intensities to [1st, 99th] percentiles to handle outliers
        p_lo = np.percentile(img, Config.NORM_MIN_PERCENTILE)
        p_hi = np.percentile(img, Config.NORM_MAX_PERCENTILE)

        # Avoid division by zero if image is completely flat
        if p_hi > p_lo:
            img = np.clip(img, p_lo, p_hi)
            img = (img - p_lo) / (p_hi - p_lo)
        else:
            if p_hi > 0:
                img = img / p_hi
            else:
                img = np.zeros_like(img)

        # 3. Channel Replication (2D -> 3D)
        # ResNet expects 3 input channels. We replicate the single slice z -> (z, z, z).
        # Shape becomes (H, W, 3)
        img = np.tile(img[..., None], (1, 1, 3))

        # 4. Load Masks (if not test mode)
        if self.mode != "test":
            h, w = row["height"], row["width"]
            masks = []
            for class_name in self.classes:
                rle = row[class_name]
                mask = rle_decode(rle, shape=(h, w))
                masks.append(mask)

            # Stack masks: (H, W, Num_Classes)
            mask_stack = np.stack(masks, axis=-1).astype(np.float32)

            # 5. Augmentations
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask_stack)
                img = augmented["image"]
                mask_stack = augmented["mask"]

            # Mask is already (C, H, W) after ToTensorV2 if it was (H, W, C) input?
            # Albumentations ToTensorV2 converts image to (C, H, W) but usually keeps mask as (H, W, C)
            # unless transpose_mask=True is set (default is False).
            # However, PyTorch models expect (N, C, H, W).
            # Let's check ToTensorV2 behavior: it converts image to tensor (C, H, W).
            # For masks, it converts to tensor but preserves dimensions unless changed.
            # We explicitly permute mask to (C, H, W).

            if mask_stack.shape[0] != len(self.classes):
                mask_stack = mask_stack.permute(2, 0, 1)

            return {"image": img, "mask": mask_stack, "id": row["id"]}

        else:
            # Test mode - no masks
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented["image"]

            return {"image": img, "id": row["id"]}
