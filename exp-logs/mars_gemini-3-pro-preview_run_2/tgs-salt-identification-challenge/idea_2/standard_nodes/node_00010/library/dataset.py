import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
CACHE_DIR = "./working/idea_2/"
INPUT_ROOT = "./input"


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms = []

    # Target dimensions for U-Net (must be divisible by 32)
    target_h, target_w = 128, 128

    if mode == "train":
        # Rigid Augmentation
        transforms.append(A.HorizontalFlip(p=0.5))

        # Non-Rigid Augmentation (Elastic Transform)
        # alpha=120, sigma=6, alpha_affine=0 as per requirements
        transforms.append(
            A.ElasticTransform(
                alpha=120,
                sigma=6,
                alpha_affine=0,
                p=0.2,
                border_mode=cv2.BORDER_REFLECT_101,
            )
        )

    # Padding (Common to all modes)
    # Pad from 101x101 to 128x128 using reflection
    transforms.append(
        A.PadIfNeeded(
            min_height=target_h,
            min_width=target_w,
            border_mode=cv2.BORDER_REFLECT_101,
            value=0,
            always_apply=True,
        )
    )

    # Normalization and Tensor Conversion
    # Normalize to [0, 1] by dividing by 255.0
    transforms.append(A.Normalize(mean=(0,), std=(1,), max_pixel_value=255.0))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class SaltDataset(Dataset):
    def __init__(self, mode, metadata_path, load_cached_data=True, transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            metadata_path (str): Path to the metadata CSV file.
            load_cached_data (bool): Whether to load data from .npy cache.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.mode = mode
        self.transform = transform
        self.df = pd.read_csv(metadata_path)

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Define cache file paths
        self.cache_images = os.path.join(CACHE_DIR, f"{mode}_images.npy")
        self.cache_masks = os.path.join(CACHE_DIR, f"{mode}_masks.npy")
        self.cache_depths = os.path.join(CACHE_DIR, f"{mode}_depths.npy")

        # Load data (either from cache or raw files)
        self.images, self.masks, self.depths = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Handles the logic for loading data from cache or processing from scratch.
        """
        # 1. Try to load from cache
        if load_cached_data:
            has_images = os.path.exists(self.cache_images)
            has_depths = os.path.exists(self.cache_depths)
            has_masks = (
                os.path.exists(self.cache_masks) if self.mode != "test" else True
            )

            if has_images and has_depths and has_masks:
                print(f"Loading {self.mode} data from cache...")
                images = np.load(self.cache_images)
                depths = np.load(self.cache_depths)
                masks = np.load(self.cache_masks) if self.mode != "test" else None
                return images, masks, depths

        # 2. Process from scratch
        print(f"Processing {self.mode} data from scratch...")
        images_list = []
        masks_list = []
        depths_list = []

        for idx, row in self.df.iterrows():
            # Load Image
            img_path = os.path.join(INPUT_ROOT, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            images_list.append(img)

            # Load Depth
            depths_list.append(row["z"])

            # Load Mask (if available)
            if self.mode != "test":
                mask_path = os.path.join(INPUT_ROOT, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                # Binarize mask (0 or 255 -> 0 or 1)
                mask = (mask > 127).astype(np.uint8)
                masks_list.append(mask)

        # Convert to numpy arrays
        images = np.array(images_list, dtype=np.uint8)  # (N, 101, 101)
        depths = np.array(depths_list, dtype=np.float32)  # (N,)

        if self.mode != "test":
            masks = np.array(masks_list, dtype=np.uint8)  # (N, 101, 101)
        else:
            masks = None

        # 3. Save to cache
        np.save(self.cache_images, images)
        np.save(self.cache_depths, depths)
        if masks is not None:
            np.save(self.cache_masks, masks)

        return images, masks, depths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve raw data
        image = self.images[idx]  # (101, 101)
        depth = self.depths[idx]  # Scalar

        # Expand dims to (101, 101, 1) for Albumentations/ToTensorV2 to produce (1, H, W)
        image = np.expand_dims(image, axis=2)

        # Normalize depth (simple scaling)
        # Depths range roughly 50-960. Dividing by 1000 keeps it in [0, 1].
        depth_tensor = torch.tensor([depth / 1000.0], dtype=torch.float32)

        image_id = self.df.iloc[idx]["id"]

        if self.mode != "test":
            mask = self.masks[idx]  # (101, 101)

            if self.transform:
                # Albumentations expects HWC or HW
                augmented = self.transform(image=image, mask=mask)
                image_tensor = augmented["image"]
                mask_tensor = augmented["mask"]
            else:
                # Fallback manual conversion
                image_tensor = (
                    torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                )
                mask_tensor = torch.from_numpy(mask).long()

            # Ensure mask is float for BCE/Dice loss
            if mask_tensor.dtype != torch.float32:
                mask_tensor = mask_tensor.float()

            return image_tensor, mask_tensor, depth_tensor, image_id

        else:
            if self.transform:
                augmented = self.transform(image=image)
                image_tensor = augmented["image"]
            else:
                image_tensor = (
                    torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                )

            return image_tensor, depth_tensor, image_id
