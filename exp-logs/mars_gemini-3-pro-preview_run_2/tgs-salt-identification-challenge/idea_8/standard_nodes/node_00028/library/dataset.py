import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMG_SIZE,
    ORIG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    AUG_PROB,
    ELASTIC_ALPHA,
    ELASTIC_SIGMA,
    ELASTIC_ALPHA_AFFINE,
    SEED,
)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Non-Rigid Augmentation: Elastic Transform
                A.ElasticTransform(
                    alpha=ELASTIC_ALPHA,
                    sigma=ELASTIC_SIGMA,
                    alpha_affine=ELASTIC_ALPHA_AFFINE,
                    p=AUG_PROB,
                ),
                # Rigid Augmentation: ShiftScaleRotate
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=AUG_PROB
                ),
                # Flip
                A.HorizontalFlip(p=0.5),
                # Normalization
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Only Normalize
        return A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


def pad_image(img, target_size=128):
    """
    Pads an image to the target size using reflection padding.
    Assumes input image is (H, W) or (H, W, C).
    """
    h, w = img.shape[:2]
    diff_h = target_size - h
    diff_w = target_size - w

    if diff_h < 0 or diff_w < 0:
        # Resize if larger (though dataset is fixed at 101)
        return cv2.resize(img, (target_size, target_size))

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # Reflection padding handles borders smoothly for segmentation
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )
    return padded


class SaltDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        mode="train",
        depth_stats=None,
        load_cached_data=True,
        transform=None,
    ):
        """
        Args:
            metadata_path (str): Path to the csv file (train.csv, val.csv, test.csv).
            mode (str): 'train', 'val', or 'test'.
            depth_stats (tuple): (mean, std) of depth from training set. Required for train/val.
            load_cached_data (bool): Whether to use .npy caching.
            transform (A.Compose): Albumentations transforms.
        """
        self.mode = mode
        self.transform = transform
        self.depth_stats = depth_stats

        # Load metadata
        self.df = pd.read_csv(metadata_path)
        self.ids = self.df["id"].values

        # Determine cache filenames based on mode/metadata filename
        # We use the basename of metadata_path to distinguish caches (train/val/test)
        meta_name = os.path.splitext(os.path.basename(metadata_path))[0]
        cache_dir = WORKING_DIR

        self.images_cache_path = os.path.join(cache_dir, f"{meta_name}_images.npy")
        self.masks_cache_path = os.path.join(cache_dir, f"{meta_name}_masks.npy")
        self.depths_cache_path = os.path.join(cache_dir, f"{meta_name}_depths.npy")

        # Load or Process Data
        self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Handles caching logic: Load if exists and requested, else process and save.
        """
        # Check if cache exists
        cache_exists = (
            os.path.exists(self.images_cache_path)
            and os.path.exists(self.depths_cache_path)
            and (self.mode == "test" or os.path.exists(self.masks_cache_path))
        )

        if load_cached_data and cache_exists:
            # Load from cache
            self.images = np.load(self.images_cache_path)
            self.depths = np.load(self.depths_cache_path)
            if self.mode != "test":
                self.masks = np.load(self.masks_cache_path)
            else:
                self.masks = None
        else:
            # Process from scratch
            self._process_and_cache()

    def _process_and_cache(self):
        """
        Reads images/masks from disk, pads them, and saves to .npy files.
        """
        images_list = []
        masks_list = []
        depths_list = []

        for idx, row in self.df.iterrows():
            # Read Image
            img_path = os.path.join(INPUT_DIR, row["image_path"])
            # Read as grayscale (101, 101)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            # Pad Image
            img_padded = pad_image(img, target_size=IMG_SIZE)
            images_list.append(img_padded)

            # Read Mask (if not test)
            if self.mode != "test":
                mask_path = os.path.join(INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    # Fallback for missing masks if any (though dataset should be clean)
                    mask = np.zeros_like(img)

                # Pad Mask
                mask_padded = pad_image(mask, target_size=IMG_SIZE)
                # Binarize just in case interpolation happened (though reflection shouldn't change values)
                mask_padded = (mask_padded > 127).astype(np.uint8)
                masks_list.append(mask_padded)

            # Depth
            depths_list.append(row["z"])

        # Convert to numpy arrays
        self.images = np.array(images_list, dtype=np.uint8)
        self.depths = np.array(depths_list, dtype=np.float32)

        if self.mode != "test":
            self.masks = np.array(masks_list, dtype=np.uint8)

        # Save to cache
        np.save(self.images_cache_path, self.images)
        np.save(self.depths_cache_path, self.depths)
        if self.mode != "test":
            np.save(self.masks_cache_path, self.masks)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get Image
        # Shape: (128, 128)
        img = self.images[idx]

        # Convert to 3 channels (Grayscale -> RGB)
        # Shape: (128, 128, 3)
        img = np.repeat(img[..., np.newaxis], 3, axis=2)

        # 2. Get Mask
        if self.mode != "test":
            mask = self.masks[idx]
        else:
            # Dummy mask for test
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # 3. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            # Fallback if no transform provided
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()

        # Ensure mask has channel dimension (1, H, W) for training
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Mask should be float for BCE/Lovasz combination usually, or handled by loss
        mask_tensor = mask_tensor.float()

        # 4. Handle Depth
        # Logic:
        # - Train/Val: Standardize using training stats.
        # - Test: Feed constant 0 (mean of standardized training depths).

        if self.mode == "test":
            z_norm = 0.0
        else:
            z = self.depths[idx]
            if self.depth_stats:
                mean, std = self.depth_stats
                z_norm = (z - mean) / std
            else:
                # Fallback (should not happen if used correctly)
                z_norm = 0.0

        # Convert depth to tensor
        depth_tensor = torch.tensor([z_norm], dtype=torch.float32)

        return img_tensor, mask_tensor, depth_tensor, self.ids[idx]


def get_depth_stats(train_metadata_path):
    """
    Calculates mean and std of depth from the training metadata.
    This ensures consistency across train/val/test splits.
    """
    df = pd.read_csv(train_metadata_path)
    z = df["z"].values
    return z.mean(), z.std()
