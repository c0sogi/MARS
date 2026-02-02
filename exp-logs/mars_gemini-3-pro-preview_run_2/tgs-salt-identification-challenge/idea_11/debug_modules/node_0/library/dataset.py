import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
INPUT_ROOT = "./input"
METADATA_ROOT = "./metadata"
CACHE_DIR = "./working/idea_11"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128

# ImageNet stats for 1 channel (Average of RGB means/stds)
# Mean: (0.485 + 0.456 + 0.406) / 3 = 0.449
# Std: (0.229 + 0.224 + 0.225) / 3 = 0.226
MEAN_1CH = [0.449]
STD_1CH = [0.226]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Spatial Alignment: Pad to 128x128 using reflection
                A.PadIfNeeded(
                    min_height=IMG_SIZE_TARGET,
                    min_width=IMG_SIZE_TARGET,
                    border_mode=cv2.BORDER_REFLECT_101,
                    always_apply=True,
                ),
                # Non-Rigid Augmentation: Elastic Transform
                # Simulates organic salt plasticity
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.2),
                # Rigid Augmentation: ShiftScaleRotate
                # Ensures geometric invariance
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                # Normalization
                A.Normalize(mean=MEAN_1CH, std=STD_1CH),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test pipeline
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_SIZE_TARGET,
                    min_width=IMG_SIZE_TARGET,
                    border_mode=cv2.BORDER_REFLECT_101,
                    always_apply=True,
                ),
                A.Normalize(mean=MEAN_1CH, std=STD_1CH),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation task.
    Handles loading, caching, preprocessing, and augmentation.
    """

    def __init__(
        self, mode="train", transform=None, load_cached=True, cache_dir=CACHE_DIR
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline. If None, uses default.
            load_cached (bool): Whether to try loading from cache.
            cache_dir (str): Directory to store/load cached .npy files.
        """
        self.mode = mode
        self.transform = transform if transform is not None else get_transforms(mode)
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Determine metadata file
        if mode == "train":
            self.meta_file = os.path.join(METADATA_ROOT, "train.csv")
        elif mode == "val":
            self.meta_file = os.path.join(METADATA_ROOT, "val.csv")
        else:
            self.meta_file = os.path.join(METADATA_ROOT, "test.csv")

        # Load data (images, masks, depths, ids)
        self.ids, self.images, self.masks, self.depths = self._load_data(load_cached)

        # Calculate Depth Statistics from Training Data for Normalization
        # We always calculate this from the training metadata to ensure consistency across splits
        self.depth_mean, self.depth_std = self._get_train_depth_stats()

    def _get_train_depth_stats(self):
        """Calculates mean and std of depths from the training set metadata."""
        train_meta_path = os.path.join(METADATA_ROOT, "train.csv")
        if os.path.exists(train_meta_path):
            df = pd.read_csv(train_meta_path)
            return df["z"].mean(), df["z"].std()
        else:
            # Fallback if metadata missing (should not happen based on task desc)
            return 0.0, 1.0

    def _load_data(self, load_cached):
        """
        Loads data from cache or processes from raw files.
        """
        # Define cache filenames
        cache_prefix = os.path.join(self.cache_dir, self.mode)
        f_ids = f"{cache_prefix}_ids.npy"
        f_images = f"{cache_prefix}_images.npy"
        f_masks = f"{cache_prefix}_masks.npy"
        f_depths = f"{cache_prefix}_depths.npy"

        # Check if all cache files exist
        cache_exists = (
            os.path.exists(f_ids)
            and os.path.exists(f_images)
            and os.path.exists(f_depths)
            and (self.mode == "test" or os.path.exists(f_masks))
        )

        if load_cached and cache_exists:
            # Load from cache
            ids = np.load(f_ids, allow_pickle=True)
            images = np.load(f_images)
            depths = np.load(f_depths)
            masks = np.load(f_masks) if self.mode != "test" else None
            return ids, images, masks, depths

        # Process from scratch
        df = pd.read_csv(self.meta_file)

        ids = df["id"].values
        depths = df["z"].values.astype(np.float32)

        images_list = []
        masks_list = []

        for idx, row in df.iterrows():
            # Load Image
            img_path = os.path.join(INPUT_ROOT, row["image_path"])
            # Load as Grayscale (1 channel)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            images_list.append(img)

            # Load Mask (if not test)
            if self.mode != "test":
                mask_path = os.path.join(INPUT_ROOT, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                masks_list.append(mask)

        # Convert to numpy arrays
        # Shape: (N, 101, 101)
        images = np.array(images_list, dtype=np.uint8)

        if self.mode != "test":
            masks = np.array(masks_list, dtype=np.uint8)
        else:
            masks = None

        # Save to cache
        np.save(f_ids, ids)
        np.save(f_images, images)
        np.save(f_depths, depths)
        if masks is not None:
            np.save(f_masks, masks)

        return ids, images, masks, depths

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape (101, 101)
        depth_val = self.depths[idx]
        id_code = self.ids[idx]

        # Expand image dim for Albumentations/Torch: (H, W) -> (H, W, 1)
        image = np.expand_dims(image, axis=-1)

        mask = None
        if self.mode != "test":
            mask = self.masks[idx]  # Shape (101, 101)
            # Mask does not strictly need expansion for Albumentations if passed as 'mask'
            # but we will handle it carefully.

        # Apply Transforms
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # Tensor (1, 128, 128)
            mask = augmented[
                "mask"
            ]  # Tensor (128, 128) or (128, 128, 1) depending on input

            # Ensure mask is float tensor (1, 128, 128)
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)  # HWC -> CHW
            else:
                # Fallback if not tensor
                mask = torch.from_numpy(mask).float().unsqueeze(0)

            # Normalize mask to [0, 1]
            mask = mask / 255.0

        else:
            augmented = self.transform(image=image)
            image = augmented["image"]
            # Create dummy mask for test
            mask = torch.zeros(
                (1, IMG_SIZE_TARGET, IMG_SIZE_TARGET), dtype=torch.float32
            )

        # Depth Processing
        # 1. Normalize Depth (Standard Scaling)
        d = (depth_val - self.depth_mean) / self.depth_std

        # 2. Bernoulli Depth Masking (Train) & Imputation (Test)
        if self.mode == "train":
            # With p=0.5, replace depth with mean (0.0)
            if np.random.rand() < 0.5:
                d = 0.0
        elif self.mode == "test":
            # For test, always inject 0 (mean depth)
            d = 0.0

        # Convert depth to tensor
        depth_tensor = torch.tensor([d], dtype=torch.float32)

        return image, mask, depth_tensor, id_code
