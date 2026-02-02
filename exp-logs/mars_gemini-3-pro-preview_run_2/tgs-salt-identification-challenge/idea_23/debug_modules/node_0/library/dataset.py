import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'valid', 'test', or 'student'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Base padding to 128x128 with reflection (stride 32 compatibility)
    # Original is 101x101. 128 - 101 = 27 pixels padding total.
    base_transforms = [
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )
    ]

    # Normalization (ImageNet stats) and Tensor conversion
    post_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    if phase == "train":
        # Teacher Augmentations: Elastic + ShiftScaleRotate
        aug_transforms = [
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
            ),
            A.HorizontalFlip(p=0.5),
        ]
        return A.Compose(base_transforms + aug_transforms + post_transforms)

    elif phase == "student":
        # Student Augmentations: Stronger distortions for consistency training
        aug_transforms = [
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
        ]
        return A.Compose(base_transforms + aug_transforms + post_transforms)

    else:  # valid or test
        # Deterministic
        return A.Compose(base_transforms + post_transforms)


def get_data_arrays(metadata_path, prefix, load_cached_data=True):
    """
    Loads dataset arrays (images, masks, depths, ids) from metadata CSV.
    Implements caching to .npy files in the working directory.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
            images: np.ndarray of shape (N, 101, 101) or (N, 101, 101, 3)
            masks: np.ndarray of shape (N, 101, 101) or None (for test)
            depths: np.ndarray of shape (N,)
            ids: np.ndarray of strings
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    img_cache_path = os.path.join(Config.CACHE_DIR, f"{prefix}_images.npy")
    mask_cache_path = os.path.join(Config.CACHE_DIR, f"{prefix}_masks.npy")
    depth_cache_path = os.path.join(Config.CACHE_DIR, f"{prefix}_depths.npy")
    id_cache_path = os.path.join(Config.CACHE_DIR, f"{prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(depth_cache_path)
            and os.path.exists(id_cache_path)
        ):
            # Check mask cache only if it's expected (not test)
            if "test" in prefix and not os.path.exists(mask_cache_path):
                pass  # Test set might not have masks
            elif not os.path.exists(mask_cache_path) and "test" not in prefix:
                # Missing masks for train/val
                pass
            else:
                # print(f"Loading {prefix} data from cache...")
                images = np.load(img_cache_path)
                depths = np.load(depth_cache_path)
                ids = np.load(id_cache_path)
                masks = None
                if os.path.exists(mask_cache_path):
                    masks = np.load(mask_cache_path)
                return images, masks, depths, ids

    # 2. Process from scratch
    # print(f"Processing {prefix} data from metadata...")
    df = pd.read_csv(metadata_path)

    ids = df["id"].values
    depths = df["z"].values

    images = []
    masks = []
    has_masks = "rle_mask" in df.columns

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        # Read as grayscale since we sum channels later or use 1 channel
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)

        # Load Mask if available
        if has_masks:
            rle = row["rle_mask"]
            mask = rle_decode(rle, shape=(101, 101))
            masks.append(mask)

    images = np.array(images, dtype=np.uint8)

    if has_masks:
        masks = np.array(masks, dtype=np.uint8)
    else:
        masks = None

    # Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)
    if masks is not None:
        np.save(mask_cache_path, masks)

    return images, masks, depths, ids


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles 101->128 padding, depth standardization, and Bernoulli depth masking.
    """

    def __init__(
        self,
        images,
        masks,
        depths,
        ids,
        transforms=None,
        depth_stats=None,
        depth_dropout_prob=0.0,
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            masks (np.ndarray, optional): Array of masks (N, H, W). Can be binary or soft.
            depths (np.ndarray): Array of depths (N,).
            ids (np.ndarray): Array of IDs (N,).
            transforms (A.Compose): Albumentations transforms.
            depth_stats (tuple): (mean, std) for depth standardization.
            depth_dropout_prob (float): Probability of replacing depth with 0 (mean).
                                        0.5 for Teacher Train, 1.0 for Test/Unlabeled, 0.0 for Valid.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transforms = transforms
        self.depth_dropout_prob = depth_dropout_prob

        # Depth standardization
        if depth_stats:
            self.depth_mean, self.depth_std = depth_stats
        else:
            # Default fallback if not provided (though should be provided)
            self.depth_mean = 0.0
            self.depth_std = 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx]
        depth_val = self.depths[idx]
        img_id = self.ids[idx]

        # Handle Mask
        if self.masks is not None:
            mask = self.masks[idx]
        else:
            # Create dummy mask for test set
            mask = np.zeros_like(image)

        # Apply Augmentations
        # Albumentations expects HWC image, so expand dims if grayscale
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
            # Repeat to 3 channels for ImageNet norm compatibility in Albumentations
            # But our model takes 1 channel sum.
            # Standard practice: Transform as RGB, then take first channel or sum.
            # Here we convert to RGB for transforms, then back to tensor.
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transforms:
            # Albumentations requires mask to be passed as 'mask'
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Image is now a Tensor (C, H, W) from ToTensorV2
        # If we used RGB for transforms, we need to convert back to 1 channel for the model
        # The model config says CHANNELS = 1.
        # We can sum the RGB channels to get a robust 1-channel input preserving pretrained weights info
        if image.shape[0] == 3:
            image = torch.sum(image, dim=0, keepdim=True)

        # Handle Depth
        # Standardize
        z = (depth_val - self.depth_mean) / self.depth_std

        # Bernoulli Masking / Dropout
        # If random float < prob, set z to 0 (mean)
        if np.random.random() < self.depth_dropout_prob:
            z = 0.0

        z = torch.tensor([z], dtype=torch.float32)

        # Handle Mask Tensor
        # If mask was transformed, it's a tensor. If not (no transforms), convert.
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        else:
            mask = mask.float()

        # Ensure mask is (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask, z, img_id
