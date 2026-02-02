import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_transforms(mode="train"):
    """
    Returns Albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'val'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=Config.AUG_HORIZONTAL_FLIP_P),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=Config.AUG_SHIFT_SCALE_ROTATE_P,
                ),
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=None,  # alpha_affine is deprecated/separate in newer versions, but keeping simple
                    p=Config.AUG_ELASTIC_P,
                ),
                # Normalization is handled manually (div by 255) to keep control over 1-channel
            ]
        )
    else:
        return A.Compose([])


def pad_image(img, target_size=128):
    """
    Pads an image to the target size using reflection padding.
    Assumes input is (H, W) or (H, W, C).
    """
    h, w = img.shape[:2]
    delta_h = target_size - h
    delta_w = target_size - w

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Use reflection padding to minimize boundary artifacts
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    # If the image was 2D (H, W), copyMakeBorder returns (H, W).
    # If we need to preserve channel dim for albumentations later, we handle it in dataset.
    return padded


def load_data(mode="train", load_cached_data=True):
    """
    Loads data from disk or cache.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from .npy cache.

    Returns:
        tuple: (images, masks, depths) for train/val, (images, depths) for test.
               images: np.array of shape (N, 128, 128)
               masks: np.array of shape (N, 128, 128) (only for train/val)
               depths: np.array of shape (N,)
    """
    # Define paths based on mode
    if mode == "train":
        csv_path = Config.TRAIN_METADATA_PATH
        cache_img = Config.CACHE_TRAIN_IMAGES
        cache_mask = Config.CACHE_TRAIN_MASKS
        cache_depth = Config.CACHE_TRAIN_DEPTHS
    elif mode == "val":
        csv_path = Config.VAL_METADATA_PATH
        cache_img = Config.CACHE_VAL_IMAGES
        cache_mask = Config.CACHE_VAL_MASKS
        cache_depth = Config.CACHE_VAL_DEPTHS
    elif mode == "test":
        csv_path = Config.TEST_METADATA_PATH
        cache_img = Config.CACHE_TEST_IMAGES
        cache_depth = Config.CACHE_TEST_DEPTHS
        cache_mask = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if mode == "test":
            if os.path.exists(cache_img) and os.path.exists(cache_depth):
                print(f"Loading {mode} data from cache...")
                images = np.load(cache_img)
                depths = np.load(cache_depth)
                return images, depths
        else:
            if (
                os.path.exists(cache_img)
                and os.path.exists(cache_mask)
                and os.path.exists(cache_depth)
            ):
                print(f"Loading {mode} data from cache...")
                images = np.load(cache_img)
                masks = np.load(cache_mask)
                depths = np.load(cache_depth)
                return images, masks, depths

    # If not cached, process from scratch
    print(f"Processing {mode} data from source...")
    df = pd.read_csv(csv_path)

    images_list = []
    masks_list = []
    depths_list = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Pad Image
        img_padded = pad_image(img, Config.IMG_TARGET_SIZE)
        images_list.append(img_padded)

        # Load Depth
        depths_list.append(row["z"])

        # Load Mask (if applicable)
        if mode != "test":
            mask_path = os.path.join(Config.INPUT_ROOT, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Should not happen based on metadata checks, but good for safety
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            # Pad Mask
            mask_padded = pad_image(mask, Config.IMG_TARGET_SIZE)
            # Binarize just in case (0 or 255 -> 0 or 1)
            mask_padded = (mask_padded > 127).astype(np.uint8)
            masks_list.append(mask_padded)

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    depths = np.array(depths_list, dtype=np.float32)

    # Save to cache
    np.save(cache_img, images)
    np.save(cache_depth, depths)

    if mode != "test":
        masks = np.array(masks_list, dtype=np.uint8)
        np.save(cache_mask, masks)
        return images, masks, depths

    return images, depths


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles normalization and augmentation.
    """

    def __init__(
        self, images, depths, masks=None, transform=None, depth_mean=0.0, depth_std=1.0
    ):
        """
        Args:
            images (np.array): Shape (N, H, W), uint8.
            depths (np.array): Shape (N,), float.
            masks (np.array, optional): Shape (N, H, W), uint8.
            transform (A.Compose, optional): Albumentations transforms.
            depth_mean (float): Mean for depth normalization.
            depth_std (float): Std for depth normalization.
        """
        self.images = images
        self.depths = depths
        self.masks = masks
        self.transform = transform
        self.depth_mean = depth_mean
        self.depth_std = depth_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.images[idx]  # (H, W)
        depth = self.depths[idx]

        # Normalize depth
        depth_norm = (depth - self.depth_mean) / self.depth_std

        if self.masks is not None:
            mask = self.masks[idx]  # (H, W)

            # Apply transforms
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

            # Convert to Tensor
            # Image: (H, W) -> (1, H, W), float [0, 1]
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

            # Mask: (H, W) -> (1, H, W), float [0, 1]
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

            # Depth: (1,)
            depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

            return img_tensor, mask_tensor, depth_tensor

        else:
            # Test mode (no mask)
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented["image"]

            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
            depth_tensor = torch.tensor([depth_norm], dtype=torch.float32)

            return img_tensor, depth_tensor


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    Calculates depth stats from Training set and applies to all.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Data Arrays
    train_images, train_masks, train_depths = load_data("train", load_cached_data)
    val_images, val_masks, val_depths = load_data("val", load_cached_data)
    test_images, test_depths = load_data("test", load_cached_data)

    # 2. Calculate Depth Statistics from Training Set
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)

    # Avoid division by zero (unlikely here given depth range)
    if depth_std == 0:
        depth_std = 1.0

    print(f"Depth Normalization Stats - Mean: {depth_mean:.4f}, Std: {depth_std:.4f}")

    # 3. Create Datasets
    train_dataset = SaltDataset(
        train_images,
        train_depths,
        train_masks,
        transform=get_transforms("train"),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    val_dataset = SaltDataset(
        val_images,
        val_depths,
        val_masks,
        transform=get_transforms("val"),
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    test_dataset = SaltDataset(
        test_images,
        test_depths,
        masks=None,
        transform=get_transforms("val"),  # No geometric augs for test
        depth_mean=depth_mean,
        depth_std=depth_std,
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
