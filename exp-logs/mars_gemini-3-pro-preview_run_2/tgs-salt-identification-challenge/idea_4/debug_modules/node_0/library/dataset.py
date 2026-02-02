import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode, pad_image


def get_transforms(phase: str):
    """
    Returns the Albumentations transformations for the specific phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if phase == "train":
        return A.Compose(
            [
                # Elastic Transform to simulate salt plasticity
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_SIGMA,
                    p=Config.AUG_PROB,
                ),
                # Geometric invariance
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_PROB,
                ),
                A.HorizontalFlip(p=0.5),
                # Normalize: divides by 255.0 then subtracts mean/divides std
                # Using ImageNet stats adapted for 1 channel (or just consistent scaling)
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just normalize and convert to tensor
        return A.Compose(
            [
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2(),
            ]
        )


def load_and_preprocess(df, mode, config):
    """
    Loads images, masks, and depths from the dataframe.
    Pads images/masks to config.INPUT_SIZE.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'val', or 'test'.
        config (Config): Configuration class.

    Returns:
        tuple: (images, masks, depths, ids) as numpy arrays.
    """
    images = []
    masks = []
    depths = []
    ids = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(config.INPUT_ROOT, row["image_path"])
        # Load as grayscale (H, W)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for safety, though data should exist
            img = np.zeros((config.ORIG_SIZE, config.ORIG_SIZE), dtype=np.uint8)

        # Pad Image to (128, 128)
        img = pad_image(img, config.INPUT_SIZE)

        # Expand dims for Albumentations (H, W, 1)
        img = np.expand_dims(img, axis=-1)

        images.append(img)
        depths.append(row["z"])
        ids.append(row["id"])

        if mode != "test":
            # Load Mask
            rle = row["rle_mask"] if pd.notna(row["rle_mask"]) else ""
            mask = rle_decode(rle, (config.ORIG_SIZE, config.ORIG_SIZE))

            # Pad Mask to (128, 128)
            mask = pad_image(mask, config.INPUT_SIZE)

            # Expand dims (H, W, 1)
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)  # (N, 128, 128, 1)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids)

    if mode != "test":
        masks = np.array(masks, dtype=np.uint8)  # (N, 128, 128, 1)
        return images, masks, depths, ids
    else:
        return images, None, depths, ids


def get_cached_data(df, mode, config, load_cached_data=True):
    """
    Handles caching logic for dataset arrays.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'val', or 'test'.
        config (Config): Configuration class.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
    """
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{mode}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"{mode}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"{mode}_depths.npy")
    id_cache_path = os.path.join(cache_dir, f"{mode}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        has_img = os.path.exists(img_cache_path)
        has_depth = os.path.exists(depth_cache_path)
        has_ids = os.path.exists(id_cache_path)
        has_mask = os.path.exists(mask_cache_path) if mode != "test" else True

        if has_img and has_depth and has_ids and has_mask:
            try:
                images = np.load(img_cache_path)
                depths = np.load(depth_cache_path)
                ids = np.load(id_cache_path, allow_pickle=True)
                masks = np.load(mask_cache_path) if mode != "test" else None
                return images, masks, depths, ids
            except Exception as e:
                print(f"Failed to load cache for {mode}: {e}. Recomputing...")

    # 2. Process from scratch
    images, masks, depths, ids = load_and_preprocess(df, mode, config)

    # 3. Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)
    if mode != "test":
        np.save(mask_cache_path, masks)

    return images, masks, depths, ids


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    """

    def __init__(self, images, masks, depths, ids, transforms=None):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.images[idx]  # (H, W, 1) uint8
        depth = self.depths[idx]  # float scalar

        data = {"image": img}
        if self.masks is not None:
            data["mask"] = self.masks[idx]  # (H, W, 1) uint8

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(**data)
            img = augmented["image"]  # Tensor (1, H, W)

            if self.masks is not None:
                mask = augmented["mask"]  # Tensor (1, H, W)
                # Convert mask to float for BCE/Lovasz loss
                mask = mask.float()
        else:
            # Manual fallback (should not be used given get_transforms)
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            if self.masks is not None:
                mask = torch.from_numpy(self.masks[idx].transpose(2, 0, 1)).float()

        # Convert depth to tensor
        depth = torch.tensor([depth], dtype=torch.float32)

        if self.masks is not None:
            return img, mask, depth, self.ids[idx]
        else:
            return img, depth, self.ids[idx]


def get_dataloaders(config, load_cached_data=True):
    """
    Factory function to create DataLoaders.
    Handles metadata loading, caching, and depth normalization.

    Args:
        config (Config): Configuration class.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Debug Mode: Truncate data
    if config.DEBUG:
        train_df = train_df.head(config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(config.DEBUG_SAMPLE_SIZE)

    # Load/Cache Data Arrays
    # This ensures we work with efficient numpy arrays instead of reading files every iteration
    train_imgs, train_masks, train_depths, train_ids = get_cached_data(
        train_df, "train", config, load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = get_cached_data(
        val_df, "val", config, load_cached_data
    )
    test_imgs, _, test_depths, test_ids = get_cached_data(
        test_df, "test", config, load_cached_data
    )

    # Depth Normalization (Standard Scaling)
    # Fit on TRAIN, Transform TRAIN/VAL/TEST
    d_mean = train_depths.mean()
    d_std = train_depths.std()

    # Avoid division by zero
    if d_std == 0:
        d_std = 1.0

    train_depths = (train_depths - d_mean) / d_std
    val_depths = (val_depths - d_mean) / d_std
    test_depths = (test_depths - d_mean) / d_std

    # Create Datasets
    train_ds = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transforms=get_transforms("train"),
    )
    val_ds = SaltDataset(
        val_imgs, val_masks, val_depths, val_ids, transforms=get_transforms("valid")
    )
    test_ds = SaltDataset(
        test_imgs, None, test_depths, test_ids, transforms=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
