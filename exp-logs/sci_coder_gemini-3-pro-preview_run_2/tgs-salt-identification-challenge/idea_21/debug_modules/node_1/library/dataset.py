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


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles padding, augmentation, and normalization.
    """

    def __init__(self, images, masks=None, depths=None, transform=None, mode="train"):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (H, W)
        image = self.images[idx]

        # Pad image to 128x128 (Reflection padding)
        # pad_image handles (H, W) -> (128, 128)
        image = pad_image(image, target_size=Config.IMG_SIZE)

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            # Pad mask
            mask = pad_image(mask, target_size=Config.IMG_SIZE)

        # Get depth
        depth = 0.0
        if self.depths is not None:
            depth = self.depths[idx]

        # Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Ensure channel dimension for mask (H, W) -> (1, H, W)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()

        # Depth to tensor
        depth = torch.tensor(depth, dtype=torch.float32)

        if self.mode == "test":
            return image, depth
        else:
            return image, mask, depth


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Mean and Std for 1-channel input (Approximation of ImageNet intensity)
    # Since we sum RGB weights in the model, we treat the 1-channel input
    # as having similar intensity distribution to standard RGB images.
    mean = [0.45]
    std = [0.225]

    if phase == "train":
        return A.Compose(
            [
                # Elastic Transform for salt plasticity
                # Note: alpha_affine is removed as it's deprecated/separate in newer albumentations
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    p=0.5,
                ),
                # Rigid geometric transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_SHIFT_SCALE_ROTATE_P,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only Normalize and ToTensor
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def load_data_split(csv_path, mode, load_cached_data=True):
    """
    Loads data from CSV or Cache.
    Strictly follows the caching logic:
    1. If cached and load_cached_data=True, load from .npy.
    2. Else, process from scratch and save to .npy.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache = os.path.join(cache_dir, f"{mode}_images.npy")
    mask_cache = os.path.join(cache_dir, f"{mode}_masks.npy")
    depth_cache = os.path.join(cache_dir, f"{mode}_depths.npy")
    id_cache = os.path.join(cache_dir, f"{mode}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        # Check existence
        files_exist = (
            os.path.exists(img_cache)
            and os.path.exists(depth_cache)
            and os.path.exists(id_cache)
        )

        if mode != "test":
            files_exist = files_exist and os.path.exists(mask_cache)

        if files_exist:
            # print(f"Loading {mode} data from cache...")
            images = np.load(img_cache)
            depths = np.load(depth_cache)
            ids = np.load(id_cache)
            if mode != "test":
                masks = np.load(mask_cache)
            else:
                masks = None
            return ids, images, masks, depths

    # Process from scratch
    # print(f"Processing {mode} data from scratch...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    ids_list = []
    images_list = []
    masks_list = []
    depths_list = []

    for _, row in df.iterrows():
        ids_list.append(row["id"])

        # Load Image (Grayscale)
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for safety, though check script ensures existence
            img = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
        images_list.append(img)

        # Load Depth
        depths_list.append(row["z"])

        # Load Mask (Train/Val only)
        if mode != "test":
            rle = row["rle_mask"] if "rle_mask" in row else None
            if pd.isna(rle):
                mask = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
            else:
                mask = rle_decode(rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE))
            masks_list.append(mask)

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    depths = np.array(depths_list, dtype=np.float32)
    ids = np.array(ids_list)

    if mode != "test":
        masks = np.array(masks_list, dtype=np.uint8)
    else:
        masks = None

    # Save to cache
    np.save(img_cache, images)
    np.save(depth_cache, depths)
    np.save(id_cache, ids)
    if masks is not None:
        np.save(mask_cache, masks)

    return ids, images, masks, depths


def get_dataloaders(load_cached_data=True):
    """
    Prepares Train and Validation DataLoaders.
    Calculates depth statistics from the training set for standardization.
    """
    # Load Data
    train_ids, train_imgs, train_masks, train_depths = load_data_split(
        Config.TRAIN_CSV, "train", load_cached_data
    )
    val_ids, val_imgs, val_masks, val_depths = load_data_split(
        Config.VAL_CSV, "val", load_cached_data
    )

    # Calculate Depth Statistics (Standard Scale)
    d_mean = train_depths.mean()
    d_std = train_depths.std() + 1e-8

    # Standardize Depths
    train_depths = (train_depths - d_mean) / d_std
    val_depths = (val_depths - d_mean) / d_std

    # Create Datasets
    train_ds = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        transform=get_transforms("train"),
        mode="train",
    )
    val_ds = SaltDataset(
        val_imgs, val_masks, val_depths, transform=get_transforms("val"), mode="val"
    )

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, (d_mean, d_std)


def get_test_loader(depth_stats, load_cached_data=True):
    """
    Prepares Test DataLoader.
    Uses depth statistics provided from training set.
    """
    test_ids, test_imgs, _, test_depths = load_data_split(
        Config.TEST_CSV, "test", load_cached_data
    )

    # Standardize Depths using training stats
    d_mean, d_std = depth_stats
    test_depths = (test_depths - d_mean) / d_std

    test_ds = SaltDataset(
        test_imgs, None, test_depths, transform=get_transforms("test"), mode="test"
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_ids
