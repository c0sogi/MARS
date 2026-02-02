import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.utils import pad_image

# Constants
CACHE_DIR = "./working/idea_33/"
INPUT_DIR = "./input"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128
# ImageNet stats for 1 channel (approximate average of RGB means)
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, transform=None, depth_stats=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            masks (np.ndarray): Array of masks (N, H, W).
            depths (np.ndarray): Array of depths (N,).
            ids (np.ndarray): Array of ids (N,).
            transform (A.Compose): Albumentations transforms.
            depth_stats (tuple): (mean, std) for depth normalization.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.depth_stats = depth_stats

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        depth = self.depths[idx]
        img_id = self.ids[idx]

        # Apply Augmentations (Albumentations expects HWC or HW)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Pad to 128x128 using reflection padding
        image = pad_image(image, target_size=IMG_SIZE_TARGET)
        mask = pad_image(mask, target_size=IMG_SIZE_TARGET)

        # Normalize Image (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

        # Add channel dimension: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)

        # Mask to float tensor (1, H, W)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        # Normalize Depth
        if self.depth_stats:
            d_mean, d_std = self.depth_stats
            # Handle potential NaNs in test set by filling with mean
            if np.isnan(depth):
                depth = d_mean
            depth = (depth - d_mean) / (d_std + 1e-8)

        # Convert to Tensors
        image_t = torch.from_numpy(image).float()
        mask_t = torch.from_numpy(mask).float()
        depth_t = torch.tensor(depth).float()

        return image_t, mask_t, depth_t, img_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
            ]
        )
    else:
        # Validation/Test only requires padding/normalization which is done in Dataset
        return None


def load_data_from_metadata(metadata_path, load_cached_data=True, cache_name="train"):
    """
    Loads data from metadata CSV, caching the result to .npy files.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_path_imgs = os.path.join(CACHE_DIR, f"{cache_name}_images.npy")
    cache_path_masks = os.path.join(CACHE_DIR, f"{cache_name}_masks.npy")
    cache_path_depths = os.path.join(CACHE_DIR, f"{cache_name}_depths.npy")
    cache_path_ids = os.path.join(CACHE_DIR, f"{cache_name}_ids.npy")

    # Try to load from cache
    if (
        load_cached_data
        and os.path.exists(cache_path_imgs)
        and os.path.exists(cache_path_masks)
        and os.path.exists(cache_path_depths)
        and os.path.exists(cache_path_ids)
    ):

        images = np.load(cache_path_imgs)
        masks = np.load(cache_path_masks)
        depths = np.load(cache_path_depths)
        ids = np.load(cache_path_ids, allow_pickle=True)
        return images, masks, depths, ids

    # Process from scratch
    df = pd.read_csv(metadata_path)

    images = []
    masks = []
    depths = []
    ids = []

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_DIR, row["image_path"])
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Load Mask
        mask = np.zeros_like(img)
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = os.path.join(INPUT_DIR, row["mask_path"])
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mask = (m > 127).astype(np.uint8)

        # Load Depth (fill NaN with 0 for now, handled in Dataset)
        z = row["z"] if "z" in row and pd.notna(row["z"]) else 0.0

        images.append(img)
        masks.append(mask)
        depths.append(z)
        ids.append(row["id"])

    images = np.array(images)
    masks = np.array(masks)
    depths = np.array(depths)
    ids = np.array(ids)

    # Save to cache
    np.save(cache_path_imgs, images)
    np.save(cache_path_masks, masks)
    np.save(cache_path_depths, depths)
    np.save(cache_path_ids, ids)

    return images, masks, depths, ids


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Creates training and validation dataloaders.
    Calculates depth statistics from the training set.
    """
    train_meta = "./metadata/train.csv"
    val_meta = "./metadata/val.csv"

    # Load Data
    train_imgs, train_masks, train_depths, train_ids = load_data_from_metadata(
        train_meta, load_cached_data, "train"
    )
    val_imgs, val_masks, val_depths, val_ids = load_data_from_metadata(
        val_meta, load_cached_data, "val"
    )

    # Calculate Depth Statistics from Training Data
    d_mean = np.mean(train_depths)
    d_std = np.std(train_depths)
    depth_stats = (d_mean, d_std)

    # Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transform=get_transforms("train"),
        depth_stats=depth_stats,
    )

    val_dataset = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transform=get_transforms("val"),
        depth_stats=depth_stats,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, depth_stats


def get_test_loader(
    batch_size=32, num_workers=2, depth_stats=None, load_cached_data=True
):
    """
    Creates the test dataloader.
    """
    test_meta = "./metadata/test.csv"

    # Load Data (Masks will be zeros)
    test_imgs, test_masks, test_depths, test_ids = load_data_from_metadata(
        test_meta, load_cached_data, "test"
    )

    # Use provided depth stats or default
    if depth_stats is None:
        depth_stats = (0.0, 1.0)

    test_dataset = SaltDataset(
        test_imgs,
        test_masks,
        test_depths,
        test_ids,
        transform=get_transforms("test"),
        depth_stats=depth_stats,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
