import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import process_dataset


def get_transforms(data_split, img_size=512):
    """
    Returns the Albumentations transformation pipeline for a given data split.

    Args:
        data_split (str): One of 'train', 'val', or 'test'.
        img_size (int): Target spatial dimension for the images.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if data_split == "train":
        return A.Compose(
            [
                # Resize is technically redundant if process_dataset handles it,
                # but ensures safety if upstream logic changes.
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Label-Consistent CoarseDropout
                # mask_fill_value=0 ensures that if an opacity is occluded in the image,
                # it is also removed from the mask to prevent label noise.
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(img_size * 0.1),  # ~10% of image height
                    max_width=int(img_size * 0.1),  # ~10% of image width
                    min_holes=1,
                    fill_value=0,  # Black pixels in image
                    mask_fill_value=0,  # Remove label in mask
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (Deterministic)
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-COVID19 Detection.
    Wraps pre-loaded numpy arrays and applies augmentations.
    """

    def __init__(self, images, masks, labels, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, 3).
            masks (np.ndarray): Array of masks (N, H, W, 1).
            labels (np.ndarray): Array of one-hot labels (N, 4).
            transforms (A.Compose): Albumentations transforms.
        """
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask is float for BCE loss (C, H, W)
        # Albumentations ToTensorV2 converts HWC to CHW
        mask = mask.float()

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, mask, label


def get_loaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    test_csv_path=Config.TEST_CSV,
    img_size=Config.IMG_SIZE,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    load_cached_data=True,
):
    """
    Constructs and returns DataLoaders for train, val, and test sets.

    Args:
        train_csv_path (str): Path to training metadata CSV.
        val_csv_path (str): Path to validation metadata CSV.
        test_csv_path (str): Path to test metadata CSV.
        img_size (int): Target image resolution.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, loads only a small subset of data.
        debug_sample_size (int): Number of samples to load if debug is True.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # 1. Load Metadata
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Determine sample size for debugging
    sample_n = debug_sample_size if debug else None

    # 2. Process Data (Load from DICOM or Cache)
    # process_dataset handles the heavy lifting: DICOM reading, resizing, mask generation, and caching.

    # Train Data
    train_images, train_masks, train_labels = process_dataset(
        train_df,
        subset_name="train",
        load_cached_data=load_cached_data,
        img_size=img_size,
        sample_size=sample_n,
    )

    # Validation Data
    val_images, val_masks, val_labels = process_dataset(
        val_df,
        subset_name="val",
        load_cached_data=load_cached_data,
        img_size=img_size,
        sample_size=sample_n,
    )

    # Test Data
    test_images, test_masks, test_labels = process_dataset(
        test_df,
        subset_name="test",
        load_cached_data=load_cached_data,
        img_size=img_size,
        sample_size=sample_n,
    )

    # 3. Instantiate Datasets
    train_dataset = SIIMDataset(
        train_images,
        train_masks,
        train_labels,
        transforms=get_transforms("train", img_size),
    )

    val_dataset = SIIMDataset(
        val_images, val_masks, val_labels, transforms=get_transforms("val", img_size)
    )

    test_dataset = SIIMDataset(
        test_images,
        test_masks,
        test_labels,
        transforms=get_transforms("test", img_size),
    )

    # 4. Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
