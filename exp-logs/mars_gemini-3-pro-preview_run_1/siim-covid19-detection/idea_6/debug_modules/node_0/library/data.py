import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import process_dataset


def get_transforms(data_split="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data_split (str): "train" or "val".
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                # Label-consistent CoarseDropout:
                # mask_fill_value=0 ensures that if an opacity is occluded in the image,
                # it is also removed from the ground truth mask.
                A.CoarseDropout(
                    max_holes=Config.COARSE_DROPOUT_HOLES,
                    max_height=Config.COARSE_DROPOUT_HEIGHT,
                    max_width=Config.COARSE_DROPOUT_WIDTH,
                    min_holes=1,
                    fill_value=0,
                    mask_fill_value=Config.MASK_FILL_VALUE,
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
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class SIIMDataset(Dataset):
    """
    PyTorch Dataset for SIIM-FISABIO-RSNA COVID-19 Detection.
    Operates on pre-loaded numpy arrays for maximum throughput.
    """

    def __init__(self, images, masks, labels, transforms=None):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # images: (H, W, 3) uint8
        image = self.images[idx]
        # masks: (H, W, 1) float32
        mask = self.masks[idx]
        # labels: (4,) float32
        label = self.labels[idx]

        if self.transforms:
            # Albumentations expects mask to be (H, W) or (H, W, C)
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Post-processing for Mask
        # Image is already converted to Tensor (C, H, W) by ToTensorV2
        # Mask needs to be converted to Tensor and (C, H, W)
        if not isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask[None, :, :]  # (H, W) -> (1, H, W)
            elif mask.ndim == 3:
                mask = np.transpose(mask, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            mask = torch.from_numpy(mask)

        # Ensure correct types
        return image, mask.float(), torch.tensor(label).float()


def prepare_train_val_loaders(debug=False):
    """
    Orchestrates the loading of training and validation data.
    Uses library.utils.process_dataset to handle caching of heavy arrays.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        print("Debug mode: Reduced dataset size.")

    # 2. Process Data (Load from cache or compute)
    cache_dir = Config.WORKING_DIR

    print("Preparing Train Data...")
    train_images, train_masks, train_labels = process_dataset(
        train_df,
        cache_dir,
        image_size=Config.IMG_SIZE,
        load_cached_data=True,
        split_name="train",
    )

    print("Preparing Val Data...")
    val_images, val_masks, val_labels = process_dataset(
        val_df,
        cache_dir,
        image_size=Config.IMG_SIZE,
        load_cached_data=True,
        split_name="val",
    )

    # 3. Create Datasets
    train_dataset = SIIMDataset(
        train_images, train_masks, train_labels, transforms=get_transforms("train")
    )

    val_dataset = SIIMDataset(
        val_images, val_masks, val_labels, transforms=get_transforms("val")
    )

    # 4. Create Loaders
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

    return train_loader, val_loader


def prepare_test_loader(test_df):
    """
    Prepares the test data loader.
    """
    cache_dir = Config.WORKING_DIR

    # Process test data (handles missing labels/boxes gracefully)
    images, masks, labels = process_dataset(
        test_df,
        cache_dir,
        image_size=Config.IMG_SIZE,
        load_cached_data=True,
        split_name="test",
    )

    dataset = SIIMDataset(
        images,
        masks,
        labels,
        transforms=get_transforms("val"),  # Use Val transforms (Resize+Norm)
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
