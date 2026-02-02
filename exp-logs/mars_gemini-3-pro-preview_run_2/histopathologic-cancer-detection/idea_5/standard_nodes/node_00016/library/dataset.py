import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase: str):
    """
    Constructs the Albumentations transform pipeline based on the 'Augment-then-Crop' strategy.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composition of transforms.
    """
    transforms = []

    # --- Training Augmentations (Applied to full 96x96 image) ---
    if phase == "train":
        transforms.extend(
            [
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Color Augmentations (Intensity Statistics)
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
            ]
        )

    # --- Common Preprocessing (Applied to all phases) ---
    transforms.extend(
        [
            # Contextual Crop: 96x96 -> 64x64
            # Preserves 32x32 ROI + 16px context buffer
            A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
            # Normalization using Dataset-Specific Stats
            A.Normalize(
                mean=Config.DATASET_MEAN,
                std=Config.DATASET_STD,
                max_pixel_value=255.0,
            ),
            # Convert to PyTorch Tensor
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def _load_and_cache_data(metadata_path: str, cache_name: str, load_cached_data: bool):
    """
    Internal function to load images from disk, process them, and cache as .npy files.

    Args:
        metadata_path (str): Path to the CSV metadata file.
        cache_name (str): Identifier for the dataset split (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_array, labels_array)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Handle Debug Mode suffixes to avoid polluting full cache
    suffix = "_debug" if Config.DEBUG else ""
    images_cache_path = os.path.join(
        Config.CACHE_DIR, f"{cache_name}_images{suffix}.npy"
    )
    labels_cache_path = os.path.join(
        Config.CACHE_DIR, f"{cache_name}_labels{suffix}.npy"
    )

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(images_cache_path)
        and os.path.exists(labels_cache_path)
    ):
        print(f"Loading {cache_name} set from cache: {images_cache_path}")
        try:
            images = np.load(images_cache_path)
            labels = np.load(labels_cache_path)
            return images, labels
        except Exception as e:
            print(f"Failed to load cache ({e}). Re-processing...")

    # 2. Process from scratch
    print(f"Processing {cache_name} set from raw images...")

    # Load Metadata
    df = pd.read_csv(metadata_path)

    # Debug Limiter
    if Config.DEBUG:
        print(
            f"DEBUG MODE: Limiting {cache_name} to {Config.DEBUG_DATA_LIMIT} samples."
        )
        df = df.head(Config.DEBUG_DATA_LIMIT)

    n_samples = len(df)

    # Pre-allocate arrays for memory efficiency
    # Images are stored as uint8 (0-255) to save space before normalization
    images = np.empty(
        (n_samples, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
    )
    labels = np.empty((n_samples,), dtype=np.float32)

    for idx, row in df.iterrows():
        # Construct absolute path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read Image
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for corrupt/missing images (though verification passed)
            # Create a black image to maintain array shape
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[idx] = img
        labels[idx] = float(row["label"])

    # Save to cache
    print(f"Saving {cache_name} set to cache...")
    np.save(images_cache_path, images)
    np.save(labels_cache_path, labels)

    return images, labels


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for Pathology Patches.
    Serves images from memory and applies Albumentations transforms.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transforms=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, C) containing RGB images.
            labels (np.ndarray): Array of shape (N,) containing binary labels.
            transforms (A.Compose): Albumentations transforms to apply.
        """
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return image and label
        # Label is returned as a float tensor for BCEWithLogitsLoss
        return image, torch.tensor([label], dtype=torch.float32)


def create_datasets(load_cached_data: bool = True):
    """
    Factory function to create Train, Validation, and Test datasets.
    Handles caching logic internally.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load Data Arrays
    train_imgs, train_lbls = _load_and_cache_data(
        Config.TRAIN_META_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls = _load_and_cache_data(
        Config.VAL_META_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls = _load_and_cache_data(
        Config.TEST_META_PATH, "test", load_cached_data
    )

    # Instantiate Datasets with appropriate transforms
    train_dataset = PathologyDataset(train_imgs, train_lbls, get_transforms("train"))
    val_dataset = PathologyDataset(val_imgs, val_lbls, get_transforms("val"))
    test_dataset = PathologyDataset(
        test_imgs, test_lbls, get_transforms("test")  # Test uses same transforms as Val
    )

    return train_dataset, val_dataset, test_dataset
