import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    """

    def __init__(self, images, labels=None, ids=None, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of binary labels (N,).
            ids (np.ndarray, optional): Array of image IDs/filenames (N,).
            transforms (albumentations.Compose, optional): Transformations to apply.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback: Convert to tensor and normalize to [0, 1]
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return (image, label) if labels exist, else just image
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label

        return image


def get_transforms(phase: str):
    """
    Returns the albumentations transformations for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=Config.H_FLIP_PROB),
                A.VerticalFlip(p=Config.V_FLIP_PROB),
                # Normalize pixel values to [0, 1]
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Only Normalize
        return A.Compose(
            [
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def _load_cached_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and images, with caching to .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    images_cache = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_images.npy")
    labels_cache = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")
    ids_cache = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(images_cache) and os.path.exists(ids_cache):
            # Check if labels cache exists (it might not for test set if we didn't save it,
            # though usually we save what we have)
            # For test set, we might have labels in metadata (placeholders),
            # but we check existence of file.

            # If it's a labeled set (train/val) or test set where we saved everything
            if os.path.exists(labels_cache) or "test" in cache_prefix:
                print(f"Loading cached data for {cache_prefix}...")
                images = np.load(images_cache)
                ids = np.load(ids_cache)
                labels = np.load(labels_cache) if os.path.exists(labels_cache) else None
                return images, labels, ids

    # Process from scratch
    print(f"Processing data for {cache_prefix} from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    n_samples = len(df)
    # Pre-allocate array for images: (N, 32, 32, 3)
    images = np.zeros(
        (n_samples, Config.IMAGE_SIZE, Config.IMAGE_SIZE, Config.IN_CHANNELS),
        dtype=np.uint8,
    )
    ids = df["id"].values

    # Check if labels exist in metadata
    if "has_cactus" in df.columns:
        labels = df["has_cactus"].values.astype(np.float32)
    else:
        labels = None

    # Load images
    for i, row in df.iterrows():
        # file_path in metadata is relative to input dir (e.g., "train/xxx.jpg")
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Failed to load image at {full_path}")
            continue

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[i] = img

    # Save to cache
    print(f"Saving cache for {cache_prefix}...")
    np.save(images_cache, images)
    np.save(ids_cache, ids)
    if labels is not None:
        np.save(labels_cache, labels)

    return images, labels, ids


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data
    train_imgs, train_lbls, train_ids = _load_cached_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = _load_cached_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = _load_cached_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Create Datasets
    train_dataset = CactusDataset(
        images=train_imgs,
        labels=train_lbls,
        ids=train_ids,
        transforms=get_transforms("train"),
    )

    val_dataset = CactusDataset(
        images=val_imgs,
        labels=val_lbls,
        ids=val_ids,
        transforms=get_transforms("valid"),
    )

    # For test dataset, we don't pass labels to __getitem__ (returns only image)
    # The labels loaded from metadata are placeholders (0.5) anyway.
    test_dataset = CactusDataset(
        images=test_imgs, labels=None, ids=test_ids, transforms=get_transforms("test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for batch norm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
