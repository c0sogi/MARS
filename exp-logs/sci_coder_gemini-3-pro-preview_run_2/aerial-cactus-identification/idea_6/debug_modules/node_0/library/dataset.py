import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    IMAGE_SIZE,
)


def get_transforms(phase: str):
    """
    Returns the data augmentation and normalization pipeline using Albumentations.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    # Normalization parameters to scale [0, 255] -> [0, 1]
    # We use mean=0, std=1 with max_pixel_value=255.0 which effectively divides by 255.
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, only normalize
        return A.Compose(
            [
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def load_processed_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata, reading images and caching them as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cached files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_npy, labels_npy, ids_npy)
    """
    # Define cache paths
    images_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_ids.npy")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(images_cache_path)
            and os.path.exists(labels_cache_path)
            and os.path.exists(ids_cache_path)
        ):

            # print(f"Loading {cache_prefix} data from cache...")
            images = np.load(images_cache_path)
            labels = np.load(labels_cache_path)
            ids = np.load(ids_cache_path)
            return images, labels, ids

    # If cache miss or force reload, process from scratch
    # print(f"Processing {cache_prefix} data from scratch...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays
    num_samples = len(df)
    # Images are 32x32x3
    images = np.zeros((num_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    labels = np.zeros(num_samples, dtype=np.float32)
    ids = np.array(df["id"].values)

    for i, row in df.iterrows():
        # Metadata file_path is relative to INPUT_DIR
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images (should not happen given metadata check)
            img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[i] = img
        labels[i] = row["has_cactus"]

    # Save to cache
    np.save(images_cache_path, images)
    np.save(labels_cache_path, labels)
    np.save(ids_cache_path, ids)

    return images, labels, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            ids (np.ndarray): Array of IDs (N,).
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image tensor, label tensor, and ID
        return image, torch.tensor(label, dtype=torch.float32), img_id


def get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
    max_samples=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached .npy files.
        max_samples (int, optional): Limit dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Load Data
    train_imgs, train_lbls, train_ids = load_processed_data(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_processed_data(
        VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_processed_data(
        TEST_METADATA_PATH, "test", load_cached_data
    )

    # Apply debugging limit if requested
    if max_samples is not None:
        train_imgs = train_imgs[:max_samples]
        train_lbls = train_lbls[:max_samples]
        train_ids = train_ids[:max_samples]

        val_imgs = val_imgs[:max_samples]
        val_lbls = val_lbls[:max_samples]
        val_ids = val_ids[:max_samples]

        test_imgs = test_imgs[:max_samples]
        test_lbls = test_lbls[:max_samples]
        test_ids = test_ids[:max_samples]

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
