import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_SHAPE,
)


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Identification.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray = None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in this pipeline)
            # Convert to tensor and normalize to [0, 1] manually
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # For test set, we might return the index or ID if needed,
            # but usually just the image is enough for inference if order is preserved.
            # Here we just return the image.
            return image


def get_transforms(split: str):
    """
    Returns the augmentation pipeline for a specific split.

    Args:
        split (str): 'train', 'val', or 'test'.
    """
    # Note: Input images are uint8.
    # Normalize(mean=0, std=1, max_pixel_value=255.0) scales inputs to [0, 1].

    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=(0, 0, 0),
                    std=(1, 1, 1),
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Only Normalize and ToTensor
        return A.Compose(
            [
                A.Normalize(
                    mean=(0, 0, 0),
                    std=(1, 1, 1),
                    max_pixel_value=255.0,
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )


def load_and_cache_data(metadata_path: str, cache_prefix: str, load_cached_data: bool):
    """
    Loads data from disk or cache.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cached .npy files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
            images: np.ndarray of shape (N, 32, 32, 3)
            labels: np.ndarray of shape (N,) or None
            ids: np.ndarray of shape (N,) containing image IDs
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    images_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(images_cache_path) and os.path.exists(ids_cache_path):
            # Check if labels cache exists (it might not for test set if we didn't save it,
            # though usually we save it even if placeholders)
            # For simplicity, we check labels path existence only if it's not test or if we expect it.
            # We'll just try to load everything.

            try:
                print(f"Loading {cache_prefix} data from cache...")
                images = np.load(images_cache_path)
                ids = np.load(ids_cache_path)

                if os.path.exists(labels_cache_path):
                    labels = np.load(labels_cache_path)
                else:
                    labels = None

                return images, labels, ids
            except Exception as e:
                print(f"Failed to load cache for {cache_prefix}: {e}. Recomputing...")
        else:
            print(f"Cache not found for {cache_prefix}. Processing from scratch...")

    # 2. Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    # Pre-allocate if possible or just append (dataset is small enough for append)
    # 14k images * 32 * 32 * 3 bytes is small (~40MB).

    for _, row in df.iterrows():
        # Construct full path. Metadata contains relative path 'train/id.jpg'
        # INPUT_DIR is './input'
        # file_path is like 'train/...'
        full_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not load image {full_path}. Skipping.")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        id_list.append(row["id"])

        if "has_cactus" in row:
            label_list.append(row["has_cactus"])

    images = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)

    if label_list:
        labels = np.array(label_list, dtype=np.float32)
    else:
        labels = None

    # 3. Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(images_cache_path, images)
    np.save(ids_cache_path, ids)
    if labels is not None:
        np.save(labels_cache_path, labels)

    return images, labels, ids


def get_datasets(load_cached_data: bool = True):
    """
    Main function to get PyTorch datasets for train, val, and test.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, test_ids)
    """
    # Load raw data arrays
    train_imgs, train_lbls, _ = load_and_cache_data(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, _ = load_and_cache_data(
        VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_and_cache_data(
        TEST_METADATA_PATH, "test", load_cached_data
    )

    # Create Datasets with transforms
    train_dataset = CactusDataset(
        images=train_imgs, labels=train_lbls, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(
        images=val_imgs, labels=val_lbls, transform=get_transforms("val")
    )

    # For test dataset, labels might be placeholders, but we don't strictly need them for inference
    # However, passing them keeps the __getitem__ consistent if we wanted to evaluate.
    # For submission, we just need the images.
    test_dataset = CactusDataset(
        images=test_imgs,
        labels=None,  # We ignore labels for test inference
        transform=get_transforms("test"),
    )

    return train_dataset, val_dataset, test_dataset, test_ids
