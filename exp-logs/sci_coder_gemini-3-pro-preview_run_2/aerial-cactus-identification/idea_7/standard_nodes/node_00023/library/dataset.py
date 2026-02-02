import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    Loads images from numpy arrays (pre-loaded into memory).
    Applies Albumentations transforms for augmentation and tensor conversion.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (A.Compose, optional): Albumentations transformations.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are already RGB numpy arrays
        image = self.images[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.labels is not None:
            label = self.labels[idx]
            # Return float32 for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            return image


def get_transforms(phase: str):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Normalize to [0, 1] by dividing by 255.0 (max_pixel_value)
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Normalize only
        return A.Compose(
            [
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def _load_images_from_metadata(metadata_path, input_dir):
    """
    Helper to load images and labels from a metadata CSV.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images = []
    labels = []
    ids = []

    # Iterate and load
    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Load image with OpenCV
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing images (though metadata check passed)
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images.append(img)
        ids.append(row["id"])

        if "has_cactus" in row:
            labels.append(row["has_cactus"])

    images = np.array(images, dtype=np.uint8)
    ids = np.array(ids)

    # If labels were found, return them as float32 array
    if labels:
        labels = np.array(labels, dtype=np.float32)
        return images, labels, ids
    else:
        return images, None, ids


def process_data(load_cached_data: bool = True):
    """
    Loads data from disk, caching it as .npy files for faster subsequent access.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files first.

    Returns:
        tuple: (train_data, val_data, test_data)
               train_data: (images, labels)
               val_data: (images, labels)
               test_data: (images, ids)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_paths = {
        "train_images": os.path.join(Config.WORKING_DIR, "train_images.npy"),
        "train_labels": os.path.join(Config.WORKING_DIR, "train_labels.npy"),
        "val_images": os.path.join(Config.WORKING_DIR, "val_images.npy"),
        "val_labels": os.path.join(Config.WORKING_DIR, "val_labels.npy"),
        "test_images": os.path.join(Config.WORKING_DIR, "test_images.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_images = np.load(cache_paths["train_images"])
        train_labels = np.load(cache_paths["train_labels"])
        val_images = np.load(cache_paths["val_images"])
        val_labels = np.load(cache_paths["val_labels"])
        test_images = np.load(cache_paths["test_images"])
        test_ids = np.load(cache_paths["test_ids"], allow_pickle=True)
    else:
        print("Processing data from scratch (loading images)...")

        # Load Train
        train_images, train_labels, _ = _load_images_from_metadata(
            Config.TRAIN_METADATA, Config.INPUT_DIR
        )

        # Load Val
        val_images, val_labels, _ = _load_images_from_metadata(
            Config.VAL_METADATA, Config.INPUT_DIR
        )

        # Load Test (Labels in metadata are placeholders, ignore them for dataset, but keep IDs)
        test_images, _, test_ids = _load_images_from_metadata(
            Config.TEST_METADATA, Config.INPUT_DIR
        )

        # Save to cache
        print("Saving data to cache...")
        np.save(cache_paths["train_images"], train_images)
        np.save(cache_paths["train_labels"], train_labels)
        np.save(cache_paths["val_images"], val_images)
        np.save(cache_paths["val_labels"], val_labels)
        np.save(cache_paths["test_images"], test_images)
        np.save(cache_paths["test_ids"], test_ids)

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_ids),
    )
