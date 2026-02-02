import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IDEA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGE_SIZE,
)


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
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
            # Fallback normalization if no transform is provided
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Return a placeholder for test data
            return image, torch.tensor(-1.0, dtype=torch.float32)


def load_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads images and labels, using a caching mechanism to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cached .npy files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # Ensure working directory exists
    os.makedirs(IDEA_DIR, exist_ok=True)

    images_path = os.path.join(IDEA_DIR, f"{cache_prefix}_images.npy")
    labels_path = os.path.join(IDEA_DIR, f"{cache_prefix}_labels.npy")
    ids_path = os.path.join(IDEA_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(images_path) and os.path.exists(ids_path):
        try:
            images = np.load(images_path)
            ids = np.load(ids_path, allow_pickle=True)
            labels = None
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
            return images, labels, ids
        except Exception:
            # If loading fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    df = pd.read_csv(metadata_path)

    images_list = []
    labels_list = []
    ids_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image in RGB
        img = cv2.imread(full_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images_list.append(img)
        ids_list.append(row["id"])

        if "has_cactus" in row:
            labels_list.append(row["has_cactus"])

    images = np.array(images_list, dtype=np.uint8)
    ids = np.array(ids_list)

    # Save to cache
    np.save(images_path, images)
    np.save(ids_path, ids)

    if labels_list:
        labels = np.array(labels_list, dtype=np.float32)
        np.save(labels_path, labels)
    else:
        labels = None

    return images, labels, ids


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Normalize to [0, 1] by dividing by 255.0
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Normalize to [0, 1]
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
):
    """
    Constructs and returns DataLoaders for train, val, and test sets.
    Also returns test_ids for submission generation.
    """
    # Load data
    train_imgs, train_lbls, _ = load_data(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, _ = load_data(VAL_METADATA_PATH, "val", load_cached_data)
    test_imgs, _, test_ids = load_data(TEST_METADATA_PATH, "test", load_cached_data)

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))
    # Note: We pass None for labels to the test dataset
    test_dataset = CactusDataset(test_imgs, None, transform=get_transforms("test"))

    # Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
