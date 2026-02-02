import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    CACHE_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NORM_MEAN,
    NORM_STD,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import set_seed


def get_transforms(split="train"):
    """
    Returns the data augmentation and normalization pipeline.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus identification task.
    """

    def __init__(self, images, labels=None, transform=None, ids=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray, optional): Array of labels (N,).
            transform (callable, optional): Transform to be applied on a sample.
            ids (np.ndarray, optional): Array of image IDs (filenames).
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.ids = ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C) in RGB format
        img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        # For test set or unlabeled data, return dummy label
        return img, torch.tensor(0.0)


def _load_raw_data(metadata_path):
    """
    Helper function to load raw images from disk based on metadata CSV.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    n_samples = len(df)
    # Pre-allocate array for efficiency
    images = np.zeros((n_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    labels = np.zeros(n_samples, dtype=np.float32)
    ids = []

    for i, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Read image using OpenCV
        img = cv2.imread(full_path)

        if img is None:
            # Should not happen based on metadata verification, but handle safely
            img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[i] = img
        labels[i] = row["has_cactus"]
        ids.append(row["id"])

    return images, labels, np.array(ids)


def load_dataset_with_caching(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads dataset with strict caching logic.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_imgs.npy")
    lbl_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_lbls.npy")
    id_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(img_cache_path)
        and os.path.exists(lbl_cache_path)
        and os.path.exists(id_cache_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached data for {cache_prefix}...")
        images = np.load(img_cache_path)
        labels = np.load(lbl_cache_path)
        ids = np.load(id_cache_path)
    else:
        print(f"Processing raw data for {cache_prefix}...")
        images, labels, ids = _load_raw_data(metadata_path)

        # Save to cache
        print(f"Saving data to cache for {cache_prefix}...")
        np.save(img_cache_path, images)
        np.save(lbl_cache_path, labels)
        np.save(id_cache_path, ids)

    return images, labels, ids


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for the pipeline.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        dict: Contains 'train', 'val', 'test' DataLoaders and 'test_ids'.
    """
    # Load Data
    train_imgs, train_lbls, train_ids = load_dataset_with_caching(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_dataset_with_caching(
        VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_dataset_with_caching(
        TEST_METADATA_PATH, "test", load_cached_data
    )

    # Apply Debugging Limit if specified
    if DEBUG_SAMPLE_SIZE is not None:
        print(
            f"Debugging mode enabled. Limiting dataset to {DEBUG_SAMPLE_SIZE} samples."
        )
        train_imgs = train_imgs[:DEBUG_SAMPLE_SIZE]
        train_lbls = train_lbls[:DEBUG_SAMPLE_SIZE]
        val_imgs = val_imgs[:DEBUG_SAMPLE_SIZE]
        val_lbls = val_lbls[:DEBUG_SAMPLE_SIZE]
        test_imgs = test_imgs[:DEBUG_SAMPLE_SIZE]
        test_ids = test_ids[:DEBUG_SAMPLE_SIZE]

    # Initialize Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train"), ids=train_ids
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, transform=get_transforms("val"), ids=val_ids
    )
    # For test set, pass None for labels to indicate inference mode
    test_dataset = CactusDataset(
        test_imgs, None, transform=get_transforms("test"), ids=test_ids
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Helps with BatchNorm stability and Mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "test_ids": test_ids,
    }
